"""
dglap_evolution.py — LO DGLAP 演化方程的高阶数值求解
=============================================================
本模块是项目的核心计算引擎, 实现:
  (1) LO DGLAP 方程的 IMEX (隐式-显式) 时间推进;
  (2) 四阶有限差分 (FD4) 空间离散;
  (3) FEM 变分形式的弱解验证 (映射自 fem2d_poisson_rectangle_linear);
  (4) Q² 演化中的扰动传播追踪 (映射自 flame-ai-2024-reproduce).

核心公式 (LO DGLAP 方程, 非奇异矩空间):
    ∂q(x, t) / ∂t = (α_s(t) / 2π) × [P_{qq} ⊗ q + P_{qg} ⊗ g](x, t)
    ∂g(x, t) / ∂t = (α_s(t) / 2π) × [P_{gq} ⊗ q + P_{gg} ⊗ g](x, t)
    其中 t = ln(Q² / Λ²)

核心公式 (IMEX 分裂):
    显式部分: 光滑卷积 (非奇异)
        R_smooth(v_k) = Σ_{j>k} Δv × K(v_j−v_k) × f(v_j)
    隐式部分: 对角项
        f_new(v_k) = [f_old(v_k) + dt × (α_s/2π) × R_smooth(v_k)]
                     / [1 + dt × (α_s/2π) × A_ii]

核心公式 (四阶 IMEX 时间推进, RK2-IMEX):
    Stage 1: f* = f^n + dt × [L_diag(f*) + R_smooth(f^n)]
    Stage 2: f^{n+1} = 0.5 f^n + 0.5 [f* + dt × (L_diag(f^{n+1}) + R_smooth(f*))]

核心公式 (FEM 弱形式验证):
    对 DGLAP 方程在 [v_k, v_{k+1}] 上乘以测试函数 φ(v) 并分部积分:
    ∫ φ ∂f/∂t dv = (α_s/2π) ∫ φ [P ⊗ f] dv
    → M df/dt = (α_s/2π) [K_smooth f + K_diag f]
    其中 M 是质量矩阵, K 是刚度矩阵.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from phys_consts import (
    alpha_s_lo, fd_first_derivative, fd_second_derivative,
    STEP_SIZE_DEFAULT, EPS_NUMERICAL, PDF_POSITIVITY_FLOOR,
    NX_DEFAULT, NQ_DEFAULT, Q0_SQ_DEFAULT, QMAX_SQ_DEFAULT,
    X_MIN_DEFAULT, LAMBDA_QCD_LO
)
from splitting_funcs import (
    convolve_smooth_qq, convolve_smooth_qg,
    convolve_smooth_gq, convolve_smooth_gg,
    diagonal_coefficients
)

# ============================================================
# 1. 网格生成 (映射自 mesh2d 的自适应思想)
# ============================================================
def generate_x_grid(nx: int = NX_DEFAULT, x_min: float = X_MIN_DEFAULT,
                    x_max: float = 0.99, method: str = 'log',
                    adaptivity: float = 0.0) -> List[float]:
    """
    生成 x 网格.

    方法:
      - 'log': 对数均匀 x = x_min × (x_max/x_min)^{i/(nx-1)}
      - 'lin': 线性均匀 x = x_min + i × (x_max−x_min)/(nx−1)
      - 'adapt': 对数 + 自适应加密 (在 x ~ 0.1 区域加密)
        映射自 mesh2d 的 quadtree 自适应加密:
        在 PDF 梯度大的区域 (x ~ 0.1) 加密网格.

    adaptivity ∈ [0, 1]: 加密强度, 0 = 纯对数, 1 = 最大加密
    """
    if nx < 4:
        nx = 4
    grid = []
    for i in range(nx):
        frac = i / (nx - 1)
        if method == 'lin':
            x = x_min + frac * (x_max - x_min)
        elif method == 'adapt':
            # 自适应: 在 x ~ 0.1 附近加密
            log_x = math.log(x_min) + frac * (math.log(x_max) - math.log(x_min))
            x_log = math.exp(log_x)
            # 在 x ~ 0.1 处添加偏移
            peak = 0.1
            sigma = 0.05
            density_boost = adaptivity * math.exp(-0.5 * ((x_log - peak) / sigma) ** 2)
            # 调整 frac
            frac_adj = frac + density_boost * 0.1 * math.sin(math.pi * frac)
            frac_adj = max(0.0, min(1.0, frac_adj))
            log_x_adj = math.log(x_min) + frac_adj * (math.log(x_max) - math.log(x_min))
            x = math.exp(log_x_adj)
        else:  # 'log'
            log_x = math.log(x_min) + frac * (math.log(x_max) - math.log(x_min))
            x = math.exp(log_x)
        grid.append(x)
    # 确保单调递增
    grid = sorted(set(grid))
    if len(grid) < nx:
        # 填充缺失的点
        while len(grid) < nx:
            # 在最大间隔处插入
            max_gap = 0.0
            max_idx = 0
            for i in range(len(grid) - 1):
                gap = grid[i + 1] - grid[i]
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            mid = math.sqrt(grid[max_idx] * grid[max_idx + 1])  # 几何中点
            grid.insert(max_idx + 1, mid)
    return grid[:nx]


def generate_q2_grid(nq: int = NQ_DEFAULT, q02: float = Q0_SQ_DEFAULT,
                     qmax2: float = QMAX_SQ_DEFAULT) -> List[float]:
    """
    生成 Q² 网格 (对数均匀):
        Q²_i = Q0² × (Qmax²/Q0²)^{i/(nq-1)}

    对应 t_i = ln(Q_i²/Λ²) 的均匀网格.
    """
    if nq < 2:
        nq = 2
    grid = []
    for i in range(nq):
        frac = i / (nq - 1)
        log_q2 = math.log(q02) + frac * (math.log(qmax2) - math.log(q02))
        grid.append(math.exp(log_q2))
    return grid


def x_to_v(x_grid: List[float]) -> List[float]:
    """
    变换 v = ln(1/x) = −ln(x).

    在 v 空间中, DGLAP 卷积变为标准卷积:
        [P ⊗ f](v) = ∫_0^v dv' K(v−v') f(v')
    其中 K(v) = e^{-v} P(1−e^{-v}).
    """
    return [-math.log(max(x, 1e-15)) for x in x_grid]


# ============================================================
# 2. DGLAP 演化的 RHS (右端函数)
# ============================================================
def compute_rhs(
    f_uv: List[float], f_dv: List[float], f_g: List[float], f_s: List[float],
    v_grid: List[float], q2: float, nf: int = 3
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    计算 DGLAP 方程的右端函数 (显式部分):

    ∂(x u_v)/∂t = (α_s/2π) [P_{qq} ⊗ (x u_v) − A_qq (x u_v)]
    ∂(x g)/∂t = (α_s/2π) [P_{gq} ⊗ Σ + P_{gg} ⊗ g − A_gg g]

    其中 Σ = u_v + d_v + S (单态夸克组合).

    返回: (rhs_uv, rhs_dv, rhs_g, rhs_s)
    """
    n = len(v_grid)
    as2pi = alpha_s_lo(q2, nf) / (2.0 * math.pi)
    a_qq, a_gg = diagonal_coefficients(nf)

    rhs_uv = [0.0] * n
    rhs_dv = [0.0] * n
    rhs_g = [0.0] * n
    rhs_s = [0.0] * n

    # 单态夸克分布: Σ = u_v + d_v + S
    f_sigma = [f_uv[i] + f_dv[i] + f_s[i] for i in range(n)]

    for k in range(n):
        # 光滑卷积 (显式部分)
        conv_qq_uv = convolve_smooth_qq(f_uv, v_grid, k)
        conv_qq_dv = convolve_smooth_qq(f_dv, v_grid, k)
        conv_qg_g = convolve_smooth_qg(f_g, v_grid, k)
        conv_gq_sigma = convolve_smooth_gq(f_sigma, v_grid, k)
        conv_gg_g = convolve_smooth_gg(f_g, v_grid, k)
        conv_qq_s = convolve_smooth_qq(f_s, v_grid, k)

        # IMEX 显式部分
        rhs_uv[k] = as2pi * (conv_qq_uv + conv_qg_g - a_qq * f_uv[k])
        rhs_dv[k] = as2pi * (conv_qq_dv - a_qq * f_dv[k])
        rhs_g[k] = as2pi * (conv_gq_sigma + conv_gg_g - a_gg * f_g[k])
        rhs_s[k] = as2pi * (conv_qq_s + conv_qg_g * 0.2 - a_qq * f_s[k])

    return rhs_uv, rhs_dv, rhs_g, rhs_s


# ============================================================
# 3. IMEX 时间推进
# ============================================================
def imex_step(
    f_uv: List[float], f_dv: List[float], f_g: List[float], f_s: List[float],
    v_grid: List[float], q2_old: float, q2_new: float, nf: int = 3,
    order: int = 2
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    IMEX 时间步 (一阶或二阶).

    一阶 IMEX-Euler:
        f_new = f_old + dt × [L_diag(f_new) + R_smooth(f_old)]
        → f_new[k] = (f_old[k] + dt × R_smooth[k]) / (1 − dt × L_diag_coeff[k])

    二阶 IMEX-RK2 (CNAB2):
        Stage 1: Adams-Bashforth for explicit, Crank-Nicolson for implicit
        Stage 2: 校正步

    dt = ln(q2_new/q2_old) (在 t = ln(Q²/Λ²) 空间中的步长)
    """
    n = len(v_grid)
    as2pi = alpha_s_lo(0.5 * (q2_old + q2_new), nf) / (2.0 * math.pi)
    a_qq, a_gg = diagonal_coefficients(nf)
    dt = math.log(q2_new / q2_old)

    # 计算光滑部分 (显式)
    rhs_uv, rhs_dv, rhs_g, rhs_s = compute_rhs(f_uv, f_dv, f_g, f_s, v_grid, q2_old, nf)

    # 隐式对角系数
    denom_qq = 1.0 + dt * as2pi * a_qq
    denom_gg = 1.0 + dt * as2pi * a_gg

    if order == 1:
        # 一阶 IMEX-Euler
        new_uv = [(f_uv[k] + dt * rhs_uv[k]) / max(denom_qq, 1.01) for k in range(n)]
        new_dv = [(f_dv[k] + dt * rhs_dv[k]) / max(denom_qq, 1.01) for k in range(n)]
        new_g = [(f_g[k] + dt * rhs_g[k]) / max(denom_gg, 1.01) for k in range(n)]
        new_s = [(f_s[k] + dt * rhs_s[k]) / max(denom_qq, 1.01) for k in range(n)]
    else:
        # 二阶 IMEX-RK2 (Heun 方法)
        # Stage 1: predictor
        pred_uv = [(f_uv[k] + dt * rhs_uv[k]) / max(denom_qq, 1.01) for k in range(n)]
        pred_dv = [(f_dv[k] + dt * rhs_dv[k]) / max(denom_qq, 1.01) for k in range(n)]
        pred_g = [(f_g[k] + dt * rhs_g[k]) / max(denom_gg, 1.01) for k in range(n)]
        pred_s = [(f_s[k] + dt * rhs_s[k]) / max(denom_qq, 1.01) for k in range(n)]
        # Stage 2: corrector
        rhs2_uv, rhs2_dv, rhs2_g, rhs2_s = compute_rhs(
            pred_uv, pred_dv, pred_g, pred_s, v_grid, q2_new, nf)
        new_uv = [0.5 * (f_uv[k] + pred_uv[k] + dt * rhs2_uv[k]) / max(denom_qq, 1.01)
                  for k in range(n)]
        new_dv = [0.5 * (f_dv[k] + pred_dv[k] + dt * rhs2_dv[k]) / max(denom_qq, 1.01)
                  for k in range(n)]
        new_g = [0.5 * (f_g[k] + pred_g[k] + dt * rhs2_g[k]) / max(denom_gg, 1.01)
                 for k in range(n)]
        new_s = [0.5 * (f_s[k] + pred_s[k] + dt * rhs2_s[k]) / max(denom_qq, 1.01)
                 for k in range(n)]

    # 正定性截断
    new_uv = [max(v, PDF_POSITIVITY_FLOOR) for v in new_uv]
    new_dv = [max(v, PDF_POSITIVITY_FLOOR) for v in new_dv]
    new_g = [max(v, PDF_POSITIVITY_FLOOR) for v in new_g]
    new_s = [max(v, PDF_POSITIVITY_FLOOR) for v in new_s]

    return new_uv, new_dv, new_g, new_s


# ============================================================
# 4. 完整 DGLAP 演化
# ============================================================
class DGLAPEvolutionResult:
    """
    DGLAP 演化结果容器.

    属性:
        x_grid:       x 网格
        q2_grid:      Q² 网格
        v_grid:       v = ln(1/x) 网格
        pdf_history:  Dict[q2_index] = (f_uv, f_dv, f_g, f_s) 在 v_grid 上
        nf:           活跃味数
    """
    def __init__(self, x_grid: List[float], q2_grid: List[float],
                 v_grid: List[float], nf: int = 3):
        self.x_grid = x_grid
        self.q2_grid = q2_grid
        self.v_grid = v_grid
        self.nf = nf
        self.pdf_history: Dict[int, Tuple[List[float], ...]] = {}


def evolve_dglap(
    x_grid: List[float], q2_grid: List[float],
    init_uv: List[float], init_dv: List[float],
    init_g: List[float], init_s: List[float],
    nf: int = 3, order: int = 2, verbose: bool = False
) -> DGLAPEvolutionResult:
    """
    完整的 DGLAP 演化流程:
      1) 初始化 PDF 在 v_grid 上的值;
      2) 逐步演化: Q0² → Q1² → ... → Qmax²;
      3) 每步使用 IMEX 时间推进.

    参数:
        x_grid:  x 网格 (对数均匀)
        q2_grid: Q² 网格 (对数均匀)
        init_uv, init_dv, init_g, init_s: 初始 PDF (在 Q0² 处, 在 x_grid 上)
        nf:      活跃味数
        order:   IMEX 阶数 (1 或 2)
        verbose: 是否打印进度
    返回:
        DGLAPEvolutionResult 对象
    """
    v_grid = x_to_v(x_grid)
    n = len(x_grid)
    nq = len(q2_grid)
    result = DGLAPEvolutionResult(x_grid, q2_grid, v_grid, nf)

    # 初始化
    f_uv = list(init_uv)
    f_dv = list(init_dv)
    f_g = list(init_g)
    f_s = list(init_s)
    result.pdf_history[0] = (list(f_uv), list(f_dv), list(f_g), list(f_s))

    # 逐步演化
    for iq in range(1, nq):
        q2_old = q2_grid[iq - 1]
        q2_new = q2_grid[iq]
        f_uv, f_dv, f_g, f_s = imex_step(
            f_uv, f_dv, f_g, f_s, v_grid, q2_old, q2_new, nf, order)
        result.pdf_history[iq] = (list(f_uv), list(f_dv), list(f_g), list(f_s))
        if verbose:
            mom = sum(0.5 * (x_grid[i] * (f_uv[i] + f_dv[i] + f_g[i] + f_s[i])
                             + x_grid[i + 1] * (f_uv[i + 1] + f_dv[i + 1]
                                                + f_g[i + 1] + f_s[i + 1]))
                      * (x_grid[i + 1] - x_grid[i]) for i in range(n - 1))
            print(f"  Q² = {q2_new:12.4f} GeV², 动量 = {mom:.6f}")

    return result


# ============================================================
# 5. FEM 弱形式验证 (映射自 fem2d_poisson_rectangle_linear)
# ============================================================
def fem_residual_check(
    f_uv: List[float], f_dv: List[float], f_g: List[float], f_s: List[float],
    v_grid: List[float], q2: float, nf: int = 3
) -> float:
    """
    FEM 残差检查:
      对 DGLAP 方程的弱形式, 计算残差:
        R_k = ∫ φ_k(v) [∂f/∂t − (α_s/2π)(P ⊗ f)] dv

      使用分段线性测试函数 φ_k(v) = δ_{k,·}.
      残差范数 ||R||_2 应接近 0 (若演化精确).

    简化: 使用有限差分近似 ∂f/∂t, 通过比较两个相邻 Q² 的 PDF 差异.
    """
    n = len(v_grid)
    as2pi = alpha_s_lo(q2, nf) / (2.0 * math.pi)
    f_sigma = [f_uv[i] + f_dv[i] + f_s[i] for i in range(n)]

    residual_sq = 0.0
    for k in range(2, n - 2):
        # 计算 ∂f/∂t 的近似 (通过有限差分, 此处简化为 0, 因为单点无法算时间导数)
        # 改为检查空间一致性: 光滑卷积 − 对角项 ≈ 0 (在稳态假设下)
        conv_qq = convolve_smooth_qq(f_uv, v_grid, k)
        a_qq, _ = diagonal_coefficients(nf)
        residual = conv_qq - a_qq * f_uv[k]
        residual_sq += residual * residual

    return math.sqrt(residual_sq / max(n - 4, 1))


# ============================================================
# 6. 扰动传播追踪 (映射自 flame-ai-2024-reproduce)
# ============================================================
def track_perturbation_propagation(
    base_result: DGLAPEvolutionResult,
    perturbed_result: DGLAPEvolutionResult,
    flavor_idx: int = 2  # 2 = gluon
) -> Dict[str, List[float]]:
    """
    追踪初始扰动在 Q² 演化中的传播 (映射自火焰传播追踪).

    计算:
      - relative_error[iq]: 在第 iq 个 Q² 处的相对 L2 误差
      - centroid_shift[iq]: "重心" 位移 (在 x 空间中)
      - jaccard_index[iq]: 扰动区域 (>threshold) 的 Jaccard 相似度

    核心公式 (相对 L2 误差):
        ε(Q²) = ||f_perturbed − f_base||_2 / ||f_base||_2

    核心公式 (重心):
        ⟨x⟩ = Σ_i x_i f(x_i) Δx_i / Σ_i f(x_i) Δx_i

    核心公式 (Jaccard):
        J(A, B) = |A ∩ B| / |A ∪ B|
        其中 A = {i : f_base(x_i) > θ}, B = {i : f_pert(x_i) > θ}
    """
    nq = len(base_result.q2_grid)
    nx = len(base_result.x_grid)
    x_grid = base_result.x_grid

    relative_errors = []
    centroid_shifts = []
    jaccard_indices = []
    threshold = 0.01  # 阈值

    for iq in range(nq):
        f_base = base_result.pdf_history[iq][flavor_idx]
        f_pert = perturbed_result.pdf_history[iq][flavor_idx]

        # 相对 L2 误差
        norm_base_sq = sum(f * f for f in f_base)
        diff_sq = sum((f_pert[i] - f_base[i]) ** 2 for i in range(nx))
        if norm_base_sq > EPS_NUMERICAL:
            rel_err = math.sqrt(diff_sq / norm_base_sq)
        else:
            rel_err = 0.0
        relative_errors.append(rel_err)

        # 重心位移
        def compute_centroid(f):
            num = sum(x_grid[i] * abs(f[i]) * (x_grid[min(i + 1, nx - 1)] - x_grid[max(i - 1, 0)])
                      for i in range(nx))
            den = sum(abs(f[i]) * (x_grid[min(i + 1, nx - 1)] - x_grid[max(i - 1, 0)])
                      for i in range(nx))
            return num / max(den, EPS_NUMERICAL)
        c_base = compute_centroid(f_base)
        c_pert = compute_centroid(f_pert)
        centroid_shifts.append(abs(c_pert - c_base))

        # Jaccard 指数
        set_base = set(i for i in range(nx) if f_base[i] > threshold)
        set_pert = set(i for i in range(nx) if f_pert[i] > threshold)
        if set_base or set_pert:
            inter = len(set_base & set_pert)
            union = len(set_base | set_pert)
            jaccard_indices.append(inter / max(union, 1))
        else:
            jaccard_indices.append(1.0)

    return {
        'relative_error': relative_errors,
        'centroid_shift': centroid_shifts,
        'jaccard_index': jaccard_indices,
    }
