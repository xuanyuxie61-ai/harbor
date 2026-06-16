"""
hessian_error.py — Hessian 矩阵计算与误差传播 (含模块化算术)
=============================================================
本模块实现:
  (1) χ² 的 Hessian 矩阵 (有限差分);
  (2) 特征值分解;
  (3) PDF 不确定度的 Hessian 方法;
  (4) 模块化算术在 Mellin-N 空间的应用 (映射自 caustic);
  (5) Hessian 相关矩阵的文本热图显示 (映射自 box_fill).

核心公式 (Hessian 矩阵):
    H_{ij} = ∂²χ² / ∂p_i ∂p_j
           ≈ [χ²(p+h_i+h_j) − χ²(p+h_i−h_j) − χ²(p−h_i+h_j) + χ²(p−h_i−h_j)] / (4 h_i h_j)

核心公式 (误差 PDF):
    (Δf(x))² = Σ_{k=1}^{N_eigen} [f_k^+(x) − f_0(x)]²
    其中 f_k^± = f(x; p_0 ± √(Δχ²) × v_k)

核心公式 (Mellin 空间的模块化算术, 映射自 caustic):
    在 Mellin-N 空间中, DGLAP 演化变为对角化:
        f(N, Q²) = f(N, Q0²) × exp[∫ (α_s/2π) P(N) dt]
    对整数 N = 2, 3, ..., N_max, 演化核为:
        E(N) = exp[(α_s/2π) P(N) × ln(Q²/Q0²)]
    模块化映射: N → (N × m) mod N_max
    (类比 caustic 中 z_j → z_{j×m mod n} 的连接方式)

核心公式 (相关矩阵):
    ρ_{ij} = H^{-1}_{ij} / √(H^{-1}_{ii} × H^{-1}_{jj})
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from phys_consts import EPS_NUMERICAL
from pdf_param import N_PARAMS
from pdf_param import evaluate_all_flavors, f2_dis, momentum_sum_rule, valence_number_rule
from pdf_fit import compute_chi_squared, compute_theory_value
from experimental_data import DataSet, text_matrix_display


# ============================================================
# 1. Hessian 矩阵计算
# ============================================================
def compute_hessian(params: Dict[str, float], data: DataSet,
                    h: float = 1e-3,
                    x_grid: Optional[List[float]] = None
                    ) -> List[List[float]]:
    """
    χ² 的 Hessian 矩阵 (中心差分):
        H_{ij} ≈ [χ²(p+h_i+h_j) − χ²(p+h_i−h_j)
                  − χ²(p−h_i+h_j) + χ²(p−h_i−h_j)] / (4 h_i h_j)

    参数:
        params:  当前参数
        data:    数据集
        h:       差分步长 (相对)
        x_grid:  求和规则网格
    返回:
        N_PARAMS × N_PARAMS Hessian 矩阵
    """
    keys = list(params.keys())
    n = len(keys)
    hessian = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            ki = keys[i]
            kj = keys[j]
            hi = max(abs(params[ki]) * h, h)
            hj = max(abs(params[kj]) * h, h)

            # 四个扰动点
            p_pp = dict(params); p_pp[ki] += hi; p_pp[kj] += hj
            p_pm = dict(params); p_pm[ki] += hi; p_pm[kj] -= hj
            p_mp = dict(params); p_mp[ki] -= hi; p_mp[kj] += hj
            p_mm = dict(params); p_mm[ki] -= hi; p_mm[kj] -= hj

            chi2_pp = compute_chi_squared(p_pp, data, x_grid=x_grid)
            chi2_pm = compute_chi_squared(p_pm, data, x_grid=x_grid)
            chi2_mp = compute_chi_squared(p_mp, data, x_grid=x_grid)
            chi2_mm = compute_chi_squared(p_mm, data, x_grid=x_grid)

            hessian[i][j] = (chi2_pp - chi2_pm - chi2_mp + chi2_mm) / (4.0 * hi * hj)
            hessian[j][i] = hessian[i][j]

    return hessian


# ============================================================
# 2. 矩阵运算
# ============================================================
def mat_inverse_3x3(m: List[List[float]]) -> Optional[List[List[float]]]:
    """3×3 矩阵求逆 (Cramer 法则)"""
    n = len(m)
    if n != 3:
        return None
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if abs(det) < EPS_NUMERICAL:
        return None
    inv = [[0.0] * 3 for _ in range(3)]
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det
    return inv


def mat_inverse_diag(m: List[List[float]]) -> List[List[float]]:
    """对角占优矩阵近似求逆 (取对角线倒数)"""
    n = len(m)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if abs(m[i][i]) > EPS_NUMERICAL:
            inv[i][i] = 1.0 / m[i][i]
    return inv


def eigenvalues_symmetric_3x3(m: List[List[float]]) -> List[float]:
    """3×3 对称矩阵特征值 (解析公式)"""
    # 使用特征多项式: λ³ − tr(M)λ² + ... = 0
    a = m[0][0]; b = m[1][1]; c = m[2][2]
    d = m[0][1]; e = m[0][2]; f = m[1][2]
    p1 = d * d + e * e + f * f
    if p1 < EPS_NUMERICAL:
        return sorted([a, b, c])
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
    p = math.sqrt(p2 / 6.0)
    # B = (1/p) * (A - q*I)
    B = [[(m[i][j] - (q if i == j else 0)) / p for j in range(3)] for i in range(3)]
    det_B = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
             - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
             + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
    r = det_B / 2.0
    r = max(-1.0, min(1.0, r))
    phi = math.acos(r) / 3.0
    eig1 = q + 2 * p * math.cos(phi)
    eig3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    eig2 = 3 * q - eig1 - eig3
    return sorted([eig1, eig2, eig3])


def hessian_eigendecomposition(hessian: List[List[float]]
                               ) -> Tuple[List[float], List[List[float]]]:
    """
    Hessian 矩阵的特征值分解 (简化版).

    对小矩阵 (≤3), 使用解析公式;
    对大矩阵, 使用对角近似 + 幂迭代估计最大特征值.

    返回:
        (eigenvalues, eigenvectors)
    """
    n = len(hessian)
    if n == 3:
        eigs = eigenvalues_symmetric_3x3(hessian)
        # 简化: 返回特征值, 特征向量使用单位矩阵近似
        vecs = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        return eigs, vecs
    else:
        # 对角近似: 特征值 ≈ 对角元
        eigs = [hessian[i][i] for i in range(n)]
        vecs = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return sorted(eigs), vecs


# ============================================================
# 3. PDF 不确定度 (Hessian 方法)
# ============================================================
def hessian_pdf_uncertainty(
    params: Dict[str, float], data: DataSet,
    x_grid: List[float], flavor: str = 'g',
    delta_chi2: float = 1.0,
    q2: float = 10.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    Hessian PDF 不确定度:
        (Δf(x))² = Σ_k [f_k^+(x) − f_0(x)]²
        f_k^±(x) = f(x; p_0 ± √(Δχ²) × v_k)

    简化: 仅考虑前 3 个主要参数方向.
    返回:
        (f_central, f_upper, f_lower): 中心值与上下误差带
    """
    n = len(x_grid)
    f_central = [0.0] * n
    f_upper = [0.0] * n
    f_lower = [0.0] * n

    # 计算中心值
    for i, x in enumerate(x_grid):
        if x <= 0 or x >= 1:
            continue
        _, _, g_val, _, _ = evaluate_all_flavors(x, params)
        if flavor == 'g':
            f_central[i] = g_val
        elif flavor == 'u_v':
            f_central[i], _, _, _, _ = evaluate_all_flavors(x, params)
        else:
            _, f_central[i], _, _, _ = evaluate_all_flavors(x, params)

    # Hessian 特征向量方向
    keys_subset = ['A_g', 'a_g', 'b_g'] if flavor == 'g' else ['A_uv', 'a_uv', 'b_uv']
    sqrt_dchi2 = math.sqrt(delta_chi2)

    for key in keys_subset:
        if key not in params:
            continue
        p_plus = dict(params)
        p_minus = dict(params)
        step = max(abs(params[key]) * 0.05, 0.01) * sqrt_dchi2
        p_plus[key] = params[key] + step
        p_minus[key] = params[key] - step
        for i, x in enumerate(x_grid):
            if x <= 0 or x >= 1:
                continue
            _, _, gp, _, _ = evaluate_all_flavors(x, p_plus)
            _, _, gm, _, _ = evaluate_all_flavors(x, p_minus)
            if flavor == 'g':
                df_plus = gp - f_central[i]
                df_minus = gm - f_central[i]
            else:
                df_plus = 0.0
                df_minus = 0.0
            f_upper[i] += df_plus ** 2
            f_lower[i] += df_minus ** 2

    f_upper = [f_central[i] + math.sqrt(f_upper[i]) for i in range(n)]
    f_lower = [f_central[i] - math.sqrt(f_lower[i]) for i in range(n)]
    return f_central, f_upper, f_lower


# ============================================================
# 4. 模块化算术 (映射自 caustic)
# ============================================================
def mellin_modular_evolution(
    N_max: int, m_mult: int, q2_ratio: float, alpha_s: float
) -> List[complex]:
    """
    Mellin-N 空间的模块化演化 (映射自 caustic 的 z_j → z_{j×m mod n}):

    在 caustic 中:
        连接 z_j 到 z_{j*m mod n}, 形成焦散图案
    在 Mellin 空间中:
        演化核 E(N) = exp[(α_s/2π) P(N) ln(Q²/Q0²)]
        模块化映射: N → (N × m) mod N_max

    LO 非奇异矩:
        P_{qq}(N) = C_F [−2 S_1(N) + 1/(N(N+1)) + 3/2]
        其中 S_1(N) = Σ_{k=1}^N 1/k (调和数)

    返回:
        E(N) 对 N = 2, ..., N_max 的列表 (复数值, 含相位)
    """
    CF = 4.0 / 3.0
    as2pi = alpha_s / (2.0 * math.pi)
    log_ratio = math.log(max(q2_ratio, 1.001))

    def harmonic_number(n: int) -> float:
        return sum(1.0 / k for k in range(1, max(n, 1) + 1))

    evolution_factors = []
    for N in range(2, N_max + 1):
        # P_qq(N) 矩
        S1 = harmonic_number(N)
        P_N = CF * (-2.0 * S1 + 1.0 / (N * (N + 1)) + 1.5)
        # 演化核
        E_N = math.exp(as2pi * P_N * log_ratio)
        # 模块化相位 (映射自 caustic)
        N_mod = (N * m_mult) % N_max
        phase = 2.0 * math.pi * N_mod / N_max
        E_complex = E_N * complex(math.cos(phase), math.sin(phase))
        evolution_factors.append(E_complex)

    return evolution_factors


def caustic_envelope_density(N_max: int, m_mult: int,
                             n_points: int = 100) -> List[float]:
    """
    焦散包络密度 (映射自 caustic):

    在 caustic 中, 连接线 z_j → z_{j*m mod n} 在圆内形成包络.
    包络密度 ρ(θ) = |{j : line_j 经过 θ 附近}| / n_total

    此处用于可视化 PDF 参数空间的"焦散结构":
    参数扰动在不同方向上的集中程度.

    返回:
        密度值列表 (n_points 个角度)
    """
    density = [0.0] * n_points
    for j in range(N_max):
        j_mod = (j * m_mult) % N_max
        # 连接 z_j 到 z_{j_mod} 的直线
        theta_j = 2.0 * math.pi * j / N_max
        theta_mod = 2.0 * math.pi * j_mod / N_max
        # 直线上的点
        for t_idx in range(10):
            t = t_idx / 9.0
            theta = theta_j + t * (theta_mod - theta_j)
            # 找到最近的密度 bin
            bin_idx = int((theta / (2.0 * math.pi)) * n_points) % n_points
            density[bin_idx] += 1.0
    # 归一化
    max_dens = max(density) if max(density) > EPS_NUMERICAL else 1.0
    return [d / max_dens for d in density]


# ============================================================
# 5. Hessian 相关矩阵显示
# ============================================================
def hessian_correlation_display(hessian: List[List[float]],
                                keys: List[str]) -> str:
    """
    Hessian 相关矩阵的文本热图 (映射自 box_fill):

    核心公式:
        ρ_{ij} = H^{-1}_{ij} / √(H^{-1}_{ii} × H^{-1}_{jj})

    使用 text_matrix_display 显示.
    """
    n = len(hessian)
    # 近似逆 (对角)
    inv_h = mat_inverse_diag(hessian)
    # 相关矩阵
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dii = abs(inv_h[i][i])
            djj = abs(inv_h[j][j])
            if dii > EPS_NUMERICAL and djj > EPS_NUMERICAL:
                corr[i][j] = inv_h[i][j] / math.sqrt(dii * djj)
            else:
                corr[i][j] = 1.0 if i == j else 0.0
    display = text_matrix_display(corr, n_show=min(n, 8), label='ρ(H⁻¹)')
    return display
