"""
phys_consts.py — 计算高能物理：PDF 全局拟合中的基本物理常数与数值格式
=============================================================
本项目核心任务: 在 x ∈ [x_min, 1] 与 Q² ∈ [Q0², Qmax²] 的演化区域内,
以高阶有限差分 (四阶/六阶) 求解 LO DGLAP 演化方程, 并完成 PDF 的全局
chi² 拟合与误差传播分析。

本模块封装:
  (1) 标准模型与 QCD 基本常数;
  (2) 圈数-依赖的强耦合常数 α_s(Q²);
  (3) 高阶有限差分模板 (FD4/FD6 中心差分, 四阶二阶导);
  (4) 四阶 Runge–Kutta 与 IMEX 半步系数;
  (5) 数值鲁棒性相关的 epsilon 与正则化参数。

核心公式 (LO β 函数):
    β_0 = (33 − 2 n_f) / 3
    α_s(Q²) = 4π / [β_0 ln(Q² / Λ²)]

核心公式 (四阶中心差分一阶导):
    f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h) + O(h⁴)

核心公式 (四阶中心差分二阶导):
    f''(x) ≈ [−f(x+2h) + 16f(x+h) − 30f(x) + 16f(x−h) − f(x−2h)] / (12h²) + O(h⁴)
"""
from __future__ import annotations
import math
from typing import List, Tuple

# ============================================================
# 1. 标准模型与 QCD 基本常数
# ============================================================
# 强耦合常数在 Z 玻色子质量处的实验值 (PDG 2022)
ALPHA_S_MZ: float = 0.1180
M_Z: float = 91.1876            # Z 玻色子质量 (GeV)
M_W: float = 80.379             # W 玻色子质量 (GeV)
M_H: float = 125.10             # Higgs 玻色子质量 (GeV)
M_TOP: float = 172.76           # 顶夸克质量 (GeV)
M_BOTTOM: float = 4.18          # 底夸克质量 (GeV)
M_CHARM: float = 1.27           # 粲夸克质量 (GeV)
# 电弱混合角 (sin^2 theta_W, MS-bar 方案)
SIN2THETA_W: float = 0.23122
# 电磁耦合常数 (在 Thomson 极限)
ALPHA_EM: float = 1.0 / 137.035999084
# 费米子质量阈值 (GeV)
MASS_THRESHOLD: List[float] = [0.0, M_CHARM, M_BOTTOM, M_TOP]
# 活跃味数 (按能量区间)
N_F_LIGHT: int = 3              # u, d, s (在 Q < m_c 时)
N_F_DEFAULT: int = 5            # 在 Q > m_b 时

# ============================================================
# 2. QCD 圈系数与 Λ_QCD
# ============================================================
def beta_coefficient(nf: int, order: int = 0) -> float:
    """
    QCD β 函数系数:
        β_0 = (33 − 2 n_f) / 12π        (对应 α_s / (4π) 归一)
        β_1 = (153 − 19 n_f) / (48π²)

    此处返回未归一化的形式以简化后续计算:
        b_0 = (33 − 2 n_f) / 3
        b_1 = (102 − 38 n_f / 3)
    """
    if order == 0:
        return (33.0 - 2.0 * nf) / 3.0
    elif order == 1:
        return 102.0 - 38.0 * nf / 3.0
    else:
        raise ValueError(f"beta_coefficient: order={order} 不支持, 仅支持 0 或 1")


def compute_lambda_qcd(nf: int = 5, alpha_s_ref: float = ALPHA_S_MZ,
                       mu_ref: float = M_Z) -> float:
    """
    由 α_s(μ_ref) 反推 Λ_QCD (LO):
        Λ_QCD = μ_ref × exp[−2π / (β_0 α_s(μ_ref))]
             = μ_ref × exp[−6π / ((33 − 2n_f) α_s(μ_ref))]

    此关系源自 LO 跑动耦合的精确反解:
        α_s(Q²) = 4π / [β_0 ln(Q² / Λ²)]
    """
    b0 = beta_coefficient(nf, order=0)
    exponent = -2.0 * math.pi / (b0 * alpha_s_ref)
    return mu_ref * math.exp(exponent)


# 默认 Λ_QCD (n_f=5, LO)
LAMBDA_QCD_LO: float = compute_lambda_qcd(N_F_DEFAULT, ALPHA_S_MZ, M_Z)


def alpha_s_lo(q2: float, nf: int = N_F_DEFAULT,
               lambda_qcd: float = LAMBDA_QCD_LO) -> float:
    """
    LO 跑动耦合:
        α_s(Q²) = 4π / [β_0 × ln(Q² / Λ²)]

    当 Q² < Λ² 时 (Landau pole 区域), 返回截断值以避免发散。
    鲁棒性处理: Q² 下限为 Λ² × (1 + ε)
    """
    if q2 <= 0.0:
        return 12.0  # 极端红外截断
    lam2 = lambda_qcd * lambda_qcd
    q2_safe = max(q2, lam2 * 1.05)
    log_val = math.log(q2_safe / lam2)
    if log_val < 0.1:
        log_val = 0.1
    b0 = beta_coefficient(nf, order=0)
    return 4.0 * math.pi / (b0 * log_val)


def alpha_s_nlo(q2: float, nf: int = N_F_DEFAULT,
                lambda_qcd: float = LAMBDA_QCD_LO) -> float:
    """
    NLO 跑动耦合 (迭代解):
        α_s(Q²) = (4π / β_0) × (1/L) × [1 − (β_1/β_0²) × ln(L)/L]
    其中 L = ln(Q²/Λ²), β_0 = (33−2n_f)/3, β_1 = 102 − 38n_f/3

    此处使用一次迭代近似, 以 LO 为初值。
    """
    if q2 <= 0.0:
        return 12.0
    lam2 = lambda_qcd * lambda_qcd
    q2_safe = max(q2, lam2 * 1.05)
    L = math.log(q2_safe / lam2)
    if L < 0.1:
        L = 0.1
    b0 = beta_coefficient(nf, order=0)
    b1 = beta_coefficient(nf, order=1)
    a_lo = 4.0 * math.pi / (b0 * L)
    correction = 1.0 - (b1 / (b0 * b0)) * math.log(L) / L
    return a_lo * correction


# ============================================================
# 3. 高阶有限差分模板
# ============================================================
# 四阶中心差分一阶导 (5 点模板):
#   f'(x) ≈ Σ_k c_k f(x + k h) / h
#   k = [-2, -1, 0, 1, 2]
#   c = [1/12, -8/12, 0, 8/12, -1/12]
FD4_FIRST_DERIV_STENCIL: Tuple[float, ...] = (
    1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0
)
FD4_FIRST_DERIV_OFFSETS: Tuple[int, ...] = (-2, -1, 0, 1, 2)

# 六阶中心差分一阶导 (7 点模板):
#   c = [-1/60, 9/60, -45/60, 0, 45/60, -9/60, 1/60]
FD6_FIRST_DERIV_STENCIL: Tuple[float, ...] = (
    -1.0 / 60.0, 9.0 / 60.0, -45.0 / 60.0, 0.0,
    45.0 / 60.0, -9.0 / 60.0, 1.0 / 60.0
)
FD6_FIRST_DERIV_OFFSETS: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)

# 四阶中心差分二阶导 (5 点模板):
#   c = [-1/12, 16/12, -30/12, 16/12, -1/12]
FD4_SECOND_DERIV_STENCIL: Tuple[float, ...] = (
    -1.0 / 12.0, 16.0 / 12.0, -30.0 / 12.0, 16.0 / 12.0, -1.0 / 12.0
)
FD4_SECOND_DERIV_OFFSETS: Tuple[int, ...] = (-2, -1, 0, 1, 2)


def fd_first_derivative(fvals: List[float], idx: int, h: float,
                        order: int = 4) -> float:
    """
    计算数组 fvals 在索引 idx 处的有限差分一阶导数。

    参数:
        fvals: 函数值数组 (等距采样)
        idx:   求导点索引
        h:     网格步长
        order: 4 (四阶) 或 6 (六阶)

    边界处理:
        - 靠近边界时, 自动降级为低阶单侧差分, 保持 O(h²) 精度
        - 若 idx 在边界 2 格内, 使用前向/后向差分模板

    返回:
        df/dx 在 idx 处的近似值
    """
    n = len(fvals)
    if n < 3:
        return 0.0
    if order == 6 and 3 <= idx < n - 3:
        stencil = FD6_FIRST_DERIV_STENCIL
        offsets = FD6_FIRST_DERIV_OFFSETS
    elif 2 <= idx < n - 2:
        stencil = FD4_FIRST_DERIV_STENCIL
        offsets = FD4_FIRST_DERIV_OFFSETS
    else:
        # 边界降级: 二阶单侧差分
        # f'(x_0) ≈ (−3f_0 + 4f_1 − f_2) / (2h)
        if idx == 0:
            return (-3.0 * fvals[0] + 4.0 * fvals[1] - fvals[2]) / (2.0 * h)
        elif idx == n - 1:
            return (3.0 * fvals[-1] - 4.0 * fvals[-2] + fvals[-3]) / (2.0 * h)
        elif idx == 1:
            return (-fvals[0] + fvals[2]) / (2.0 * h)
        else:  # idx == n-2
            return (-fvals[-3] + fvals[-1]) / (2.0 * h)

    result = 0.0
    for c, off in zip(stencil, offsets):
        result += c * fvals[idx + off]
    return result / h


def fd_second_derivative(fvals: List[float], idx: int, h: float) -> float:
    """
    四阶中心差分二阶导:
        f''(x) ≈ [−f(x+2h) + 16f(x+h) − 30f(x) + 16f(x−h) − f(x−2h)] / (12h²)

    边界降级: 标准三点二阶导 f''(x) ≈ (f(x+h) − 2f(x) + f(x−h)) / h²
    """
    n = len(fvals)
    if n < 3:
        return 0.0
    if 2 <= idx < n - 2:
        stencil = FD4_SECOND_DERIV_STENCIL
        offsets = FD4_SECOND_DERIV_OFFSETS
        result = 0.0
        for c, off in zip(stencil, offsets):
            result += c * fvals[idx + off]
        return result / (h * h)
    else:
        # 三点二阶导 (边界降级)
        if idx == 0:
            return (fvals[0] - 2.0 * fvals[1] + fvals[2]) / (h * h)
        elif idx == n - 1:
            return (fvals[-3] - 2.0 * fvals[-2] + fvals[-1]) / (h * h)
        else:
            ip1 = min(idx + 1, n - 1)
            im1 = max(idx - 1, 0)
            return (fvals[im1] - 2.0 * fvals[idx] + fvals[ip1]) / (h * h)


# ============================================================
# 4. 数值鲁棒性参数
# ============================================================
EPS_NUMERICAL: float = 1.0e-14       # 通用机器精度阈值
PDF_POSITIVITY_FLOOR: float = 1.0e-10  # PDF 最小值截断 (防止负值)
ALPHA_S_MAX: float = 12.0            # α_s 最大截断 (红外区域)
ALPHA_S_MIN: float = 0.05            # α_s 最小截断 (紫外区域)
CHI2_CONVERGENCE_TOL: float = 1.0e-6  # χ² 收敛判据
STEP_SIZE_DEFAULT: float = 0.01      # 默认演化步长 dt = ln(Q²/Q0²)
X_MIN_DEFAULT: float = 1.0e-3        # 默认 x 最小值
X_MAX_DEFAULT: float = 1.0 - 1.0e-8  # 默认 x 最大值 (避免 x=1 奇点)
Q0_SQ_DEFAULT: float = 2.0           # 初始演化尺度 Q0² (GeV²)
QMAX_SQ_DEFAULT: float = 1.0e4       # 最大演化尺度 Qmax² (GeV²)
NX_DEFAULT: int = 24                 # 默认 x 网格点数
NQ_DEFAULT: int = 12                 # 默认 Q² 网格点数


def clamp_alpha_s(alpha: float) -> float:
    """将 α_s 限制在物理合理范围内, 避免 Landau pole 导致发散"""
    return max(ALPHA_S_MIN, min(alpha, ALPHA_S_MAX))
