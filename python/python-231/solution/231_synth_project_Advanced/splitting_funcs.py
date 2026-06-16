"""
splitting_funcs.py — LO DGLAP 分裂函数与光滑部分提取
=============================================================
本模块实现 LO (Leading Order) DGLAP 分裂函数 P_{ij}(z),
以及它们的"光滑部分"分解 (用于 IMEX 时间推进):

    P_{ij}(z) = P_smooth_{ij}(z) + δ_{ij} × A_i × δ(1−z)

其中 A_i 是对角系数 (奇异性强度), δ(1−z) 项通过 + 规定 (plus prescription)
处理.

核心公式 (LO 非奇异分裂函数):
    P_{qq}(z) = C_F × [ (1+z²)/(1−z) ]_+
    P_{qg}(z) = T_R × [ z² + (1−z)² ]
    P_{gq}(z) = C_F × [ (1+(1−z)²) / z ]
    P_{gg}(z) = 2C_A × [ z/(1−z) + (1−z)/z + z(1−z) ]_+

核心公式 (+ 规定分解):
    ∫_0^1 dz [f(z)]_+ g(z) = ∫_0^1 dz f(z) [g(z) − g(1)]

    → P_smooth_{qq}(z) = C_F × (1+z²)/(1−z)  (z ≠ 1)
       A_qq = C_F × ∫_0^1 dz (1+z²)/(1−z) 的正则化值
            = C_F × (3/2)

核心公式 (颜色因子):
    C_F = (N_c² − 1) / (2N_c) = 4/3    (SU(3) 基础表示)
    C_A = N_c = 3                        (SU(3) 伴随表示)
    T_R = 1/2                            (基础表示的 Dynkin 指标)
"""
from __future__ import annotations
import math
from typing import Tuple

# ============================================================
# 1. QCD 颜色因子 (SU(3))
# ============================================================
N_C: int = 3           # 色数
C_F: float = 4.0 / 3.0  # (N_c² − 1)/(2N_c)
C_A: float = 3.0        # N_c
T_R: float = 0.5        # 基础表示 Dynkin 指标


# ============================================================
# 2. LO 分裂函数
# ============================================================
def p_qq_raw(z: float) -> float:
    """
    P_{qq} 的原始 (非正则化) 形式:
        P_{qq}(z) = C_F × (1 + z²) / (1 − z)

    注意: z → 1 时发散, 需要 + 规定或光滑部分提取
    鲁棒性: 当 1−z < ε 时返回截断值
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    omz = 1.0 - z
    if omz < 1e-12:
        return C_F * 2.0 / 1e-12  # 截断
    return C_F * (1.0 + z * z) / omz


def p_qg(z: float) -> float:
    """
    P_{qg}(z): 胶子 → 夸克-反夸克
        P_{qg}(z) = T_R × [ z² + (1−z)² ]

    此函数在 z ∈ (0,1) 上处处光滑, 无奇异性.
    物理解释: 胶子分裂为 q q̄ 对时, 夸克的动量分数分布
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    return T_R * (z * z + (1.0 - z) * (1.0 - z))


def p_gq(z: float) -> float:
    """
    P_{gq}(z): 夸克 → 夸克 + 胶子
        P_{gq}(z) = C_F × [ 1 + (1−z)² ] / z

    z → 0 时 ~ 1/z 发散 (软胶子辐射)
    鲁棒性: z < ε 时截断
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    if z < 1e-12:
        return C_F * 2.0 / 1e-12
    return C_F * (1.0 + (1.0 - z) * (1.0 - z)) / z


def p_gg_raw(z: float) -> float:
    """
    P_{gg} 的原始形式:
        P_{gg}(z) = 2 C_A × [ z/(1−z) + (1−z)/z + z(1−z) ]

    z → 1 和 z → 0 时均发散 (软胶子 + 共线辐射)
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    z_safe = max(z, 1e-12)
    omz = max(1.0 - z, 1e-12)
    return 2.0 * C_A * (z_safe / omz + omz / z_safe + z_safe * omz)


# ============================================================
# 3. 光滑部分分解 (IMEX 方案所需)
# ============================================================
# 对角系数 A_ii (来自 + 规定的 delta(1−z) 项):
#   A_qq = C_F × 3/2  (来自 ∫_0^1 dz (1+z²)/(1−z) 的正则化)
#   A_gg = C_A × 11/6 + (−2 n_f T_R) × 2/3
#        = C_A × 11/6 − n_f × 2/3
#   注意: A_gg 的完整表达式包含 n_f 依赖
A_QQ_DIAGONAL: float = C_F * 1.5    # 3/2 × C_F = 2
A_GG_COEFF_CA: float = 11.0 / 6.0   # 11/6 × C_A
A_GG_COEFF_NF: float = 2.0 / 3.0    # 2/3 × T_R × 2 = 2/3 (每个活跃味)


def a_gg_diagonal(nf: int = 3) -> float:
    """
    P_{gg} 的对角系数:
        A_gg = C_A × 11/6 − n_f × 2/3

    此系数来自 + 规定:
        ∫_0^1 dz [z/(1−z)]_+ = −1 (正则化后)
    与 β_0 的关系: A_gg = β_0/2 = (33−2n_f)/12
    """
    return C_A * A_GG_COEFF_CA - nf * A_GG_COEFF_NF


def p_qq_smooth(z: float) -> float:
    """
    P_{qq} 的光滑部分 (减去了 z=1 处的奇异性):
        P_smooth_{qq}(z) = C_F × (1+z²)/(1−z)   对 z ≠ 1
        与 P_raw 相同, 但对 z → 1 时施加截断.

    在 IMEX 方案中, 对角项 (δ(1−z)) 被隐式处理,
    光滑部分用于显式卷积.
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    omz = 1.0 - z
    # 截断处理: 当 omz 极小时, 返回 C_F × 2 / omz 的截断
    if omz < 1e-10:
        return C_F * 2.0 / 1e-10
    return C_F * (1.0 + z * z) / omz


def p_gg_smooth(z: float) -> float:
    """
    P_{gg} 的光滑部分:
        P_smooth_{gg}(z) = 2 C_A × [ z/(1−z) + (1−z)/z + z(1−z) ]

    z → 1 处截断, z → 0 处由 (1−z)/z ~ 1/z 主导.
    在 IMEX 中, 1/z 奇异性在 v = ln(1/x) 空间被指数 Jacobian 软化.
    """
    if z <= 0.0 or z >= 1.0:
        return 0.0
    z_safe = max(z, 1e-10)
    omz = max(1.0 - z, 1e-10)
    return 2.0 * C_A * (z_safe / omz + omz / z_safe + z_safe * omz)


# ============================================================
# 4. 核函数卷积 (用于 DGLAP 演化)
# ============================================================
def convolve_smooth_qq(f_vals: list, v_grid: list, idx: int) -> float:
    """
    光滑 qq 核的离散卷积:
        [P_smooth_{qq} ⊗ f](v_k) = Σ_{j>k} Δv × e^{−(v_j−v_k)}
                                    × P_smooth_{qq}(1 − e^{−(v_j−v_k)})
                                    × f(v_j)

    其中 v = ln(1/x), 所以 z = x/y = e^{−(v_y − v_x)}.
    Jacobian: dz = e^{−(v_y−v_x)} dv_y

    参数:
        f_vals: PDF 在 v_grid 上的值
        v_grid: v = ln(1/x) 的网格
        idx:    当前计算点的索引
    返回:
        卷积值 (不含对角项)
    """
    n = len(v_grid)
    result = 0.0
    vk = v_grid[idx]
    for j in range(idx + 1, n):
        dv = v_grid[j] - vk
        if dv < 1e-15:
            continue
        z = math.exp(-dv)
        if z <= 0.0 or z >= 1.0:
            continue
        pz = p_qq_smooth(z)
        jacobian = math.exp(-dv)
        result += dv * jacobian * pz * f_vals[j]
    return result


def convolve_smooth_qg(f_g: list, v_grid: list, idx: int) -> float:
    """
    qg 核的离散卷积 (胶子 → 夸克):
        [P_{qg} ⊗ g](v_k) = Σ_j Δv × e^{−(v_j−v_k)}
                             × P_{qg}(1 − e^{−(v_j−v_k)})
                             × g(v_j)
    """
    n = len(v_grid)
    result = 0.0
    vk = v_grid[idx]
    for j in range(idx + 1, n):
        dv = v_grid[j] - vk
        if dv < 1e-15:
            continue
        z = math.exp(-dv)
        if z <= 0.0 or z >= 1.0:
            continue
        pz = p_qg(z)
        jacobian = math.exp(-dv)
        result += dv * jacobian * pz * f_g[j]
    return result


def convolve_smooth_gq(f_q: list, v_grid: list, idx: int) -> float:
    """
    gq 核的离散卷积 (夸克 → 胶子):
        [P_{gq} ⊗ q](v_k) = Σ_j Δv × e^{−(v_j−v_k)}
                             × P_{gq}(1 − e^{−(v_j−v_k)})
                             × q(v_j)
    """
    n = len(v_grid)
    result = 0.0
    vk = v_grid[idx]
    for j in range(idx + 1, n):
        dv = v_grid[j] - vk
        if dv < 1e-15:
            continue
        z = math.exp(-dv)
        if z <= 0.0 or z >= 1.0:
            continue
        pz = p_gq(z)
        jacobian = math.exp(-dv)
        result += dv * jacobian * pz * f_q[j]
    return result


def convolve_smooth_gg(f_g: list, v_grid: list, idx: int) -> float:
    """
    gg 核的离散卷积 (胶子 → 胶子):
        [P_smooth_{gg} ⊗ g](v_k) = Σ_{j>k} Δv × e^{−(v_j−v_k)}
                                    × P_smooth_{gg}(1 − e^{−(v_j−v_k)})
                                    × g(v_j)
    """
    n = len(v_grid)
    result = 0.0
    vk = v_grid[idx]
    for j in range(idx + 1, n):
        dv = v_grid[j] - vk
        if dv < 1e-15:
            continue
        z = math.exp(-dv)
        if z <= 0.0 or z >= 1.0:
            continue
        pz = p_gg_smooth(z)
        jacobian = math.exp(-dv)
        result += dv * jacobian * pz * f_g[j]
    return result


def diagonal_coefficients(nf: int = 3) -> Tuple[float, float]:
    """
    返回对角系数 (A_qq, A_gg):
        A_qq = C_F × 3/2 = 2
        A_gg = C_A × 11/6 − n_f × 2/3

    这些系数决定 IMEX 隐式步的衰减率.
    """
    return A_QQ_DIAGONAL, a_gg_diagonal(nf)
