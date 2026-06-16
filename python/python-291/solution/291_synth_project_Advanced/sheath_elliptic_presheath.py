"""
sheath_elliptic_presheath.py
============================
椭圆积分与磁前鞘模块。

本模块融合种子项目 332_ellipsoid 的椭圆积分算法，
用于计算磁化等离子体前鞘（magnetic presheath）的几何因子。

物理背景：
    在磁化等离子体中，磁场以角度 θ_B 入射到壁面。
    前鞘区的离子运动受到洛伦兹力影响，其轨迹由
    椭圆积分描述。

    磁通量计算涉及椭球面积公式：
        A = 2π c² + 2π ab / sin(φ) * F(φ, k)
    其中 F 为第一类不完全椭圆积分

    Chodura 鞘层条件：
        在磁前鞘中，离子必须满足 Chodura 条件：
        M_⊥ ≥ 1 (垂直 Mach 数)
    其中 M_⊥ = u_⊥ / c_s

核心公式：
    第一类不完全椭圆积分：
        F(φ, k) = ∫₀^φ dθ / √(1 - k² sin²θ)

    第二类不完全椭圆积分：
        E(φ, k) = ∫₀^φ √(1 - k² sin²θ) dθ

    第一类完全椭圆积分 K(k) = F(π/2, k)
    第二类完全椭圆积分 E(k) = E(π/2, k)
"""

import numpy as np
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from scipy import integrate
from typing import Tuple, Optional
import math


def elliptic_F(phi: float, k: float) -> float:
    """
    第一类不完全椭圆积分
    （源自种子项目 332_ellipsoid: rf, rd 等辅助函数）

        F(φ, k) = ∫₀^φ dθ / √(1 - k² sin²θ)

    参数：
        phi: 振幅角 (0 ≤ φ ≤ π/2)
        k: 模数 (0 ≤ k < 1)

    返回：
        F: 积分值
    """
    k = min(abs(k), 0.9999999)
    phi = max(0.0, min(phi, math.pi / 2))
    return float(ellipkinc(phi, k**2))


def elliptic_E(phi: float, k: float) -> float:
    """
    第二类不完全椭圆积分
    （源自种子项目 332_ellipsoid: ellipsoid_area）

        E(φ, k) = ∫₀^φ √(1 - k² sin²θ) dθ

    参数：
        phi: 振幅角
        k: 模数

    返回：
        E: 积分值
    """
    k = min(abs(k), 0.9999999)
    phi = max(0.0, min(phi, math.pi / 2))
    return float(ellipeinc(phi, k**2))


def elliptic_K(k: float) -> float:
    """
    第一类完全椭圆积分
        K(k) = F(π/2, k) = ∫₀^{π/2} dθ / √(1 - k² sin²θ)

    参数：
        k: 模数 (0 ≤ k < 1)

    返回：
        K: 积分值
    """
    k = min(abs(k), 0.9999999)
    return float(ellipk(k**2))


def elliptic_E_complete(k: float) -> float:
    """
    第二类完全椭圆积分
        E(k) = E(π/2, k) = ∫₀^{π/2} √(1 - k² sin²θ) dθ
    """
    k = min(abs(k), 0.9999999)
    return float(ellipe(k**2))


def ellipsoid_surface_area(a: float, b: float, c: float) -> float:
    """
    椭球表面积
    （源自种子项目 332_ellipsoid: ellipsoid_area）

    对于椭球 (x/a)² + (y/b)² + (z/c)² = 1 (a ≥ b ≥ c)：
        S = 2πc² + 2πab sin(φ) F(φ,k)
            + 2π(a²-c²)/(a² sin(φ)) * [E(φ,k) - (1-k²)F(φ,k)]/3
    简化形式 (Knud Thomsen 近似)：
        S ≈ 4π [(a^p b^p + a^p c^p + b^p c^p)/3]^{1/p}
    其中 p = 1.6075

    精确形式使用椭圆积分

    参数：
        a, b, c: 半轴长度

    返回：
        S: 表面积
    """
    if a < b:
        a, b = b, a
    if b < c:
        b, c = c, b
    if a < b:
        a, b = b, a

    # 退化情况
    if c < 1e-15:
        return math.pi * a * b  # 扁平椭圆

    # 使用 Knud Thomsen 近似 (高精度)
    p = 1.6075
    ap = a**p
    bp = b**p
    cp = c**p
    S_approx = 4.0 * math.pi * ((ap * bp + ap * cp + bp * cp) / 3.0)**(1.0 / p)

    return S_approx


def ellipsoid_volume(a: float, b: float, c: float) -> float:
    """
    椭球体积
    （源自种子项目 332_ellipsoid: ellipsoid_volume）

        V = (4/3) π a b c
    """
    return (4.0 / 3.0) * math.pi * a * b * c


def magnetic_flux_factor(
    theta_B: float,
    rho_s: float,
    lambda_D: float,
) -> float:
    """
    磁通量几何因子

    计算磁化前鞘中的有效磁通量：
        Γ_B = ∫₀^{π/2} √(1 - sin²θ_B sin²φ) dφ
    这是第二类不完全椭圆积分 E(π/2, sin θ_B)

    参数：
        theta_B: 磁场入射角
        rho_s: 声速拉莫尔半径
        lambda_D: 德拜长度

    返回：
        flux_factor: 磁通量几何因子
    """
    k = math.sin(abs(theta_B))
    return elliptic_E(math.pi / 2, k)


def chodura_condition(
    theta_B: float,
    M_parallel: float,
    M_perp: float = 0.0,
) -> Tuple[bool, float, float]:
    """
    Chodura 鞘层条件检验

    Chodura (1982) 证明在磁化鞘层中，
    离子在进入磁前鞘时必须满足：
        M_∥² + M_⊥² ≥ 1 + sin²θ_B / (k² λ_De²)

    简化条件（流体极限）：
        u_∥ ≥ c_s cos θ_B

    参数：
        theta_B: 磁场入射角
        M_parallel: 平行 Mach 数 u_∥/c_s
        M_perp: 垂直 Mach 数 u_⊥/c_s

    返回：
        satisfied: 是否满足 Chodura 条件
        M_total: 总 Mach 数
        M_critical: 临界 Mach 数
    """
    cos_theta = math.cos(theta_B)
    sin_theta = math.sin(theta_B)

    M_total = math.sqrt(M_parallel**2 + M_perp**2)
    M_critical = cos_theta  # 简化条件

    satisfied = M_parallel >= M_critical - 1e-10

    return satisfied, M_total, M_critical


def presheath_transit_time(
    theta_B: float,
    L_presheath: float,
    c_s: float,
    omega_ci: float,
) -> float:
    """
    磁前鞘渡越时间

    离子穿越磁前鞘的时间：
        τ = ∫ dx / u_∥(x)

    在磁前鞘中，离子加速受电场和磁场共同影响：
        u_∥(x) = c_s √(cos²θ_B + 2x/L_p sin²θ_B)

    积分给出椭圆积分形式

    参数：
        theta_B: 磁场入射角
        L_presheath: 前鞘长度
        c_s: 声速
        omega_ci: 离子回旋频率

    返回：
        tau: 渡越时间
    """
    cos_t = math.cos(theta_B)
    sin_t = math.sin(theta_B)

    if abs(sin_t) < 1e-10:
        return L_presheath / c_s

    # 积分：τ = ∫₀^L dx / (c_s √(cos²θ + 2x sin²θ/L))
    # 令 u = cos²θ + 2x sin²θ/L
    # du = 2 sin²θ/L dx
    # τ = L/(2 c_s sin²θ) ∫ u^{-1/2} du
    # = L/(c_s sin²θ) [√u]_0^1 (近似)

    u_start = cos_t**2
    u_end = cos_t**2 + 2.0 * sin_t**2

    tau = L_presheath / (c_s * sin_t**2) * (math.sqrt(u_end) - math.sqrt(max(u_start, 1e-15)))

    return tau


def magnetic_presheath_profile(
    x: np.ndarray,
    theta_B: float,
    n_0: float = 1.0,
    c_s: float = 1.0,
    L_presheath: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    磁前鞘离子密度和速度分布

    基于 Chodura 模型：
        n_i(x) = n_0 / √(cos²θ_B + 2(x/L_p) sin²θ_B)
        u_∥(x) = c_s √(cos²θ_B + 2(x/L_p) sin²θ_B)

    参数：
        x: 位置数组 (归一化)
        theta_B: 磁场入射角
        n_0: 参考密度
        c_s: 声速
        L_presheath: 前鞘长度

    返回：
        n_i: 离子密度
        u_parallel: 平行速度
        u_perp: 垂直速度
    """
    cos_t = math.cos(theta_B)
    sin_t = math.sin(theta_B)

    x_norm = np.clip(x / L_presheath, 0.0, 1.0)

    # Bohm-Chodura 加速
    u_sq = cos_t**2 + 2.0 * x_norm * sin_t**2
    u_sq = np.maximum(u_sq, 1e-10)

    u_parallel = c_s * np.sqrt(u_sq)

    # 连续性方程：n u = const
    n_i = n_0 * cos_t / np.sqrt(u_sq)

    # E×B 漂移引起的垂直速度
    # u_perp ~ ω_ci ρ_s sin θ_B
    u_perp = np.zeros_like(x)
    if abs(sin_t) > 1e-10:
        u_perp = 0.1 * c_s * sin_t * np.exp(-x_norm)

    return n_i, u_parallel, u_perp
