"""
sheath_resonance.py
===================
共振模式分析与 Fermat 分解模块。

本模块融合种子项目 420_fermat_factor 的 Fermat 因式分解算法，
用于等离子体鞘层稳定性分析中的共振模式识别。

物理背景：
    鞘层不稳定性通常由模式共振引起。对于频率 ω 和波数 k 的扰动，
    共振条件为：
        ω = m ω_ci + n ω_pi    (离子-电子混合模式)
    其中 m, n 为正整数。

    使用 Fermat 分解 N = a²-b² = (a+b)(a-b) 可以系统地
    找出满足共振条件的 (m,n) 对。

核心算法：
    1. Fermat 因式分解
    2. 共振模式搜索
    3. 色散关系求解
    4. 交叉点检测
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple, List, Optional
import math


def fermat_factor(n: int) -> Tuple[int, int]:
    """
    Fermat 因式分解算法
    （源自种子项目 420_fermat_factor: fermat_factor）

    算法：寻找 a, b 使得 N = a² - b² = (a+b)(a-b)

    方法：
        a = ceil(sqrt(N))
        逐步增大 a，检查 a² - N 是否为完全平方数

    当两个因子接近时效率最高（即 b 较小）。

    参数：
        n: 待分解的正整数

    返回：
        (f1, f2): 因子对，使得 f1 * f2 = n, f1 ≤ f2

    特殊情况：
        - n 为质数时返回 (1, n)
        - n ≤ 0 时引发 ValueError
    """
    if n <= 0:
        raise ValueError(f"Fermat 分解需要正整数，得到 n={n}")
    if n == 1:
        return (1, 1)
    if n % 2 == 0:
        # 偶数：简单分解
        f1 = 2
        while n % f1 == 0:
            f1 *= 2
        f1 //= 2
        return (f1, n // f1)

    a = math.isqrt(n)
    if a * a < n:
        a += 1

    max_iter = max(10000, n)
    for _ in range(max_iter):
        b2 = a * a - n
        if b2 < 0:
            a += 1
            continue
        b = math.isqrt(b2)
        if b * b == b2:
            f1 = a - b
            f2 = a + b
            return (f1, f2)
        a += 1

    # 无法分解 → 质数
    return (1, n)


def fermat_factor_all(n: int) -> List[Tuple[int, int]]:
    """
    找出 N 的所有因子对

    通过遍历可能的因子并使用 Fermat 方法验证

    参数：
        n: 正整数

    返回：
        factors: 所有 (f1, f2) 对，f1 ≤ f2, f1 * f2 = n
    """
    factors = []
    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return factors


def find_resonance_modes(
    omega_ci: float,
    omega_pi: float,
    omega_max: float,
    max_mode: int = 20,
) -> List[Tuple[int, int, float]]:
    """
    寻找离子-电子共振模式

    共振条件：
        ω_{m,n} = m ω_ci + n ω_pi

    对于给定频率范围，找出所有 (m, n) 对

    参数：
        omega_ci: 离子回旋频率
        omega_pi: 离子等离子体频率
        omega_max: 最大搜索频率
        max_mode: 最大模式数

    返回：
        modes: List of (m, n, omega) 模式列表
    """
    modes = []
    for m in range(1, max_mode + 1):
        for n in range(0, max_mode + 1):
            omega = m * omega_ci + n * omega_pi
            if omega <= omega_max:
                modes.append((m, n, omega))

    # 按频率排序
    modes.sort(key=lambda x: x[2])
    return modes[:max_mode]


def find_resonance_via_fermat(
    target_product: int,
    omega_ci: float,
    omega_pi: float,
) -> List[Tuple[int, int, float]]:
    """
    使用 Fermat 分解寻找共振模式

    将目标频率 ω_target 乘以缩放因子得到整数 N，
    然后对 N 进行 Fermat 分解找出可能的 (m, n) 组合。

    参数：
        target_product: 缩放后的目标频率乘积
        omega_ci: 离子回旋频率
        omega_pi: 离子等离子体频率

    返回：
        resonance_pairs: 共振模式对
    """
    factors = fermat_factor_all(target_product)

    resonance_pairs = []
    for f1, f2 in factors:
        # 尝试将因子解释为模式数
        for m in [f1, f2]:
            n_candidate = target_product // m if m > 0 else 0
            if n_candidate >= 0 and m >= 1:
                omega = m * omega_ci + n_candidate * omega_pi
                resonance_pairs.append((m, n_candidate, omega))

    return resonance_pairs


def ion_acoustic_dispersion(
    k: float,
    T_e: float,
    T_i: float,
    m_i: float,
    Z: int = 1,
) -> complex:
    """
    离子声波色散关系

    含阻尼的离子声波色散关系：
        ω² = k² c_s² / (1 + k² λ_De²)

    含 Landau 阻尼修正：
        ω = ω_r + i γ
        γ / ω_r ≈ -√(π/8) * (ω_r / (k v_th_e))³
                  - √(π/8) * (c_s / v_th_i)³ * exp(-c_s² / (2 v_th_i²))

    参数：
        k: 波数
        T_e: 电子温度 [eV]
        T_i: 离子温度 [eV]
        m_i: 离子质量 [kg]
        Z: 电荷数

    返回：
        omega: 复数频率 ω = ω_r + iγ
    """
    from sheath_constants import E_CHARGE, EPSILON_0, M_ELECTRON, PI, K_BOLTZMANN

    T_e_J = T_e * E_CHARGE
    T_i_J = T_i * E_CHARGE

    # 德拜长度
    n_0 = 1e16  # 参考密度
    lambda_De = math.sqrt(EPSILON_0 * T_e_J / (n_0 * E_CHARGE**2))

    # 声速
    c_s = math.sqrt(Z * T_e_J / m_i)

    # 热速度
    v_th_e = math.sqrt(T_e_J / M_ELECTRON)
    v_th_i = math.sqrt(T_i_J / m_i) if T_i > 0 else 1e-6

    # 实部频率
    klambda = k * lambda_De
    omega_r_sq = k**2 * c_s**2 / (1.0 + klambda**2)
    omega_r = math.sqrt(max(omega_r_sq, 0.0))

    # Landau 阻尼率
    gamma_e = -math.sqrt(PI / 8.0) * omega_r * (omega_r / (k * v_th_e))**2 if k * v_th_e > 1e-30 else 0.0
    gamma_i_term = 0.0
    if v_th_i > 1e-30 and T_i > 0:
        arg = -0.5 * (c_s / v_th_i)**2
        if arg > -50:
            gamma_i_term = -math.sqrt(PI / 8.0) * (c_s / v_th_i)**3 * math.exp(arg)

    gamma = gamma_e + gamma_i_term * omega_r

    return complex(omega_r, gamma)


def sheath_dispersion_relation(
    k: float,
    phi_0: float,
    lambda_D: float,
    u_i: float,
    omega_ci: float = 0.0,
) -> complex:
    """
    鞘层修正色散关系

    考虑鞘层电势梯度和磁场效应的修正色散关系：
        D(ω, k) = 1 + χ_e(ω, k) + χ_i(ω, k) = 0

    其中极化率：
        χ_e = 1/(k²λ_De²) * [1 + ζ_e Z(ζ_e)]
        χ_i = 1/(k²λ_De²) * [1 + ζ_i Z(ζ_i)]
        ζ_s = ω / (k v_th_s)

    简化模型（流体极限）：
        ω² = k² c_s² (1 + k²λ_De²)^{-1} + ω_ci² sin²θ_B
             + 3 k² v_th_i²

    参数：
        k: 波数
        phi_0: 鞘层电势幅度
        lambda_D: 德拜长度
        u_i: 离子漂移速度
        omega_ci: 离子回旋频率

    返回：
        omega: 复数频率
    """
    c_s = 1.0  # 归一化声速
    v_th_i = 0.1  # 归一化离子热速度

    klambda = k * lambda_D

    # 流体色散
    omega_sq = (
        k**2 * c_s**2 / (1.0 + klambda**2)
        + omega_ci**2 * 0.1  # 磁场修正 (sin²θ 平均)
        + 3.0 * k**2 * v_th_i**2  # 有限温度修正
    )

    # 鞘层电势修正
    phi_correction = 0.1 * abs(phi_0) * k / (1.0 + k**2)

    omega_r = math.sqrt(max(omega_sq, 0.0)) + phi_correction

    # 简单阻尼模型
    gamma = -0.01 * omega_r * klambda**2 / (1.0 + klambda**2)

    return complex(omega_r, gamma)


def find_mode_crossings(
    k_range: np.ndarray,
    T_e: float,
    T_i: float,
    m_i: float,
    n_modes: int = 5,
) -> List[Tuple[float, float]]:
    """
    寻找色散曲线的交叉点

    交叉点对应共振不稳定性

    参数：
        k_range: 波数范围
        T_e, T_i, m_i: 等离子体参数
        n_modes: 考虑的模数

    返回：
        crossings: List of (k_cross, omega_cross)
    """
    # 计算各模式的色散曲线
    curves = []
    for mode_idx in range(n_modes):
        omega_curve = []
        for k in k_range:
            k_scaled = k * (1.0 + 0.5 * mode_idx)
            omega = ion_acoustic_dispersion(k_scaled, T_e, T_i, m_i)
            omega_curve.append(omega.real)
        curves.append(np.array(omega_curve))

    # 寻找交叉点
    crossings = []
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            diff = curves[i] - curves[j]
            for idx in range(len(diff) - 1):
                if diff[idx] * diff[idx + 1] < 0:
                    # 线性插值找交叉点
                    k_cross = k_range[idx] - diff[idx] * (
                        k_range[idx + 1] - k_range[idx]
                    ) / (diff[idx + 1] - diff[idx])
                    omega_cross = 0.5 * (curves[i][idx] + curves[i][idx + 1])
                    crossings.append((k_cross, omega_cross))

    return crossings
