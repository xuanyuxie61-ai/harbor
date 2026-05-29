"""
level_density.py
================
核能级密度统计模块

本模块基于 log_normal 分布理论与壳模型能谱统计，
实现核能级密度（Nuclear Level Density, NLD）的计算与统计推断。

物理背景：
- 低激发能区：离散能级，壳效应显著
- 中高能区：能级密度呈指数增长，可用 Fermi 气体模型描述
- 能级间距分布：Wigner-Dyson 分布（量子混沌）vs Poisson 分布（可积系统）

数学模型：
1. Bethe 公式（Fermi 气体）：
   ρ(E) = (√π / 12a^{1/4}) · exp(2√(aE)) / E^{5/4}
   其中 a 为能级密度参数，a ≈ A/8 MeV⁻¹

2. 对关联修正（BCS）：
   ρ_{BCS}(E) = ρ_{FG}(E) · tanh(√(aE) · Δ/E)

3. 对数正态分布（用于能级间距统计）：
   P(s) = 1/(s σ√(2π)) · exp[-(ln s - μ)²/(2σ²)]

4. Wigner  surmise（GOE 统计）：
   P_W(s) = (πs/2) exp(-πs²/4)
"""

import numpy as np
from math import sqrt, pi, exp, log, erf


def bethe_formula(E, a_parameter, spin=None, parity=None):
    """
    Bethe 能级密度公式（Fermi 气体近似）。

    ρ(E) = (1 / (12 √a)) · (exp(2√(aE)) / E^{5/4})

    对于特定自旋 I：
    ρ(E, I) = ρ(E) · (2I + 1) / (2√(2σ²)) · exp[-(I + 1/2)²/(2σ²)]
    其中 σ² = 6√(aE) / π² 为自旋切断参数。

    参数
    ----
    E : float 或 ndarray
        激发能 (MeV)
    a_parameter : float
        能级密度参数 (MeV⁻¹)
    spin : float, optional
        自旋量子数 I
    parity : int, optional
        宇称（±1），暂不影响计算

    返回
    ----
    rho : float 或 ndarray
        能级密度 (MeV⁻¹)
    """
    E = np.asarray(E, dtype=float)
    rho = np.zeros_like(E)
    mask = E > 1e-6

    if np.any(mask):
        Em = E[mask]
        sqrt_aE = np.sqrt(a_parameter * Em)
        rho_base = (1.0 / (12.0 * np.sqrt(a_parameter))) * np.exp(2.0 * sqrt_aE) / (Em ** (5.0 / 4.0))

        if spin is not None:
            sigma2 = 6.0 * sqrt_aE / (pi ** 2)
            spin_factor = (2.0 * spin + 1.0) / (2.0 * np.sqrt(2.0 * sigma2))
            spin_factor *= np.exp(-(spin + 0.5) ** 2 / (2.0 * sigma2))
            rho[mask] = rho_base * spin_factor
        else:
            rho[mask] = rho_base

    return rho


def bcs_level_density(E, a_parameter, delta, spin=None):
    """
    BCS 对关联修正的能级密度。

    ρ_{BCS}(E) = ρ_{FG}(E + Δ) · tanh(√(a(E + Δ)) · Δ / (E + Δ))

    其中 Δ 为对能隙。

    参数
    ----
    E : float 或 ndarray
        激发能 (MeV)
    a_parameter : float
        能级密度参数
    delta : float
        对能隙 (MeV)
    spin : float, optional
        自旋

    返回
    ----
    rho : float 或 ndarray
        能级密度
    """
    E_eff = E + delta
    rho_fg = bethe_formula(E_eff, a_parameter, spin)
    sqrt_aE = np.sqrt(a_parameter * np.maximum(E_eff, 1e-10))
    suppression = np.tanh(sqrt_aE * delta / np.maximum(E_eff, 1e-10))
    return rho_fg * suppression


def log_normal_pdf(x, mu, sigma):
    """
    对数正态概率密度函数。

    f(x; μ, σ) = 1/(x σ √(2π)) · exp[-(ln x - μ)²/(2σ²)]

    参数
    ----
    x : float 或 ndarray
        正值随机变量（如能级间距）
    mu : float
        ln x 的均值
    sigma : float
        ln x 的标准差

    返回
    ----
    pdf : float 或 ndarray
        概率密度
    """
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        xm = x[mask]
        pdf[mask] = (1.0 / (xm * sigma * sqrt(2.0 * pi))) * np.exp(
            -((np.log(xm) - mu) ** 2) / (2.0 * sigma ** 2)
        )
    return pdf


def log_normal_cdf(x, mu, sigma):
    """
    对数正态累积分布函数。

    F(x; μ, σ) = (1/2) [1 + erf((ln x - μ)/(σ√2))]
    """
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + erf((log(x) - mu) / (sigma * sqrt(2.0))))


def log_normal_sample(mu, sigma, size=1, seed=None):
    """
    对数正态分布随机采样（基于 log_normal_sample 思想）。

    方法：若 X ~ N(μ, σ²)，则 Y = exp(X) ~ LogNormal(μ, σ²)。
    """
    rng = np.random.default_rng(seed)
    normal_samples = rng.normal(loc=mu, scale=sigma, size=size)
    return np.exp(normal_samples)


def level_spacing_distribution(s, regime='goe'):
    """
    理论能级间距分布。

    参数
    ----
    s : ndarray
        归一化能级间距（平均间距为 1）
    regime : str
        'poisson', 'goe', 'gue', 'gse' 之一

    返回
    ----
    P : ndarray
        间距分布密度
    """
    s = np.asarray(s, dtype=float)
    if regime == 'poisson':
        return np.exp(-s)
    elif regime == 'goe':
        return (pi / 2.0) * s * np.exp(-pi * s * s / 4.0)
    elif regime == 'gue':
        return (32.0 / pi ** 2) * s * s * np.exp(-4.0 * s * s / pi)
    elif regime == 'gse':
        return (2.0 ** 18 / (3.0 ** 6 * pi ** 3)) * s ** 4 * np.exp(-64.0 * s * s / (9.0 * pi))
    else:
        raise ValueError("regime 必须是 'poisson', 'goe', 'gue', 'gse' 之一")


def unfolding_spectrum(energies):
    """
    能谱展开（unfolding）：将原始能谱映射为平均间距为 1 的序列。

    方法：
    1. 对能级排序
    2. 拟合累积能级密度 N(E) = a E^b（或更复杂的平滑函数）
    3. 变换：ε_i = N(E_i)
    4. 间距：s_i = ε_{i+1} - ε_i

    参数
    ----
    energies : ndarray
        一组能级能量 (MeV)

    返回
    ----
    s : ndarray
        归一化能级间距
    N_smooth : callable
        平滑累积能级密度函数
    """
    E_sorted = np.sort(energies)
    n_levels = len(E_sorted)

    # 累积计数
    N_cum = np.arange(1, n_levels + 1)

    # 用低阶多项式拟合累积密度
    # 取对数：log N ≈ b log E + log a
    mask = E_sorted > 1e-3
    if np.sum(mask) > 3:
        logE = np.log(E_sorted[mask])
        logN = np.log(N_cum[mask])
        coeffs = np.polyfit(logE, logN, deg=2)
        poly = np.poly1d(coeffs)

        def N_smooth(E):
            Ea = np.asarray(E, dtype=float)
            val = np.exp(poly(np.log(np.maximum(Ea, 1e-10))))
            return val
    else:
        def N_smooth(E):
            return np.asarray(E, dtype=float) / np.mean(np.diff(E_sorted))

    epsilon = N_smooth(E_sorted)
    s = np.diff(epsilon)
    # 进一步归一化使平均值为 1
    if len(s) > 0 and np.mean(s) > 0:
        s = s / np.mean(s)

    return s, N_smooth


def nuclear_level_density_parameter(A, shell_correction=0.0):
    """
    计算能级密度参数 a。

    经验公式：a ≈ A / 8 MeV⁻¹
    壳修正：a_eff = a [1 + δE_shell / E] 的简化形式
    """
    a_base = A / 8.0
    # 壳修正通常降低低激发能区的能级密度
    a_eff = a_base * (1.0 + 0.1 * shell_correction)
    return max(a_eff, 0.1)


def total_level_density_table(A, E_max=20.0, n_points=100):
    """
    生成核素 A 的能级密度表。

    返回
    ----
    energies : ndarray
        激发能网格 (MeV)
    rho_total : ndarray
        总能级密度
    rho_positive : ndarray
        正宇称能级密度
    rho_negative : ndarray
        负宇称能级密度
    """
    a = nuclear_level_density_parameter(A)
    delta = 12.0 / sqrt(A)
    energies = np.linspace(0.5, E_max, n_points)

    rho_total = bethe_formula(energies, a)
    rho_bcs = bcs_level_density(energies, a, delta)

    # 宇称分布近似：低能区正宇称占优，高能区趋于 1:1
    f_pos = 0.5 + 0.5 * np.exp(-energies / 5.0)
    rho_positive = rho_bcs * f_pos
    rho_negative = rho_bcs * (1.0 - f_pos)

    return energies, rho_total, rho_positive, rho_negative
