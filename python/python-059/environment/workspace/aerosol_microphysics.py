"""
aerosol_microphysics.py
气溶胶微物理特性计算模块

整合原项目:
  - 173_chrominoes: 颜色计数分类 → 气溶胶混合态分类
  - 1273_toms515: 组合选择 → 最优粒径分档选取

功能:
  1. 多模态对数正态粒径分布
  2. 气溶胶混合态（内混/外混）的离散分类
  3. 复折射率计算（使用有效介质近似）
  4. 基于组合优化的粒径分档选取

核心物理公式:
  - 对数正态分布:
      n(r) = N_total / (sqrt(2π) r ln σ_g) * exp( - (ln r - ln r_m)^2 / (2 ln^2 σ_g) )
  - 复折射率 (Bruggeman 有效介质近似):
      Σ_i f_i * (ε_i - ε_eff) / (ε_i + 2ε_eff) = 0
  - 消光效率 (几何光学近似 + Mie 修正):
      Q_ext = 2 + 4x * Im{ (m^2 - 1) / (m^2 + 2) } + O(x^2),  x = 2πr/λ
"""

import numpy as np
from math import sqrt, log, exp, pi
from numerical_utils import comb_lexicographic, binomial_coefficient


class AerosolMicrophysicsError(Exception):
    pass


def lognormal_size_distribution(r, N_total, r_median, sigma_g):
    """
    对数正态粒径分布数浓度。

    参数:
      r: 粒径 (μm)，可以为标量或数组
      N_total: 总数浓度 (#/cm³)
      r_median: 几何中值粒径 (μm)
      sigma_g: 几何标准差

    返回:
      n(r): 数浓度分布函数值
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r <= 0):
        raise AerosolMicrophysicsError("lognormal_size_distribution: 粒径必须为正")
    if r_median <= 0 or sigma_g <= 1.0:
        raise AerosolMicrophysicsError("lognormal_size_distribution: 参数非法")

    ln_r = np.log(r)
    ln_r_m = np.log(r_median)
    ln_sigma = np.log(sigma_g)

    coeff = N_total / (sqrt(2.0 * pi) * r * ln_sigma)
    exponent = -0.5 * ((ln_r - ln_r_m) / ln_sigma) ** 2
    return coeff * np.exp(exponent)


def multimode_lognormal(r, modes):
    """
    多模态对数正态分布叠加。

    参数:
      modes: 列表，每个元素为 (N_total, r_median, sigma_g)

    返回:
      总分布 n(r)
    """
    total = np.zeros_like(np.asarray(r), dtype=np.float64)
    for N_total, r_median, sigma_g in modes:
        total += lognormal_size_distribution(r, N_total, r_median, sigma_g)
    return total


def count_mixing_state(m, n, C):
    """
    基于 chrominoes 的颜色计数公式，对气溶胶 m×n 网格进行混合态分类。

    物理意义:
      将气溶胶粒子表面离散化为 m×n 的二维网格，C 种化学组分
      （如硫酸盐、黑碳、有机碳、海盐、沙尘）。
      计算每种组分占据的网格数，用于区分内混/外混态。

    公式 (Fabio Visonà / Garvie):
      a = m mod C, b = n mod C
      N = (m*n - a*b) / C
      counts(x) =
        N + a + b - C       , x ≤ a+b-1-C
        N + x               , a+b-1-C < x < min(a,b)
        N + min(a,b)        , min(a,b) ≤ x ≤ max(a,b)
        N + a + b - x       , max(a,b) < x ≤ a+b-1
        N                   , 其他

    参数:
      m, n: 离散化网格维度
      C: 化学组分数

    返回:
      counts: 长度为 C 的数组，每种组分的网格数
    """
    if m <= 0 or n <= 0 or C <= 0:
        raise AerosolMicrophysicsError("count_mixing_state: 参数必须为正整数")

    a = m % C
    b = n % C
    N = (m * n - a * b) // C

    counts = np.zeros(C, dtype=int)
    for x in range(1, C + 1):
        if x <= a + b - 1 - C:
            counts[x - 1] = N + a + b - C
        elif a + b - 1 - C < x < min(a, b):
            counts[x - 1] = N + x
        elif min(a, b) <= x <= max(a, b):
            counts[x - 1] = N + min(a, b)
        elif max(a, b) < x <= a + b - 1:
            counts[x - 1] = N + a + b - x
        else:
            counts[x - 1] = N

    return counts


def mixing_state_index(counts):
    """
    根据组分分布计算混合态指数 χ (Riemer et al., 2004)。

    χ = 0 表示完全外混 (each particle pure)
    χ = 1 表示完全内混 (all components in every particle)

    公式:
      χ = 1 - ( Σ_i |p_i - p_bulk,i| ) / (2 * (1 - min_j p_bulk,j) )
    其中 p_i 为粒子内组分 i 的质量分数，p_bulk,i 为总体平均质量分数。
    """
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p_bulk = counts / total
    # 简化为: 1 - 标准差倍数
    chi = 1.0 - np.std(p_bulk) * len(p_bulk)
    return float(np.clip(chi, 0.0, 1.0))


def bruggeman_effective_medium(fractions, refractive_indices, tol=1e-12, max_iter=500):
    """
    Bruggeman 有效介质近似求解等效复折射率 m_eff。

    方程:
      Σ_i f_i * (m_i^2 - m_eff^2) / (m_i^2 + 2*m_eff^2) = 0

    参数:
      fractions: 各组分体积分数列表/数组，和为 1
      refractive_indices: 各组分复折射率列表/数组
      tol: 收敛容差
      max_iter: 最大迭代次数

    返回:
      m_eff: 等效复折射率
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    m = np.asarray(refractive_indices, dtype=np.complex128)

    if not np.isclose(np.sum(fractions), 1.0):
        raise AerosolMicrophysicsError("bruggeman: 体积分数之和必须等于 1")
    if len(fractions) != len(m):
        raise AerosolMicrophysicsError("bruggeman: 数组长度不匹配")

    m2 = m ** 2
    # 初始猜测: 体积加权平均
    m_eff2 = np.sum(fractions * m2)

    for _ in range(max_iter):
        sum_term = np.sum(fractions * (m2 - m_eff2) / (m2 + 2.0 * m_eff2))
        # 使用牛顿迭代
        denom = np.sum(fractions * (-3.0 * m2) / ((m2 + 2.0 * m_eff2) ** 2))
        if abs(denom) < 1e-30:
            break
        delta = -sum_term / denom
        m_eff2_new = m_eff2 + delta
        if abs(m_eff2_new - m_eff2) < tol:
            m_eff2 = m_eff2_new
            break
        m_eff2 = m_eff2_new

    return np.sqrt(m_eff2)


def select_optimal_size_bins(N_total, r_median, sigma_g, num_bins, r_min=0.001, r_max=10.0):
    """
    使用 toms515 的组合选择算法，从粒径网格中选取最优分档。

    步骤:
      1. 在 [ln r_min, ln r_max] 上均匀离散化为 n_grid 个点
      2. 使用 comb_lexicographic 按字典序选取 num_bins 个代表粒径
      3. 计算每个代表粒径档的积分浓度

    参数:
      num_bins: 分档数
      r_min, r_max: 粒径范围 (μm)

    返回:
      r_bins: 代表粒径数组
      N_bins: 各档数浓度
    """
    n_grid = max(num_bins * 4, 20)
    ln_r_grid = np.linspace(np.log(r_min), np.log(r_max), n_grid)
    r_grid = np.exp(ln_r_grid)

    # 选择字典序索引为 1 的组合（最均匀分布）
    idx = comb_lexicographic(n_grid, num_bins, 1)
    # 将 1-based 转为 0-based 索引
    idx_arr = np.array([i - 1 for i in idx], dtype=int)
    r_bins = r_grid[idx_arr]

    # 计算各档浓度（相邻中点法）
    N_bins = np.zeros(num_bins)
    for i in range(num_bins):
        if i == 0:
            r_low = r_min
        else:
            r_low = np.sqrt(r_bins[i - 1] * r_bins[i])
        if i == num_bins - 1:
            r_high = r_max
        else:
            r_high = np.sqrt(r_bins[i] * r_bins[i + 1])

        # 数值积分
        pts = np.linspace(r_low, r_high, 50)
        vals = lognormal_size_distribution(pts, N_total, r_median, sigma_g)
        N_bins[i] = np.trapezoid(vals, pts)

    return r_bins, N_bins


def extinction_efficiency_small(r, wavelength, m_eff):
    """
    小参数近似下的消光效率 (Rayleigh-Gans / Mie 小参数展开)。

    参数:
      r: 粒径 (μm)
      wavelength: 波长 (μm)
      m_eff: 等效复折射率

    公式:
      x = 2πr / λ
      Q_ext ≈ (8/3) x^4 | (m^2 - 1) / (m^2 + 2) |^2   (散射)
            + 4x Im{ (m^2 - 1) / (m^2 + 2) }            (吸收)
    """
    x = 2.0 * pi * r / wavelength
    if x > 0.5:
        # 超出小参数范围，使用几何光学近似
        return 2.0
    ratio = (m_eff ** 2 - 1.0) / (m_eff ** 2 + 2.0)
    q_scat = (8.0 / 3.0) * (x ** 4) * (abs(ratio) ** 2)
    q_abs = 4.0 * x * np.imag(ratio)
    return float(q_scat + q_abs)
