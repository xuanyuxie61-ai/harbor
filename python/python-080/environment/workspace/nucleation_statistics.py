"""
nucleation_statistics.py
空化核随机成核统计模型

核心物理模型:
1. 空化核尺寸分布（正态对数分布）:
   f(R) = 1/(R σ√(2π)) * exp(-(ln(R/μ))²/(2σ²))

2. 成核概率（经典成核理论 CNT）:
   J = J_0 * exp(-ΔG*/(k_B T))
   ΔG* = 16πσ³/(3(p_v - p_∞)²)

3. 表面成核位点模型:
   将固体表面离散为位点网格，每个位点有独立的成核概率。
   位点激活遵循泊松过程。

4. 统计模拟:
   - 蒙特卡洛采样空化核分布
   - 全牌统计（full_deck）思想：模拟多次独立实验的统计量
   - 空缺概率（supreme_vacancy）：表面位点被占据的概率

映射来源:
- 301_disk01_monte_carlo: 圆盘均匀采样 → 表面位点分布
- 334_ellipsoid_monte_carlo: 椭球采样 → 非球形核分布
- 449_full_deck_simulation: 统计模拟框架 → 多次独立成核实验统计
- 1183_supreme_vacancy: 概率累积模型 → 位点激活概率
"""

import numpy as np
from utils import disk01_sample, ellipsoid_sample, BOLTZMANN

# =====================================================================
# 物理参数
# =====================================================================
SURFACE_TENSION = 0.0728  # [N/m]
VAPOR_PRESSURE = 2338.0   # [Pa]
AMBIENT_TEMPERATURE = 293.15  # [K]


def lognormal_nuclei_distribution(R, mu_R, sigma_R):
    """
    空化核尺寸的对数正态分布概率密度。
    f(R) = 1/(R σ√(2π)) * exp(-(ln(R/μ))²/(2σ²))
    """
    ln_R = np.log(np.maximum(R, 1e-15))
    ln_mu = np.log(mu_R)
    coeff = 1.0 / (R * sigma_R * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((ln_R - ln_mu) / sigma_R) ** 2
    return coeff * np.exp(exponent)


def nucleation_barrier_energy(p_inf, p_v, sigma):
    """
    经典成核理论的临界成核自由能。
    ΔG* = 16πσ³ / (3(p_v - p_∞)²)
    """
    delta_p = p_v - p_inf
    if abs(delta_p) < 1.0:
        delta_p = np.sign(delta_p) * 1.0
    return 16.0 * np.pi * sigma**3 / (3.0 * delta_p**2)


def nucleation_rate(p_inf, p_v, sigma, T, J0=1e30):
    """
    成核速率 [m⁻³ s⁻¹]。
    J = J_0 * exp(-ΔG*/(k_B T))
    """
    delta_G = nucleation_barrier_energy(p_inf, p_v, sigma)
    return J0 * np.exp(-delta_G / (BOLTZMANN * T + 1e-30))


def sample_nuclei_monte_carlo(num_samples, mu_R, sigma_R, method='disk'):
    """
    蒙特卡洛采样空化核尺寸与位置。
    对应 301_disk01_monte_carlo 和 334_ellipsoid_monte_carlo。

    参数:
        num_samples: 采样数量
        mu_R: 中位半径 [m]
        sigma_R: 对数标准差
        method: 'disk' 或 'ellipsoid'
    返回:
        positions: 2 x num_samples 位置（在表面平面内）
        radii: num_samples 核半径
    """
    # 使用逆变换采样从对数正态分布采样半径
    u = np.random.uniform(0.0, 1.0, size=num_samples)
    radii = mu_R * np.exp(sigma_R * np.random.randn(num_samples))

    if method == 'disk':
        # 在单位圆盘内均匀采样位置，再缩放
        positions = disk01_sample(num_samples)
    elif method == 'ellipsoid':
        # 椭球采样模拟非均匀表面
        A = np.array([[2.0, 0.5], [0.5, 1.5]])
        v = np.array([0.0, 0.0])
        positions = ellipsoid_sample(2, num_samples, A, v, 1.0)
    else:
        positions = np.random.randn(2, num_samples)
        norms = np.sqrt(np.sum(positions**2, axis=0))
        norms = np.maximum(norms, 1e-15)
        positions = positions / norms * np.random.uniform(0.0, 1.0, size=num_samples)

    return positions, radii


def full_deck_nucleation_stats(num_experiments, p_inf_range, p_v, sigma, T, surface_area):
    """
    多次独立成核实验的统计量。
    对应 449_full_deck_simulation 的统计框架。

    参数:
        num_experiments: 实验次数
        p_inf_range: 远场压力范围数组 [Pa]
        surface_area: 表面积 [m²]
    返回:
        stats: 字典，包含每次实验的成核数统计
    """
    nucleation_counts = []
    dt = 1e-6  # 时间步 [s]

    for _ in range(num_experiments):
        p_inf = np.random.choice(p_inf_range)
        J = nucleation_rate(p_inf, p_v, sigma, T)
        expected_nuclei = J * surface_area * dt
        # 泊松采样
        num_nuclei = np.random.poisson(max(expected_nuclei, 0.0))
        nucleation_counts.append(num_nuclei)

    nucleation_counts = np.array(nucleation_counts, dtype=float)
    stats = {
        'min': int(np.min(nucleation_counts)),
        'max': int(np.max(nucleation_counts)),
        'mean': np.mean(nucleation_counts),
        'variance': np.var(nucleation_counts),
        'median': np.median(nucleation_counts),
    }
    return stats


def vacancy_activation_probability(num_sites, p_inf, p_v, sigma, T, span_years=1.0):
    """
    表面位点激活概率模型。
    对应 1183_supreme_vacancy 的概率累积模型。

    将每个位点类比为"法官任期"，位点激活类比为"职位空缺"。
    在成核语境下，表示在给定时间跨度内至少一个位点被激活的概率。

    参数:
        num_sites: 表面位点数量
        span_years: 时间跨度（归一化单位）
    返回:
        p_activation: 至少一个位点激活的概率
    """
    J = nucleation_rate(p_inf, p_v, sigma, T)
    # 单个位点在时间跨度内不被激活的概率
    p_no_activate_single = np.exp(-J * span_years * 1e-6)
    # 所有位点都不被激活的概率
    p_no_activate_all = p_no_activate_single ** num_sites
    # 至少一个被激活
    p_activation = 1.0 - p_no_activate_all
    return p_activation


def critical_nuclei_fraction(p_inf, p_v, sigma, T, R0, mu_R, sigma_R):
    """
    计算在给定压力下能存活（不溶解）的空化核比例。
    由 Blake 临界半径决定: R > R_crit 的核才能生长。
    """
    delta_p = p_v - p_inf
    if delta_p >= 0:
        return 1.0
    R_crit = np.sqrt(2.0 * sigma / (3.0 * abs(delta_p)))

    # 对数正态分布的累积分布函数
    from scipy.stats import lognorm
    cdf = lognorm.cdf(R_crit, s=sigma_R, scale=mu_R)
    return 1.0 - cdf


def surface_site_occupancy(num_sites, nuclei_positions, site_radius=1e-5):
    """
    计算表面位点被空化核占据的比例。
    对应 supreme_vacancy 的"职位空缺"统计。
    """
    occupied = set()
    for pos in nuclei_positions.T:
        # 找到最近的位点索引（简化：将平面离散为网格）
        ix = int(pos[0] / site_radius)
        iy = int(pos[1] / site_radius)
        occupied.add((ix, iy))

    occupancy_rate = len(occupied) / max(num_sites, 1)
    return occupancy_rate


def nucleation_event_times(poisson_rate, t_max, num_realizations=10):
    """
    模拟多个独立泊松过程的成核事件时间序列。
    用于分析成核的时间聚集性（clustering）。
    """
    events = []
    for _ in range(num_realizations):
        t = 0.0
        realization = []
        while t < t_max:
            dt = np.random.exponential(1.0 / max(poisson_rate, 1e-15))
            t += dt
            if t < t_max:
                realization.append(t)
        events.append(realization)
    return events
