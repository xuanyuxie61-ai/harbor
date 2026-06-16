"""
distribution_sampler.py - 等离子体粒子分布采样

本模块实现等离子体粒子速度分布的初始化采样。
在 Vlasov-Maxwell 仿真中，电子速度分布函数 f(x, v, t) 的
初始化和采样是计算的核心起点。

实现的分布函数:

1. Maxwell-Boltzmann 分布 (热平衡):
   f(v) = n_0 / sqrt(2*pi*v_th^2) * exp(-v^2 / (2*v_th^2))

2. 双温分布 (激光加热后):
   f(v) = alpha * f_cold(v) + (1-alpha) * f_hot(v)
   其中 f_cold, f_hot 分别为冷、热 Maxwell 分布

3. Log-normal 分布 (超热电子):
   在强激光等离子体相互作用中, 通过尾波场加速或
   J x B 加热产生的超热电子呈现近似 log-normal 分布:
       f(E) = 1/(E*sigma*sqrt(2*pi)) * exp(-(ln(E)-mu)^2 / (2*sigma^2))
   其中 E 为粒子能量, mu 和 sigma 为分布参数。

4. 2D 直方图采样:
   从 2D 相空间分布 (x, v_x) 中进行逆CDF采样,
   用于加载预先计算的分布函数数据。

采样方法:
  - 逆 CDF 变换法: u = CDF(v) -> v = CDF^{-1}(u), u ~ U(0,1)
  - 拒绝采样法: 用于复杂分布
  - Box-Muller 变换: 用于 Gaussian 采样
"""

import numpy as np


# ============================================================
# Log-normal 分布函数集
# ============================================================

def log_normal_pdf(x, mu, sigma):
    """Log-normal 概率密度函数。

    超热电子能量分布模型:
        f(x) = 1/(x * sigma * sqrt(2*pi)) * exp(-(ln(x) - mu)^2 / (2*sigma^2))

    适用条件: x > 0, sigma > 0

    物理背景:
        在激光尾波场加速中, 被捕获电子的能量分布近似 log-normal,
        因为加速过程可建模为随机乘法过程 (中心极限定理的乘法版本)。

    矩:
        均值 = exp(mu + sigma^2/2)
        方差 = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)

    Parameters
    ----------
    x : float or ndarray
        自变量 (> 0)
    mu : float
        对数均值参数
    sigma : float
        对数标准差参数 (> 0)

    Returns
    -------
    pdf : float or ndarray
        概率密度
    """
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    mask = x > 0
    xm = x[mask]
    log_xm = np.log(xm)
    pdf[mask] = np.exp(-0.5 * ((log_xm - mu) / sigma) ** 2) / (
        xm * sigma * np.sqrt(2.0 * np.pi)
    )
    return pdf


def log_normal_cdf(x, mu, sigma):
    """Log-normal 累积分布函数。

    F(x) = 0.5 * (1 + erf((ln(x) - mu) / (sigma * sqrt(2))))

    Parameters
    ----------
    x : float or ndarray
        自变量 (> 0)
    mu : float
        对数均值
    sigma : float
        对数标准差

    Returns
    -------
    cdf : float or ndarray
        累积概率
    """
    from scipy.special import erf
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x)
    mask = x > 0
    xm = x[mask]
    z = (np.log(xm) - mu) / (sigma * np.sqrt(2.0))
    cdf[mask] = 0.5 * (1.0 + erf(z))
    return cdf


def log_normal_inverse_cdf(p, mu, sigma):
    """Log-normal 逆累积分布函数 (分位数函数)。

    给定均匀随机变量 p ~ U(0,1), 返回:
        x = exp(mu + sigma * sqrt(2) * erfinv(2*p - 1))

    这是逆 CDF 采样法的核心函数。

    Parameters
    ----------
    p : float or ndarray
        累积概率值 (0 < p < 1)
    mu : float
        对数均值
    sigma : float
        对数标准差

    Returns
    -------
    x : float or ndarray
        对应的分位数
    """
    from scipy.special import erfinv
    p = np.clip(np.asarray(p, dtype=float), 1.0e-15, 1.0 - 1.0e-15)
    z = erfinv(2.0 * p - 1.0)
    return np.exp(mu + sigma * np.sqrt(2.0) * z)


def log_normal_mean(mu, sigma):
    """Log-normal 分布的均值。

    E[X] = exp(mu + sigma^2 / 2)

    Parameters
    ----------
    mu : float
        对数均值
    sigma : float
        对数标准差

    Returns
    -------
    mean : float
        分布均值
    """
    return np.exp(mu + sigma ** 2 / 2.0)


def log_normal_variance(mu, sigma):
    """Log-normal 分布的方差。

    Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)

    Parameters
    ----------
    mu : float
        对数均值
    sigma : float
        对数标准差

    Returns
    -------
    var : float
        分布方差
    """
    return (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)


def log_normal_sample(mu, sigma, n_samples, seed=None):
    """从 log-normal 分布中采样。

    使用逆 CDF 变换法:
        1. 生成 u_i ~ U(0, 1)
        2. x_i = F^{-1}(u_i) = exp(mu + sigma*sqrt(2)*erfinv(2*u_i - 1))

    Parameters
    ----------
    mu : float
        对数均值
    sigma : float
        对数标准差
    n_samples : int
        采样数量
    seed : int, optional
        随机种子

    Returns
    -------
    samples : ndarray
        采样值
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0.0, 1.0, n_samples)
    return log_normal_inverse_cdf(u, mu, sigma)


# ============================================================
# 2D 直方图分布采样
# ============================================================

def compute_discrete_cdf_2d(pdf):
    """计算 2D 离散分布的累积分布函数。

    按照字典序 (先行后列) 累加概率:
        CDF(i,j) = sum_{i'<=i, j'<=j} pdf(i', j')

    最终 CDF[-1, -1] 应等于 1 (归一化)。

    Parameters
    ----------
    pdf : ndarray (N1, N2)
        2D 概率密度 (需归一化)

    Returns
    -------
    cdf : ndarray (N1, N2)
        累积分布函数
    """
    # 归一化
    total = np.sum(pdf)
    if total > 0:
        pdf_norm = pdf / total
    else:
        pdf_norm = pdf.copy()

    cdf = np.cumsum(pdf_norm.ravel()).reshape(pdf.shape)
    return cdf


def sample_from_histogram_2d(cdf, grid_x, grid_v, n_samples, seed=None):
    """从 2D 直方图分布中采样 (逆 CDF 变换法)。

    对于每个均匀随机数 u_k:
        1. 找到 CDF 中满足 CDF(i,j) >= u_k 的最小 (i,j)
        2. 在该网格单元内均匀随机取点

    Parameters
    ----------
    cdf : ndarray (N1, N2)
        2D 累积分布函数
    grid_x : ndarray (N1,)
        x 方向网格坐标
    grid_v : ndarray (N2,)
        v 方向网格坐标
    n_samples : int
        采样数量
    seed : int, optional
        随机种子

    Returns
    -------
    samples_x : ndarray (n_samples,)
        x 坐标采样
    samples_v : ndarray (n_samples,)
        v 坐标采样
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0.0, 1.0, n_samples)

    cdf_flat = cdf.ravel()
    n1, n2 = cdf.shape

    samples_x = np.zeros(n_samples)
    samples_v = np.zeros(n_samples)

    for k in range(n_samples):
        # 找到 CDF 区间
        idx = np.searchsorted(cdf_flat, u[k])
        idx = min(idx, len(cdf_flat) - 1)

        i = idx // n2
        j = idx % n2

        # 在网格单元内均匀插值
        dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
        dv = grid_v[1] - grid_v[0] if len(grid_v) > 1 else 1.0

        samples_x[k] = grid_x[min(i, len(grid_x) - 1)] + rng.uniform() * dx
        samples_v[k] = grid_v[min(j, len(grid_v) - 1)] + rng.uniform() * dv

    return samples_x, samples_v


# ============================================================
# 等离子体分布函数初始化
# ============================================================

def maxwellian_distribution(v, n0=1.0, vth=1.0):
    """Maxwell-Boltzmann 速度分布。

    f(v) = n_0 / sqrt(2*pi*v_th^2) * exp(-v^2 / (2*v_th^2))

    满足归一化: integral f(v) dv = n_0

    Parameters
    ----------
    v : ndarray
        速度数组
    n0 : float
        数密度
    vth : float
        热速度

    Returns
    -------
    f : ndarray
        分布函数值
    """
    return n0 / np.sqrt(2.0 * np.pi * vth ** 2) * np.exp(-v ** 2 / (2.0 * vth ** 2))


def two_temperature_distribution(v, n0, vth_cold, vth_hot, alpha):
    """双温 Maxwell 分布。

    f(v) = alpha * f_cold(v; vth_cold) + (1-alpha) * f_hot(v; vth_hot)

    模拟激光加热后的等离子体: 主体为冷电子, 少量被加热的热电子。

    Parameters
    ----------
    v : ndarray
        速度数组
    n0 : float
        总数密度
    vth_cold : float
        冷组分热速度
    vth_hot : float
        热组分热速度
    alpha : float
        冷组分比例 (0 < alpha < 1)

    Returns
    -------
    f : ndarray
        分布函数值
    """
    f_cold = maxwellian_distribution(v, 1.0, vth_cold)
    f_hot = maxwellian_distribution(v, 1.0, vth_hot)
    return n0 * (alpha * f_cold + (1.0 - alpha) * f_hot)


def relativistic_correction_factor(v, vth):
    """相对论修正因子。

    对于高温等离子体 (v_th ~ c), Maxwell 分布需要相对论修正。
    修正因子 (展开到 v^2/c^2 阶):
        gamma_rel = 1 + 3/2 * (v_th/c)^2 + 15/8 * (v_th/c)^4 + ...

    归一化修正:
        C_rel = 1 / (1 + 3/2 * theta + 15/8 * theta^2)
    其中 theta = k_B*T / (m_e*c^2) 为无量纲温度。

    Parameters
    ----------
    v : ndarray
        速度 (归一化到 c)
    vth : float
        热速度 (归一化到 c)

    Returns
    -------
    correction : ndarray
        相对论修正因子
    """
    beta2 = (v / 1.0) ** 2  # v 已归一化到 c
    theta = vth ** 2
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta2, 1.0e-10))
    # Jüttner 分布的近似修正
    correction = gamma * np.exp(-0.5 * beta2 / theta) * (1.0 + 0.5 * theta)
    return correction


def initialize_phase_space(config):
    """初始化完整的相空间分布函数 f(x, v, t=0)。

    构建 (N_x, N_v) 的 2D 分布函数数组,
    其中每个空间位置有一个局部的速度分布。

    初始分布:
        f(x, v, 0) = n(x) * [alpha * M(v; vth) + (1-alpha) * LN(v; mu, sigma)]

    其中 n(x) 为密度剖面, M 为 Maxwell 分布, LN 为 log-normal 超热尾部。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置

    Returns
    -------
    f : ndarray (N_x, N_v)
        初始分布函数
    phase_space_info : dict
        相空间信息
    """
    x = config.x
    v = config.v
    n_profile = config.plasma_density_profile(x)

    f = np.zeros((config.N_x, config.N_v))

    # 超热电子比例 (随密度增加而减小)
    alpha_hot = 0.05 * np.exp(-n_profile / config.n_max)

    for i in range(config.N_x):
        ni = n_profile[i]
        if ni < 1.0e-10:
            continue

        # 冷组分 (Maxwell)
        f_cold = maxwellian_distribution(v, ni * (1.0 - alpha_hot[i]), 1.0)

        # 热组分 (log-normal 在速度空间的映射)
        # 将 log-normal 映射到速度空间: E = 0.5 * m * v^2 -> v = sqrt(2E/m)
        E_grid = 0.5 * v ** 2
        E_safe = np.maximum(E_grid, 1.0e-10)
        f_hot_E = log_normal_pdf(E_safe, config.log_normal_mu, config.log_normal_sigma)
        # 从能量分布转换到速度分布: f(v) = f_E(E) * |dE/dv| = f_E(E) * |v|
        f_hot_v = f_hot_E * np.abs(v)
        # 归一化
        integral = np.sum(f_hot_v) * config.dv
        if integral > 0:
            f_hot_v = f_hot_v / integral * ni * alpha_hot[i]

        f[i, :] = f_cold + f_hot_v

    # 计算矩
    density = np.sum(f, axis=1) * config.dv
    mean_velocity = np.where(
        density > 1.0e-10,
        np.sum(f * v[np.newaxis, :], axis=1) * config.dv / density,
        0.0
    )
    temperature = np.where(
        density > 1.0e-10,
        np.sum(f * (v[np.newaxis, :] - mean_velocity[:, np.newaxis]) ** 2, axis=1) * config.dv / density,
        0.0
    )

    phase_space_info = {
        'density': density,
        'mean_velocity': mean_velocity,
        'temperature': temperature,
        'total_particles': np.sum(f) * config.dv * config.dx,
        'max_f': np.max(f),
        'min_f': np.min(f),
    }

    return f, phase_space_info
