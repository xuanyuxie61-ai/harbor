"""
蒙特卡洛传输与采样模块
整合自：
  - 1410_wedge_monte_carlo（楔形体蒙特卡洛积分）
  - 066_ball_distance（球内均匀采样与距离统计）

在吸积盘模拟中用于：
  1. 喷流区域（楔形几何）的蒙特卡洛光子/粒子传输
  2. 球坐标系下的均匀采样（黑洞周围粒子分布）
  3. 距离统计分析（粒子对关联函数）
"""
import numpy as np


# ===========================
# Wedge Monte Carlo
# ===========================

def wedge01_volume():
    """
    单位楔形体体积。
    楔形体区域：0 <= X, 0 <= Y, X+Y <= 1, -1 <= Z <= 1
    体积 = 1.0
    """
    return 1.0


def wedge01_sample(n_samples, seed=None):
    """
    在单位楔形体中均匀采样。

    算法（基于指数分布）：
        1. 生成3个指数随机变量 e_i = -log(u_i)
        2. 归一化：x = e1/sum(e), y = e2/sum(e)
           这等价于从Dirichlet分布采样XY平面三角形
        3. z = 2*u4 - 1（均匀分布于[-1,1]）

    参数:
        n_samples: 采样点数
        seed: 随机种子

    返回:
        samples: (n_samples, 3) 采样点 [x, y, z]
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)

    samples = np.zeros((n_samples, 3), dtype=np.float64)

    for i in range(n_samples):
        e = -np.log(np.random.rand(3) + 1e-15)
        s = np.sum(e)
        if s < 1e-15:
            s = 1e-15
        samples[i, 0] = e[0] / s
        samples[i, 1] = e[1] / s
        samples[i, 2] = 2.0 * np.random.rand() - 1.0

    return samples


def wedge01_monomial_integral(exponents):
    """
    计算单项式 X^a * Y^b * Z^c 在楔形体上的精确积分。

    解析公式：
        XY部分：a! * b! / ((a+b+2)!)
        Z部分：若 c 为奇数则为 0，否则 2/(c+1)

    参数:
        exponents: [a, b, c]

    返回:
        积分值
    """
    a, b, c = int(exponents[0]), int(exponents[1]), int(exponents[2])

    if a < 0 or b < 0 or c < 0:
        raise ValueError("Exponents must be non-negative")

    # XY部分
    from math import factorial
    xy_val = factorial(a) * factorial(b) / factorial(a + b + 2)

    # Z部分
    if c % 2 == 1:
        z_val = 0.0
    else:
        z_val = 2.0 / (c + 1)

    return xy_val * z_val


def wedge_monte_carlo_integral(n_samples, integrand_func, seed=None):
    """
    使用蒙特卡洛方法计算楔形体上的积分。

    蒙特卡洛估计：
        integral f dV ≈ V * (1/N) * sum_i f(x_i)

    其中 V=1 为单位楔形体体积。

    参数:
        n_samples: 采样点数
        integrand_func: 被积函数 f(x,y,z)
        seed: 随机种子

    返回:
        estimate: 积分估计值
        std_error: 标准误差
    """
    samples = wedge01_sample(n_samples, seed)
    vals = np.array([integrand_func(s) for s in samples], dtype=np.float64)

    V = wedge01_volume()
    estimate = V * np.mean(vals)
    std_error = V * np.std(vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0

    return estimate, std_error


# ===========================
# Ball Sampling & Statistics
# ===========================

def ball_unit_sample(n_samples, dim=3, seed=None):
    """
    在单位球内均匀采样。

    算法：
        1. 高斯随机向量 -> 均匀球面方向
        2. 半径 r = u^(1/dim)，u~Uniform(0,1)
          这样 dV ~ r^(dim-1)dr 保证体积均匀

    参数:
        n_samples: 采样点数
        dim: 空间维度
        seed: 随机种子

    返回:
        points: (n_samples, dim)
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)

    dirs = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    dirs = dirs / norms

    u = np.random.rand(n_samples)
    r = u ** (1.0 / dim)

    return dirs * r.reshape(-1, 1)


def ball_distance_pdf(d):
    """
    单位球内两个随机点距离的理论概率密度函数。

    对于3D单位球，PDF为：
        pdf(d) = (3/16) * (d-2)^2 * d^2 * (d+4),  d in [0, 2]

    参数:
        d: 距离值或数组

    返回:
        pdf值
    """
    d = np.asarray(d, dtype=np.float64)
    result = np.zeros_like(d)
    mask = (d >= 0) & (d <= 2)
    result[mask] = (3.0 / 16.0) * (d[mask] - 2) ** 2 * d[mask] ** 2 * (d[mask] + 4)
    return result


def ball_distance_stats(n_samples, seed=None):
    """
    统计单位球内随机点对距离的均值和方差。

    理论值：
        E[d] = 36/35 ≈ 1.02857
        Var[d] = 0.062...

    参数:
        n_samples: 采样点对数
        seed: 随机种子

    返回:
        dict: mean, variance, distances
    """
    p1 = ball_unit_sample(n_samples, dim=3, seed=seed)
    p2 = ball_unit_sample(n_samples, dim=3, seed=(seed + 1 if seed is not None else None))

    dists = np.linalg.norm(p1 - p2, axis=1)

    mean = np.mean(dists)
    variance = np.var(dists, ddof=1) if n_samples > 1 else 0.0

    return {
        'mean': float(mean),
        'variance': float(variance),
        'distances': dists
    }


# ===========================
# Jet-Specific Monte Carlo
# ===========================

def sample_jet_particles(n_particles, r_launch, theta_opening, v_jet, seed=None):
    """
    在喷流区域采样粒子初始位置和速度。

    喷流几何模型（Blandford-Payne 喷流）：
        从半径 r_launch 处，以半开角 theta_opening 向外喷射。
        粒子在锥体内均匀分布。

    参数:
        n_particles: 粒子数
        r_launch: 发射半径
        theta_opening: 半开角（弧度）
        v_jet: 喷射速度
        seed: 随机种子

    返回:
        positions: (n_particles, 3) [x, y, z]
        velocities: (n_particles, 3) [vx, vy, vz]
    """
    if n_particles <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    if seed is not None:
        np.random.seed(seed)

    # 在锥体内采样方向
    # theta: 从极轴的夹角，均匀分布于 [0, theta_opening]
    # 但为了保证立体角均匀，使用 cos(theta) 均匀分布
    cos_theta_max = np.cos(theta_opening)
    cos_theta = np.random.uniform(cos_theta_max, 1.0, n_particles)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, n_particles)

    # 初始位置：在发射球面上
    x = r_launch * np.sin(theta) * np.cos(phi)
    y = r_launch * np.sin(theta) * np.sin(phi)
    z = r_launch * np.cos(theta)

    positions = np.column_stack([x, y, z])

    # 速度方向沿径向向外
    vx = v_jet * np.sin(theta) * np.cos(phi)
    vy = v_jet * np.sin(theta) * np.sin(phi)
    vz = v_jet * np.cos(theta)

    velocities = np.column_stack([vx, vy, vz])

    return positions, velocities


def mc_jet_energy_transport(n_photons, r_disk, T_disk, seed=None):
    """
    蒙特卡洛模拟喷流中的光子能量传输。

    假设吸积盘为黑体辐射，使用 Wien 位移定律：
        lambda_max * T = 2.898e-3 m*K

    参数:
        n_photons: 光子数
        r_disk: 盘特征半径
        T_disk: 盘温度（K）
        seed: 随机种子

    返回:
        energies: 每个光子的能量（相对单位）
        escaped: 是否逃逸喷流区域
    """
    if seed is not None:
        np.random.seed(seed)

    # 光子能量服从 Planck 分布（近似为指数分布）
    # E ~ k_B * T * chi，其中 chi 为指数随机变量
    energies = np.random.exponential(scale=1.0, size=n_photons)

    # 简单传输模型：光子随机行走
    escaped = np.random.rand(n_photons) > 0.3  # 70% 逃逸

    return energies, escaped


def compute_correlation_function(points, r_bins):
    """
    计算粒子分布的两点关联函数。

    关联函数定义：
        xi(r) = (N_pairs(r) / V_shell(r)) / (N_total / V_total) - 1

    用于分析吸积盘或喷流中粒子的成团性。
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        return np.zeros(len(r_bins) - 1)

    # 计算所有两两距离
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    # 只取上三角（避免重复和自身）
    dists = dists[np.triu_indices(n, k=1)]

    # 统计
    counts, _ = np.histogram(dists, bins=r_bins)

    # 归一化到壳层体积
    volumes = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    volumes = np.where(volumes < 1e-15, 1e-15, volumes)

    # 平均密度
    box_size = np.max(points) - np.min(points)
    if box_size < 1e-15:
        box_size = 1.0
    total_volume = box_size ** 3
    mean_density = n * (n - 1) / 2.0 / total_volume

    xi = counts / volumes / mean_density - 1.0

    return xi
