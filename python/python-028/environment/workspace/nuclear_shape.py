"""
nuclear_shape.py
================
核几何形状采样与统计距离分析模块

本模块融合 polygon_sample、polygon_distance 与 fly_simulation 的核心思想，
将二维多边形采样扩展至三维核表面采样，并计算核内粒子间距离统计量。

功能：
1. 在三维核体积内均匀/非均匀随机采样（基于形变 Fermi 分布）
2. 核子对距离统计（均值、方差、极值）—— 用于短程关联分析
3. 核表面三角形剖分与面积采样
4. Monte Carlo 模拟核子位置涨落

数学基础：
- 核内均匀采样：按体积元 dV = r² sinθ dr dθ dφ 的概率密度采样
- 形变核采样：接受-拒绝方法，按 ρ(r, θ, φ) 分布采样
- 距离统计量：⟨d⟩ = (2/N(N-1)) Σ_{i<j} ||r_i - r_j||
- 对关联函数：g(r) = (V/N²) ⟨Σ_{i≠j} δ(r - |r_i - r_j|)⟩
"""

import numpy as np
from math import sin, cos, sqrt, pi, exp


def uniform_sphere_sample(n_points, R_max, center=(0.0, 0.0, 0.0), seed=42):
    """
    在半径为 R_max 的球内均匀随机采样 n_points 个点。

    方法：
    r = R_max · u^{1/3}   (u ~ Uniform[0,1])
    θ = arccos(2v - 1)    (v ~ Uniform[0,1])
    φ = 2π w              (w ~ Uniform[0,1])

    参数
    ----
    n_points : int
        采样点数
    R_max : float
        球半径 (fm)
    center : tuple
        球心坐标
    seed : int
        随机种子

    返回
    ----
    points : ndarray, shape (n_points, 3)
        采样点坐标
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_points)
    v = rng.random(n_points)
    w = rng.random(n_points)

    r = R_max * u ** (1.0 / 3.0)
    theta = np.arccos(2.0 * v - 1.0)
    phi = 2.0 * pi * w

    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)

    return np.column_stack([x, y, z])


def deformed_fermi_sample(n_points, A, beta2=0.0, gamma=0.0,
                          R0=1.2, a=0.52, seed=42):
    """
    按形变 Fermi 分布在核体积内采样。

    密度分布：
    ρ(r, θ, φ) = ρ₀ / [1 + exp((r - R(θ, φ)) / a)]

    采用接受-拒绝采样：先在包围盒内均匀采样，再按概率 ρ/ρ₀ 接受。

    参数
    ----
    n_points : int
        目标采样点数
    A : int
        质量数
    beta2, gamma : float
        形变参数
    R0, a : float
        半径与弥散参数
    seed : int

    返回
    ----
    points : ndarray
        采样点坐标
    """
    rng = np.random.default_rng(seed)
    R = R0 * (A ** (1.0 / 3.0))
    box_size = 1.5 * R * (1.0 + abs(beta2) + 0.1)

    points = []
    n_trial = 0
    max_trials = n_points * 50

    while len(points) < n_points and n_trial < max_trials:
        # 在包围盒内均匀采样
        xyz = rng.uniform(-box_size, box_size, size=3)
        x, y, z = xyz
        r = sqrt(x * x + y * y + z * z)
        if r < 1e-10:
            n_trial += 1
            continue

        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        # 计算形变半径
        Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(theta) ** 2 - 1.0)
        Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(theta) ** 2 * cos(2.0 * phi)
        R_def = R * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real))

        # Fermi 分布概率
        prob = 1.0 / (1.0 + exp((r - R_def) / a))

        if rng.random() < prob:
            points.append([x, y, z])

        n_trial += 1

    if len(points) < n_points:
        # 补充均匀采样
        extra = uniform_sphere_sample(n_points - len(points), box_size, seed=seed + 1)
        points.extend(extra.tolist())

    return np.array(points[:n_points])


def pairwise_distance_statistics(points):
    """
    计算点集的两两点距离统计量（基于 polygon_distance_stats 思想）。

    参数
    ----
    points : ndarray, shape (N, d)
        N 个 d 维点

    返回
    ----
    stats : dict
        包含 mean, variance, min, max, rms 的字典
    """
    N = len(points)
    if N < 2:
        return {'mean': 0.0, 'variance': 0.0, 'min': 0.0, 'max': 0.0, 'rms': 0.0}

    # 计算所有不同点对距离（向量化避免 O(N²) 显式循环）
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # 取上三角（排除对角线）
    triu_indices = np.triu_indices(N, k=1)
    d = dists[triu_indices]

    return {
        'mean': float(np.mean(d)),
        'variance': float(np.var(d)),
        'min': float(np.min(d)),
        'max': float(np.max(d)),
        'rms': float(np.sqrt(np.mean(d ** 2)))
    }


def pair_correlation_function(points, dr, r_max):
    """
    计算对关联函数 g(r)。

    g(r) = (2V / N(N-1)) · (1 / 4πr² dr) · ⟨Σ_{i<j} δ(r - |r_i - r_j|)⟩

    参数
    ----
    points : ndarray, shape (N, 3)
        粒子坐标
    dr : float
        径向分箱宽度
    r_max : float
        最大径向距离

    返回
    ----
    r_bins : ndarray
        分箱中心
    g_r : ndarray
        对关联函数值
    """
    N = len(points)
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    n_bins = int(r_max / dr)
    r_bins = np.linspace(dr / 2.0, r_max - dr / 2.0, n_bins)
    g_r = np.zeros(n_bins)

    # 计算所有距离
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    triu_indices = np.triu_indices(N, k=1)
    all_dists = dists[triu_indices]

    # 分箱计数
    counts, _ = np.histogram(all_dists, bins=np.linspace(0, r_max, n_bins + 1))

    # 理想气体归一化
    V = (4.0 / 3.0) * pi * r_max ** 3
    rho0 = N / V

    for i in range(n_bins):
        r = r_bins[i]
        shell_volume = 4.0 * pi * r * r * dr
        ideal_count = 0.5 * N * (N - 1) * shell_volume / V
        if ideal_count > 0:
            g_r[i] = counts[i] / ideal_count

    return r_bins, g_r


def monte_carlo_nuclear_radius(A, n_samples=100000, beta2=0.0, gamma=0.0,
                                R0=1.2, seed=123):
    """
    基于 Monte Carlo 采样估算核的等效均匀半径与表面厚度。

    基于 fly_simulation 思想：在核体积内随机投点，统计 r 的分布。

    参数
    ----
    A : int
        质量数
    n_samples : int
        Monte Carlo 采样数
    beta2, gamma : float
        形变参数
    R0 : float
        半径参数
    seed : int

    返回
    ----
    R_eff : float
        等效均匀半径（90% 核子所在半径）
    t_surface : float
        表面厚度（10%-90% 密度点间距）
    R_rms : float
        均方根半径
    """
    points = deformed_fermi_sample(n_samples, A, beta2, gamma, R0, seed=seed)
    radii = np.sqrt(np.sum(points ** 2, axis=1))

    R_rms = float(np.sqrt(np.mean(radii ** 2)))
    R_eff = float(np.percentile(radii, 90))
    r10 = float(np.percentile(radii, 10))
    t_surface = R_eff - r10

    return R_eff, t_surface, R_rms


def triangular_deformation_analysis(n_theta, n_phi, beta2, gamma, R0):
    """
    基于 polygon_triangulate 思想，对核表面进行三角形剖分并计算面积统计。

    将核表面参数化后，每个 (θ, φ) 网格单元近似为一个四边形，
    再剖分为两个三角形，计算总面积与面积涨落。

    参数
    ----
    n_theta, n_phi : int
        网格数
    beta2, gamma : float
        形变参数
    R0 : float
        平均半径

    返回
    ----
    total_area : float
        总表面积
    area_variance : float
        三角形面积方差
    areas : list
        各三角形面积
    """
    from nuclear_grid import deformed_nuclear_surface_grid
    grid, _ = deformed_nuclear_surface_grid(beta2, gamma, R0, n_theta, n_phi)

    areas = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            # 四边形顶点索引
            idx00 = i * n_phi + j
            idx01 = i * n_phi + (j + 1)
            idx10 = (i + 1) * n_phi + j
            idx11 = (i + 1) * n_phi + (j + 1)

            # 剖分为两个三角形
            tri1 = [grid[idx00], grid[idx10], grid[idx01]]
            tri2 = [grid[idx01], grid[idx10], grid[idx11]]

            for tri in [tri1, tri2]:
                a_vec = tri[1] - tri[0]
                b_vec = tri[2] - tri[0]
                cross = np.cross(a_vec, b_vec)
                area = 0.5 * np.linalg.norm(cross)
                areas.append(area)

    total_area = float(np.sum(areas))
    area_variance = float(np.var(areas))
    return total_area, area_variance, areas
