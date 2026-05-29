# -*- coding: utf-8 -*-
"""
csd_analysis.py

博士级晶体尺寸分布 (CSD) 分析库

融合原项目算法：
- 583_image_quantization 的 K-Means 聚类算法
- 448_fresnel 的衍射理论用于粒度分析

科学应用场景：
1. K-Means 聚类：将连续的 CSD 离散化为有限个尺寸类别，
   用于简化的人口平衡模型（分区法）。
2. 衍射分析：基于 Fraunhofer/Fresnel 衍射理论的激光粒度分析，
   将衍射图样反演为 CSD。
3. 分布拟合优度检验：χ² 检验评估实验数据与模型预测的吻合度。

关键公式：
- 分区法离散化：将尺寸域 [L_min, L_max] 分为 K 个区间，
  每个区间的代表尺寸为聚类中心
- 激光衍射反演：I(θ) = ∫_0^∞ n(L)·I_L(θ,L) dL
  其中 I_L(θ,L) 为单个颗粒的衍射图样
"""

import numpy as np
from special_functions import fraunhofer_diffraction_particle_size


def kmeans_1d(data, k, max_iter=100, tol=1e-6, rng=None):
    """
    一维 K-Means 聚类（用于 CSD 尺寸类离散化）。

    算法：
        1. 初始化：随机选择 k 个数据点作为初始中心
        2. 分配：每个数据点分配到最近的中心
        3. 更新：中心 = 该类所有点的均值
        4. 重复 2-3 直到收敛

    参数：
        data : ndarray, shape (n,)
        k : int
            聚类数（尺寸类数）
        max_iter : int
        tol : float
        rng : numpy.random.Generator

    返回：
        centers : ndarray, shape (k,)
            聚类中心（代表尺寸）
        labels : ndarray, shape (n,)
            每个点的类别标签
        inertia : float
            簇内平方和
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0 or k <= 0:
        return np.array([]), np.array([], dtype=int), 0.0
    k = min(k, n)

    # 初始化：随机选择 k 个不重复的数据点
    indices = rng.choice(n, size=k, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):
        # 分配
        distances = np.abs(data[:, np.newaxis] - centers)
        labels = np.argmin(distances, axis=1)

        # 更新
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = np.mean(data[mask])
            else:
                # 空簇处理：重新随机初始化
                new_centers[i] = data[rng.integers(n)]

        # 收敛判断
        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break

    # 计算惯性
    inertia = 0.0
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            inertia += np.sum((data[mask] - centers[i]) ** 2)

    return centers, labels, inertia


def discretize_csd_kmeans(L_grid, f_values, k_classes):
    """
    使用 K-Means 将连续 CSD 离散化为 K 个尺寸类。

    方法：
        以 f(L) 作为权重，在 L 上进行加权 K-Means 聚类。
        每个类的粒子数 = ∫_{class} f(L) dL
        代表尺寸 = 类内加权平均

    参数：
        L_grid : ndarray
            尺寸网格 (m)
        f_values : ndarray
            尺寸分布密度 (#/(m³·m))
        k_classes : int
            尺寸类数

    返回：
        class_sizes : ndarray
            各类代表尺寸 (m)
        class_counts : ndarray
            各类粒子数密度 (#/m³)
        boundaries : ndarray
            类边界 (m)
    """
    L_grid = np.asarray(L_grid, dtype=float)
    f_values = np.asarray(f_values, dtype=float)

    # 归一化权重
    weights = np.maximum(f_values, 0.0)
    total_weight = np.trapezoid(weights, L_grid)
    if total_weight <= 0:
        return np.zeros(k_classes), np.zeros(k_classes), np.zeros(k_classes + 1)

    # 使用加权采样生成代表性点集
    rng = np.random.default_rng(42)
    n_samples = 10000
    # 基于权重采样
    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]
    u = rng.random(n_samples)
    sampled_indices = np.searchsorted(cum_weights, u)
    sampled_L = L_grid[sampled_indices]

    # K-Means 聚类
    centers, labels, _ = kmeans_1d(sampled_L, k_classes, rng=rng)
    centers = np.sort(centers)

    # 计算各类的粒子数和边界
    class_counts = np.zeros(k_classes)
    boundaries = np.zeros(k_classes + 1)
    boundaries[0] = L_grid[0]
    boundaries[-1] = L_grid[-1]

    # 边界为相邻中心的平均值
    for i in range(1, k_classes):
        boundaries[i] = 0.5 * (centers[i - 1] + centers[i])

    for i in range(k_classes):
        mask = (L_grid >= boundaries[i]) & (L_grid < boundaries[i + 1])
        if i == k_classes - 1:
            mask = (L_grid >= boundaries[i]) & (L_grid <= boundaries[i + 1])
        if np.any(mask):
            class_counts[i] = np.trapezoid(f_values[mask], L_grid[mask])

    return centers, class_counts, boundaries


def diffraction_inversion_feret(theta, intensity, wavelength,
                                 L_min=1e-6, L_max=1000e-6, n_bins=100):
    """
    基于 Fraunhofer 衍射的粒度分布反演。

    数学模型：
        I(θ) = ∫_{L_min}^{L_max} n(L) · I_1(θ, L) dL
        其中 I_1(θ, L) = [2·J_1(k·L/2·sinθ) / (k·L/2·sinθ)]²

        离散化为线性方程组：
        I_i = Σ_j A_{ij} · n_j · ΔL_j
        其中 A_{ij} = I_1(θ_i, L_j)

        使用非负最小二乘法 (NNLS) 求解 n_j。

    参数：
        theta : ndarray
            散射角 (rad)
        intensity : ndarray
            衍射光强
        wavelength : float
            激光波长 (m)
        L_min, L_max : float
            尺寸范围 (m)
        n_bins : int
            尺寸分档数

    返回：
        L_bins : ndarray
        n_L : ndarray
            粒度分布 (#/m³/m)
    """
    from scipy.optimize import nnls

    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    L_bins = np.linspace(L_min, L_max, n_bins)
    dL = L_bins[1] - L_bins[0]

    # 构造矩阵 A
    A = np.zeros((len(theta), n_bins), dtype=float)
    for j, L in enumerate(L_bins):
        A[:, j] = fraunhofer_diffraction_particle_size(L / 2.0, wavelength, theta)

    # NNLS 求解
    n_L, residual = nnls(A, intensity)
    n_L = n_L / dL  # 转换为密度

    return L_bins, n_L


def csd_statistical_moments(L_grid, f_values):
    """
    计算 CSD 的统计矩量。

    参数：
        L_grid : ndarray
        f_values : ndarray

    返回：
        moments : dict
            {'mean': ..., 'std': ..., 'skewness': ..., 'kurtosis': ...}
    """
    L_grid = np.asarray(L_grid, dtype=float)
    f_values = np.asarray(f_values, dtype=float)

    # 归一化
    total = np.trapezoid(f_values, L_grid)
    if total <= 0:
        return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    pdf = f_values / total

    mean_L = np.trapezoid(L_grid * pdf, L_grid)
    var_L = np.trapezoid((L_grid - mean_L) ** 2 * pdf, L_grid)
    std_L = np.sqrt(max(var_L, 0.0))

    if std_L < 1e-30:
        return {'mean': mean_L, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    skewness = np.trapezoid(((L_grid - mean_L) / std_L) ** 3 * pdf, L_grid)
    kurtosis = np.trapezoid(((L_grid - mean_L) / std_L) ** 4 * pdf, L_grid)

    return {
        'mean': mean_L,
        'std': std_L,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
