"""
spatial_partition.py
====================
基于 Voronoi 图的空间区域分解模块。

核心算法源自 florida_voronoi (Project 725)，并改造用于
湍流燃烧模拟中的并行计算域分解和混合区域划分。

Voronoi 图定义：
    给定种子点集 P = {p₁, p₂, ..., pₙ}，Voronoi 单元 V_i 为：

        V_i = { x ∈ ℝ² | ||x - p_i|| ≤ ||x - p_j||, ∀ j ≠ i }

在燃烧模拟中，Voronoi 分解用于：
1. 并行计算域分解：每个处理器负责一个 Voronoi 单元；
2. 混合分数空间分区：将混合分数空间 [0,1] 划分为多个区域，
   每个区域由不同的化学反应机理主导；
3. 粒子跟踪：在拉格朗日框架下跟踪燃料/氧化剂微团。

Delaunay 三角剖分与 Voronoi 图的对偶关系：
    Delaunay 边的垂直平分线构成 Voronoi 边。

本模块实现二维 Voronoi 单元面积计算和负载均衡分析，
以及用于燃烧室的一维/二维空间分区。
"""

import numpy as np


def voronoi_area_2d(seeds, bbox=((-1, 1), (-1, 1)), resolution=500):
    """
    使用栅格法近似计算二维 Voronoi 单元面积。

    Parameters
    ----------
    seeds : ndarray, shape (N, 2)
        种子点坐标。
    bbox : tuple
        ((xmin, xmax), (ymin, ymax)) 边界框。
    resolution : int
        栅格分辨率。

    Returns
    -------
    areas : ndarray, shape (N,)
        每个 Voronoi 单元的近似面积。
    load_balance : float
        负载均衡指标（最大面积/最小面积）。
    """
    seeds = np.asarray(seeds)
    N = len(seeds)
    if N == 0:
        return np.array([]), 1.0

    xmin, xmax = bbox[0]
    ymin, ymax = bbox[1]

    x_grid = np.linspace(xmin, xmax, resolution)
    y_grid = np.linspace(ymin, ymax, resolution)
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    areas = np.zeros(N)

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            dists = np.sqrt((seeds[:, 0] - x) ** 2 + (seeds[:, 1] - y) ** 2)
            nearest = np.argmin(dists)
            areas[nearest] += dx * dy

    min_area = np.min(areas) if np.min(areas) > 0 else 1.0e-12
    max_area = np.max(areas) if np.max(areas) > 0 else 1.0
    load_balance = max_area / min_area

    return areas, load_balance


def domain_decomposition_1d(Z_nodes, n_subdomains):
    """
    将一维混合分数空间分解为多个子域。

    Parameters
    ----------
    Z_nodes : ndarray
        混合分数空间节点。
    n_subdomains : int
        子域数量。

    Returns
    -------
    subdomain_indices : list of tuples
        每个子域的起止索引。
    subdomain_bounds : list of tuples
        每个子域的 Z 范围。
    """
    n = len(Z_nodes)
    nodes_per_sub = max(1, n // n_subdomains)

    subdomain_indices = []
    subdomain_bounds = []

    for i in range(n_subdomains):
        start = i * nodes_per_sub
        end = min((i + 1) * nodes_per_sub + 2, n) if i < n_subdomains - 1 else n
        # 重叠区保证连续性
        if i > 0:
            start = max(0, start - 1)

        subdomain_indices.append((start, end))
        subdomain_bounds.append((Z_nodes[start], Z_nodes[end - 1]))

    return subdomain_indices, subdomain_bounds


def compute_partition_quality(areas):
    """
    计算分区质量指标。

    指标包括：
    1. 负载均衡度：max(areas) / mean(areas)
    2. 面积标准差系数：std(areas) / mean(areas)
    3. 通信开销估计：与相邻单元边界长度相关

    Parameters
    ----------
    areas : ndarray
        各分区面积。

    Returns
    -------
    quality : dict
        质量指标字典。
    """
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    if mean_area < 1.0e-12:
        mean_area = 1.0e-12

    quality = {
        'load_imbalance': np.max(areas) / mean_area,
        'coefficient_of_variation': std_area / mean_area,
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'mean_area': mean_area,
        'num_partitions': len(areas),
    }

    return quality
