"""
sampling_methods.py
===================
多维采样方法集合：Latin 超立方采样、三角形网格、集合划分。

融合 latin_center、triangle_grid、set_theory 等项目的核心思想。

核心数学原理
------------
Latin Hypercube Sampling (LHS):
    将每个维度划分为 N 个等概率区间，在每个区间中随机
    抽取一个样本，并通过置换保证每行/列仅有一个样本。
    
    对于 d 维 N 点 LHS：
        x_{ij} = (π_j(i) + u_{ij}) / N
    
    其中 π_j 为第 j 维的随机置换，u_{ij} ~ U(0,1)。
    
    LHS 的方差缩减特性：
        Var(μ̂_LHS) ≈ Var(μ̂_MC) / N^{2/d}   （对单调函数）

三角形网格（Barycentric 坐标）：
    对于三角形 T(v₁,v₂,v₃)，内部点可表示为：
        p = (i·v₁ + j·v₂ + k·v₃) / n
    
    其中 i+j+k=n，i,j,k ≥ 0。网格点总数：
        N = (n+1)(n+2)/2

集合划分（等价类划分）：
    给定全集 U 和等价关系 ~，商集 U/~ 构成划分。
    在采样中用于将样本空间划分为不相交子区域。
"""

import numpy as np
from typing import Tuple, List


def latin_hypercube_sampling(dim: int, n_points: int,
                              domain: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    """
    生成 Latin Hypercube 采样点。
    
    参数:
        dim: 空间维度
        n_points: 采样点数
        domain: (low, high) 边界数组
    
    返回:
        n_points × dim 采样点矩阵
    """
    if domain is None:
        low = np.zeros(dim)
        high = np.ones(dim)
    else:
        low = np.asarray(domain[0])
        high = np.asarray(domain[1])

    samples = np.zeros((n_points, dim))
    for d in range(dim):
        perm = np.random.permutation(n_points)
        # 中心点策略：(perm + 0.5) / n_points
        samples[:, d] = (perm + 0.5) / n_points

    # 映射到实际区间
    return low + samples * (high - low)


def latin_center_sampling(dim: int, n_points: int,
                           domain: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    """
    Latin Center 采样：每个单元格中心点经过置换。
    
    源自 latin_center 项目，每个维度中每个区间的中心点
    (2k-1)/(2N) 经过随机置换。
    """
    if domain is None:
        low = np.zeros(dim)
        high = np.ones(dim)
    else:
        low = np.asarray(domain[0])
        high = np.asarray(domain[1])

    samples = np.zeros((n_points, dim))
    for d in range(dim):
        perm = np.random.permutation(n_points)
        samples[:, d] = (2.0 * perm + 1.0) / (2.0 * n_points)

    return low + samples * (high - low)


def triangle_grid_points(n: int, vertices: np.ndarray) -> np.ndarray:
    """
    生成三角形内部的规则网格点（重心坐标）。
    
    源自 triangle_grid 项目。
    
    参数:
        n: 每条边的分段数
        vertices: 3×2 或 3×3 顶点坐标
    
    返回:
        ((n+1)(n+2)/2) × dim 的点矩阵
    """
    vertices = np.asarray(vertices)
    dim = vertices.shape[1]
    n_points = (n + 1) * (n + 2) // 2
    points = np.zeros((n_points, dim))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            points[p] = (i * vertices[0] + j * vertices[1] + k * vertices[2]) / n
            p += 1
    return points


def triangle_grid_count(n: int) -> int:
    """计算 n 分段三角形网格的总点数。"""
    return (n + 1) * (n + 2) // 2


def set_partition_equivalence(n_elements: int,
                               relation_matrix: np.ndarray = None) -> List[List[int]]:
    """
    基于等价关系对集合进行划分。
    
    源自 set_theory 项目的集合划分思想。
    
    参数:
        n_elements: 元素个数（0, 1, ..., n-1）
        relation_matrix: n×n 对称矩阵，relation_matrix[i,j]=1 表示 i~j
    
    返回:
        等价类列表，每个等价类是一个索引列表
    """
    if relation_matrix is None:
        # 默认每个元素自成一类
        return [[i] for i in range(n_elements)]

    R = np.asarray(relation_matrix)
    visited = np.zeros(n_elements, dtype=bool)
    classes = []

    for i in range(n_elements):
        if visited[i]:
            continue
        # 找到所有与 i 等价的元素
        equiv_class = [i]
        visited[i] = True
        for j in range(i + 1, n_elements):
            if not visited[j] and R[i, j] > 0.5:
                equiv_class.append(j)
                visited[j] = True
        classes.append(equiv_class)

    return classes


def power_set_non_empty(n: int) -> List[List[int]]:
    """
    生成 n 元素集合的非空子集（幂集去除空集）。
    
    子集总数：2^n - 1
    """
    subsets = []
    for mask in range(1, 1 << n):
        subset = [i for i in range(n) if mask & (1 << i)]
        subsets.append(subset)
    return subsets


def stratified_sampling(n_strata: int, dim: int,
                        samples_per_stratum: int = 1) -> np.ndarray:
    """
    分层采样：将空间均匀划分为 n_strata^dim 个格子，
    每个格子内随机采样。
    
    总样本数：n_strata^dim × samples_per_stratum
    """
    total_samples = (n_strata ** dim) * samples_per_stratum
    samples = np.zeros((total_samples, dim))
    idx = 0
    # 遍历所有格子
    import itertools
    for cell in itertools.product(range(n_strata), repeat=dim):
        for _ in range(samples_per_stratum):
            s = np.random.rand(dim)
            samples[idx] = (np.array(cell) + s) / n_strata
            idx += 1
    return samples


def sobol_like_sampling(dim: int, n_points: int) -> np.ndarray:
    """
    简化的低差异序列采样（类似 Sobol 的准随机采样）。
    
    使用 Van der Corput / Halton 序列的思想：
        x_i = (i * φ) mod 1
    
    其中 φ 为黄金比例共轭（对于多维，使用不同基）。
    """
    samples = np.zeros((n_points, dim))
    for d in range(dim):
        base = d + 2  # 使用不同的基
        for i in range(n_points):
            # Halton 序列
            val = 0.0
            f = 1.0 / base
            n = i + 1
            while n > 0:
                val += f * (n % base)
                n //= base
                f /= base
            samples[i, d] = val
    return samples
