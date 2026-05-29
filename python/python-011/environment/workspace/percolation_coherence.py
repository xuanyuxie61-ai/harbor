# -*- coding: utf-8 -*-
"""
percolation_coherence.py
------------------------
渗流理论与超导相干性分析模块。

对应种子项目：865_percolation_simulation
核心算法：
  - 二维格点渗流随机生成（Bernoulli 占据）
  - 栈式深度优先搜索 (DFS) 连通分量标记
  - 跨越团簇检测与团簇尺寸分布统计

物理背景：
  欠掺杂高温超导体中，超导相以“岛状”不均匀分布，
  只有当这些超导岛在渗流阈值以上连通形成 spanning cluster 时，
  系统才表现出宏观零电阻和迈斯纳效应。
  渗流阈值 p_c ≈ 0.5927（二维方格 site percolation）。

核心公式：
  - 占据概率 p：每个格点以概率 p 被超导序占据
  - 渗流阈值 p_c：出现跨越团簇的临界概率
  - 关联长度 ξ ∝ |p - p_c|^{-ν}，ν = 4/3（二维 Ising 普适类）
"""

import numpy as np


def components_2d(A):
    """
    对二维二值矩阵 A 做四连通分量标记。

    使用栈式 DFS（深度优先搜索），对应种子项目的核心算法。

    Parameters
    ----------
    A : ndarray, shape (m, n)
        二值矩阵，1 表示占据，0 表示空。

    Returns
    -------
    C : ndarray, shape (m, n)
        整数标签矩阵，0 表示空位，>=1 表示分量标签。
    component_num : int
        分量总数。
    component_sizes : ndarray
        各分量尺寸（格点数）。
    """
    A = np.asarray(A, dtype=int)
    m, n = A.shape
    C = np.zeros((m, n), dtype=int)
    label = 0
    sizes = []

    for i in range(m):
        for j in range(n):
            if A[i, j] == 1 and C[i, j] == 0:
                label += 1
                stack = [(i, j)]
                C[i, j] = label
                count = 0
                while stack:
                    ci, cj = stack.pop()
                    count += 1
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            if A[ni, nj] == 1 and C[ni, nj] == 0:
                                C[ni, nj] = label
                                stack.append((ni, nj))
                sizes.append(count)

    return C, label, np.array(sizes, dtype=int)


def detect_spanning_clusters(C, component_num):
    """
    检测是否有团簇跨越左右边界或上下边界。

    Returns
    -------
    is_spanning : bool
    span_labels : list of int
        跨越团簇的标签列表。
    """
    if component_num == 0:
        return False, []
    m, n = C.shape
    span_labels = set()
    # 左右跨越
    left_labels = set(C[:, 0]) - {0}
    right_labels = set(C[:, n - 1]) - {0}
    span_labels |= left_labels & right_labels
    # 上下跨越
    top_labels = set(C[0, :]) - {0}
    bottom_labels = set(C[m - 1, :]) - {0}
    span_labels |= top_labels & bottom_labels
    return len(span_labels) > 0, list(span_labels)


def percolation_simulation(m, n, p, seed=None):
    """
    二维格点渗流模拟。

    Parameters
    ----------
    m, n : int
        格点尺寸。
    p : float
        占据概率，必须在 [0,1] 内。
    seed : int, optional

    Returns
    -------
    results : dict
        包含 'occupation_matrix', 'labels', 'component_num',
        'mean_size', 'spanning', 'span_labels', 'largest_size'。
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p 必须在 [0,1] 内。")
    if seed is not None:
        np.random.seed(seed)
    u = (np.random.rand(m, n) < p).astype(int)
    C, comp_num, sizes = components_2d(u)
    spanning, span_labels = detect_spanning_clusters(C, comp_num)
    mean_size = np.mean(sizes) if sizes.size > 0 else 0.0
    largest = np.max(sizes) if sizes.size > 0 else 0
    return {
        'occupation_matrix': u,
        'labels': C,
        'component_num': comp_num,
        'mean_size': mean_size,
        'spanning': spanning,
        'span_labels': span_labels,
        'largest_size': largest
    }


def find_percolation_threshold(m=100, n=100, n_trials=20):
    """
    用二分搜索估计二维方格 site percolation 阈值 p_c。

    物理上 p_c ≈ 0.592746，用于验证数值精度。
    """
    p_low = 0.0
    p_high = 1.0
    for _ in range(n_trials):
        p_mid = (p_low + p_high) * 0.5
        spans = 0
        for t in range(n_trials):
            res = percolation_simulation(m, n, p_mid, seed=t * 1000 + _)
            if res['spanning']:
                spans += 1
        ratio = spans / n_trials
        if ratio > 0.5:
            p_high = p_mid
        else:
            p_low = p_mid
    return (p_low + p_high) * 0.5


def superconducting_percolation_analysis(order_parameter_field, threshold=0.1):
    """
    对实空间超导序参量场做渗流分析。

    将 |Δ(r)| > threshold 的格点视为超导占据，
    分析连通团簇尺寸、跨越行为及相干性。

    Parameters
    ----------
    order_parameter_field : ndarray, shape (m, n)
        复数或实数序参量场。
    threshold : float
        超导阈值（序参量模的临界值）。

    Returns
    -------
    analysis : dict
    """
    field = np.abs(np.asarray(order_parameter_field, dtype=complex))
    occupied = (field > threshold).astype(int)
    C, comp_num, sizes = components_2d(occupied)
    spanning, span_labels = detect_spanning_clusters(C, comp_num)
    filling = np.mean(occupied)
    return {
        'filling_fraction': filling,
        'component_num': comp_num,
        'mean_cluster_size': np.mean(sizes) if sizes.size > 0 else 0.0,
        'spanning': spanning,
        'spanning_labels': span_labels,
        'largest_cluster_size': np.max(sizes) if sizes.size > 0 else 0,
        'labels': C
    }
