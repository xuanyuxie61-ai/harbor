"""
tsp_confidence_path.py — TSP 下降法优化采样路径

科学背景
========
在自适应不确定性量化中, 需要在参数空间中选择最优采样路径,
使得总"距离" (信息增益的代价) 最小. 这可以建模为 TSP:

    min_π Σ_{i} d(π(i), π(i+1))
    s.t. π 为 {1,...,N} 的排列

其中 d(i,j) 为采样点 i, j 之间的"代价距离".

算法来源 (种子项目 1364_tsp_descent)
====================================
直接移植下降法:
1. 随机初始排列
2. 迭代尝试 transposition 和 reversal
3. 接受代价降低的变体

在本项目中的角色
================
优化 MC 采样的空间填充路径, 使得连续样本之间的相关性最小.
这提高了 MC 估计的效率 ( decorrelated samples).

核心公式
========
1. 代价函数:  C(π) = Σ_{i=1}^{N} d(π(i), π(i+1))
2. Transposition:  交换 π 中两个非相邻元素
3. Reversal:  反转 π 中一段子序列
4. 接受准则:  C(π') < C(π)
"""

import numpy as np


def path_cost(n, distance, p):
    """计算路径总代价 (种子 1364).

    C(p) = Σ_{i=1}^{n} d(p[i], p[(i+1) mod n])

    参数
    ----
    n : int
    distance : ndarray, shape (n, n)
    p : ndarray of int, shape (n,)

    返回
    ----
    cost : float
    """
    cost = 0.0
    for i in range(n):
        j = (i + 1) % n
        cost += distance[p[i], p[j]]
    return cost


def tsp_descent(distance, variation_num=500, seed=42):
    """TSP 下降法 (种子 1364).

    参数
    ----
    distance : ndarray, shape (n, n)
    variation_num : int
    seed : int

    返回
    ----
    best_path : ndarray of int
    best_cost : float
    n_transpose : int
    n_reversal : int
    """
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    if n < 3:
        return np.arange(n), 0.0, 0, 0

    p = rng.permutation(n)
    cost = path_cost(n, distance, p)

    n_transpose = 0
    n_reversal = 0

    for _ in range(variation_num):
        # Variation 1: transposition
        # 选择两个非相邻位置 i1 < i2 (i2 > i1+1)
        while True:
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 > i2:
                i1, i2 = i2, i1
            if i2 > i1 + 1:
                break

        n_transpose += 1
        # 将位置 i2 的元素插入到位置 i1 之后
        p2 = p.copy()
        elem = p2[i2]
        # 删除 i2 位置, 插入到 i1+1
        p2 = np.delete(p2, i2)
        p2 = np.insert(p2, i1 + 1, elem)
        cost2 = path_cost(n, distance, p2)
        if cost2 < cost:
            p = p2
            cost = cost2

        # Variation 2: reversal
        i1 = rng.integers(0, n)
        i2 = rng.integers(0, n)
        if i1 > i2:
            i1, i2 = i2, i1

        n_reversal += 1
        p2 = p.copy()
        if i2 > i1:
            p2[i1:i2+1] = p[i1:i2+1][::-1]
        cost2 = path_cost(n, distance, p2)
        if cost2 < cost:
            p = p2
            cost = cost2

    return p, cost, n_transpose, n_reversal


def build_sampling_distance_matrix(sample_points, metric='correlation'):
    """构造采样点之间的距离矩阵.

    参数
    ----
    sample_points : ndarray, shape (n_samples, n_features)
    metric : str
        'euclidean' 或 'correlation'

    返回
    ----
    distance : ndarray, shape (n_samples, n_samples)
    """
    n = sample_points.shape[0]
    distance = np.zeros((n, n))

    if metric == 'euclidean':
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((sample_points[i] - sample_points[j]) ** 2))
                distance[i, j] = d
                distance[j, i] = d
    elif metric == 'correlation':
        # 基于相关性的距离: d = 1 - |corr|
        corr = np.corrcoef(sample_points)
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - abs(corr[i, j])
                distance[i, j] = d
                distance[j, i] = d

    return distance


def optimize_mc_sampling_order(mc_solutions, metric='correlation'):
    """优化 MC 样本的处理顺序.

    参数
    ----
    mc_solutions : ndarray, shape (n_mc, n_points)
        MC 解的集合

    返回
    ----
    optimized_order : ndarray of int
    original_cost : float
    optimized_cost : float
    """
    n_mc = mc_solutions.shape[0]
    if n_mc < 4:
        return np.arange(n_mc), 0.0, 0.0

    distance = build_sampling_distance_matrix(mc_solutions, metric)
    original_order = np.arange(n_mc)
    original_cost = path_cost(n_mc, distance, original_order)

    optimized_order, optimized_cost, _, _ = tsp_descent(distance, variation_num=300)

    return optimized_order, original_cost, optimized_cost
