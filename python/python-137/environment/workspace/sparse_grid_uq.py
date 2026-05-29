# -*- coding: utf-8 -*-
"""
sparse_grid_uq.py

博士级稀疏网格不确定性量化库

融合原项目算法：
- 1103_sparse_grid_cc 的 Smolyak 稀疏网格 + Clenshaw-Curtis 求积

科学应用场景：
结晶动力学参数（k_g, E_g, k_b, 等）通常存在实验不确定性。
高维不确定性量化 (UQ) 需要计算模型输出关于参数随机变量的期望：

    E[f(ξ)] = ∫_{[-1,1]^d} f(ξ) w(ξ) dξ

其中 ξ ∈ [-1,1]^d 为标准化参数空间。

稀疏网格通过 Smolyak 构造避免维数灾难：
    Q_L^{(d)} f = Σ_{|ℓ|_1 ≤ L+d-1} c(ℓ) · (Q_{ℓ_1} ⊗ ... ⊗ Q_{ℓ_d}) f
    c(ℓ) = (-1)^{L+d-1-|ℓ|_1} · C(d-1, L+d-1-|ℓ|_1)

1D Clenshaw-Curtis 规则：
    节点：x_i = cos((i-1)·π/(n-1)), i=1,...,n
    权重：w_i = c_i/(n-1) · [1 - Σ_j b_j·cos(2j·θ_i)/(4j²-1)]
    其中 c_1 = c_n = 1/2, c_i = 1 (1<i<n)
"""

import numpy as np
from itertools import combinations_with_replacement


def clenshaw_curtis_abscissa(order, i):
    """
    1D Clenshaw-Curtis 节点。

    参数：
        order : int
            规则阶数（节点数）
        i : int
            节点索引 (1-based)

    返回：
        x : float in [-1, 1]
    """
    if order == 1:
        return 0.0
    if i < 1 or i > order:
        raise ValueError("i out of range")
    return np.cos((order - i) * np.pi / (order - 1))


def clenshaw_curtis_weights(n):
    """
    1D Clenshaw-Curtis 权重。

    参数：
        n : int
            节点数

    返回：
        w : ndarray, shape (n,)
    """
    if n == 1:
        return np.array([2.0])

    w = np.zeros(n, dtype=float)
    theta = np.array([(n - i) * np.pi / (n - 1) for i in range(1, n + 1)], dtype=float)

    for i in range(n):
        ti = theta[i]
        wi = 0.0
        # 余弦级数权重
        for j in range(1, (n - 1) // 2 + 1):
            b = 2.0 / (4.0 * j * j - 1.0)
            if 2 * j == n - 1:
                b /= 2.0
            wi -= b * np.cos(2.0 * j * ti)
        wi += 1.0
        if i == 0 or i == n - 1:
            wi /= 2.0
        wi *= 2.0 / (n - 1.0)
        w[i] = wi

    return w


def comp_next(n, k):
    """
    生成整数 n 的 k 部分组合（reverse lexicographic order）。

    参数：
        n : int
        k : int

    返回：
        compositions : list of tuples
    """
    if k == 1:
        return [(n,)]
    result = []
    def helper(remaining, parts, start):
        if len(parts) == k - 1:
            result.append(tuple(parts + [remaining]))
            return
        for i in range(remaining, -1, -1):
            helper(remaining - i, parts + [i], i)
    helper(n, [], n)
    return result


def sparse_grid_cc(dim_num, level_max):
    """
    构建 Smolyak 稀疏网格（Clenshaw-Curtis 规则）。

    参数：
        dim_num : int
            维数
        level_max : int
            最大层数

    返回：
        points : ndarray, shape (n_points, dim_num)
        weights : ndarray, shape (n_points,)
    """
    if dim_num <= 0 or level_max < 0:
        return np.zeros((0, max(1, dim_num))), np.zeros(0)

    # 1D 规则：层 ℓ 对应的阶数
    # ℓ=0: order=1, ℓ>=1: order=2^ℓ+1
    def level_to_order(level):
        if level == 0:
            return 1
        return 2 ** level + 1

    # 收集所有组合
    L = level_max
    all_points = []
    all_weights = []

    from math import comb
    for total_level in range(0, L + 1):
        for comp in comp_next(total_level, dim_num):
            level_sum = sum(comp)

            # Smolyak 系数
            coeff = (-1) ** (L - level_sum)
            coeff *= comb(dim_num - 1, L - level_sum)

            # 构造张量积网格
            orders = [level_to_order(l) for l in comp]
            # 1D 节点和权重
            grids_1d = []
            weights_1d = []
            for order in orders:
                x = np.array([clenshaw_curtis_abscissa(order, i + 1) for i in range(order)])
                w = clenshaw_curtis_weights(order)
                grids_1d.append(x)
                weights_1d.append(w)

            # 张量积
            from itertools import product
            for idx_tuple in product(*[range(len(g)) for g in grids_1d]):
                pt = np.array([grids_1d[d][idx_tuple[d]] for d in range(dim_num)])
                wt = coeff
                for d in range(dim_num):
                    wt *= weights_1d[d][idx_tuple[d]]
                all_points.append(pt)
                all_weights.append(wt)

    if not all_points:
        return np.zeros((0, dim_num)), np.zeros(0)

    points = np.array(all_points)
    weights = np.array(all_weights)

    # 合并重复点
    # 使用四舍五入避免浮点误差
    tol = 1e-12
    unique_indices = []
    unique_points = []
    unique_weights = []

    for i in range(len(points)):
        found = False
        for j, up in enumerate(unique_points):
            if np.all(np.abs(points[i] - up) < tol):
                unique_weights[j] += weights[i]
                found = True
                break
        if not found:
            unique_points.append(points[i].copy())
            unique_weights.append(weights[i])

    points = np.array(unique_points)
    weights = np.array(unique_weights)

    return points, weights


def sparse_grid_integrate(func, dim_num, level_max):
    """
    使用稀疏网格计算多维积分。

    参数：
        func : callable
            接受 shape (n_points, dim_num) 的数组
        dim_num : int
        level_max : int

    返回：
        integral : float
    """
    points, weights = sparse_grid_cc(dim_num, level_max)
    if points.size == 0:
        return 0.0
    f_vals = func(points)
    f_vals = np.asarray(f_vals, dtype=float)
    return float(np.dot(weights, f_vals))


def uncertainty_quantification_crystallization(model_func, param_distributions,
                                                level_max=3):
    """
    对结晶模型进行不确定性量化。

    参数：
        model_func : callable
            model_func(params) -> output，params 为 ndarray shape (dim,)
        param_distributions : list of dict
            每个字典包含 'mean' 和 'std'，假设参数服从 [-1,1] 上的均匀分布
            （通过 Legendre 变换映射）
        level_max : int

    返回：
        mean : float
        variance : float
        std : float
    """
    dim_num = len(param_distributions)
    points, weights = sparse_grid_cc(dim_num, level_max)

    if points.size == 0:
        return 0.0, 0.0, 0.0

    # 将 [-1,1] 映射到实际参数范围
    means = np.array([d['mean'] for d in param_distributions])
    stds = np.array([d['std'] for d in param_distributions])
    # 对于均匀分布 U[a,b]，在 [-1,1] 上映射为 x' = a + (b-a)/2 * (x+1)
    # 这里使用均值±3σ 作为范围
    lows = means - 3.0 * stds
    highs = means + 3.0 * stds

    actual_points = lows + 0.5 * (highs - lows) * (points + 1.0)

    outputs = np.array([model_func(p) for p in actual_points])
    mean_val = float(np.dot(weights, outputs))
    mean_sq = float(np.dot(weights, outputs ** 2))
    variance = max(mean_sq - mean_val ** 2, 0.0)
    std_val = np.sqrt(variance)

    return mean_val, variance, std_val
