"""
monte_carlo_integrator.py
高维积分与球面积分模块

基于种子项目的核心算法：
- 1124_sphere_monte_carlo: 球面采样、球面积分、单项式积分
- 654_lattice_rule: Fibonacci 格点积分规则
- 1103_sparse_grid_cc: Clenshaw-Curtis 稀疏网格求积

在离子通道问题中的应用：
- 球面蒙特卡洛用于计算离子溶剂化壳层的球面平均电势
- Fibonacci 格点用于周期性边界条件下的高维相空间积分
- 稀疏网格用于自由能面的高维积分（多反应坐标）
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# 球面蒙特卡洛（源自 sphere01_sample.m / sphere01_monomial_integral.m）
# ---------------------------------------------------------------------------
def sphere01_sample(n):
    """
    在单位球面上均匀采样 n 个点（Gaussian 投影法）。

    算法：
        1. 生成 3D 标准正态随机向量 v ~ N(0, I)
        2. 归一化 x = v / ||v||
    """
    x = np.random.randn(3, n)
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    x = x / norm
    return x


def sphere01_monomial_integral(e):
    """
    计算单位球面上单项式 x^e1 y^e2 z^e3 的精确积分。

    解析公式：
        若任一 e_i 为奇数，积分 = 0
        否则：
            I = 2 * Π_i Γ((e_i + 1)/2) / Γ( (Σ_i (e_i + 1)) / 2 )
    """
    e = np.array(e)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    if np.all(e == 0):
        return 2.0 * np.sqrt(np.pi ** 3) / np.math.gamma(1.5)
    if np.any(e % 2 == 1):
        return 0.0
    integral = 2.0
    for ei in e:
        integral *= math.gamma(0.5 * (ei + 1))
    integral /= math.gamma(0.5 * np.sum(e + 1))
    return integral


def spherical_mean_integrand(func, n_samples=10000):
    """
    对给定函数 func(x,y,z) 在单位球面上进行蒙特卡洛平均。
    """
    pts = sphere01_sample(n_samples)
    vals = np.array([func(pts[0, i], pts[1, i], pts[2, i]) for i in range(n_samples)])
    return np.mean(vals), np.std(vals) / np.sqrt(n_samples)


# ---------------------------------------------------------------------------
# Fibonacci 格点积分（源自 fibonacci_lattice_b.m / lattice.m）
# ---------------------------------------------------------------------------
def fibonacci(n):
    """
    计算第 n 个 Fibonacci 数。
    """
    if n <= 0:
        return 0
    if n == 1 or n == 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def fibonacci_lattice_2d(k, func):
    """
    二维 Fibonacci 格点积分规则（源自 fibonacci_lattice_b.m 简化版）。

    格点构造：
        x_j = (j * F_{k-1} / F_k) mod 1
        y_j = j / F_k

    用于周期性边界条件下的平面积分。
    """
    if k < 3:
        raise ValueError("k 必须 >= 3")
    m = fibonacci(k)
    n = fibonacci(k - 1)

    quad = 0.0
    for j in range(m):
        x = (j * n % m) / m
        y = j / m
        quad += func(np.array([x, y]))
    quad /= m
    return quad


def lattice_rule_nd(dim_num, m, z, func):
    """
    多维标准格点积分规则（源自 lattice.m）。

    格点：
        x_j = mod( j * z / m, 1 )

    Parameters
    ----------
    dim_num : int
        空间维度
    m : int
        格点数量
    z : ndarray
        生成向量，1 <= z_i < m
    func : callable
        被积函数 func(x) -> float
    """
    quad = 0.0
    for j in range(m):
        x = (j * z) % m / m
        quad += func(x)
    quad /= m
    return quad


# ---------------------------------------------------------------------------
# Clenshaw-Curtis 稀疏网格（源自 sparse_grid_cc.m 等）
# ---------------------------------------------------------------------------
def cc_abscissa(order, idx):
    """
    Clenshaw-Curtis 一维节点（闭型，包含端点）。

    节点公式（n 阶，i = 0,...,n-1）：
        x_i = cos( π * (n - 1 - i) / (n - 1) )
    """
    if order == 1:
        return 0.0
    if idx < 0 or idx >= order:
        raise IndexError("CC 节点索引越界")
    return np.cos(np.pi * (order - 1 - idx) / (order - 1))


def cc_weights(order):
    """
    Clenshaw-Curtis 一维权值（简化版，基于 FFT 的高效算法）。

    对于闭型 CC 规则，节点 x_j = cos(jπ/n)，j=0..n，权值：
        w_j = c_j / (n * d_j)
    其中 c_j, d_j 为余弦级数系数。
    """
    if order == 1:
        return np.array([2.0])
    n = order - 1
    theta = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = 1.0 / (n ** 2 - 1)
        w[n] = w[0]
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
        v = v - np.cos(n * theta[1:-1]) / (n ** 2 - 1)
    else:
        w[0] = 1.0 / n ** 2
        w[n] = w[0]
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2.0 * v / n
    return w


def sparse_grid_cc_1d(level):
    """
    一维 Clenshaw-Curtis 稀疏网格节点与权值。

    层数 L 对应的阶数：
        n(L) = 2^L + 1  (L >= 1)
        n(0) = 1
    """
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    order = 2 ** level + 1
    nodes = np.array([cc_abscissa(order, i) for i in range(order)])
    weights = cc_weights(order)
    return nodes, weights


def tensor_product_grid(nodes_list, weights_list):
    """
    构造多维张量积网格（用于稀疏网格的 Smolyak 构造）。
    """
    dims = len(nodes_list)
    if dims == 1:
        return nodes_list[0].reshape(-1, 1), weights_list[0]

    # 递归构造
    nodes_prev, weights_prev = tensor_product_grid(nodes_list[:-1], weights_list[:-1])
    nodes_last = nodes_list[-1]
    weights_last = weights_list[-1]

    n_prev = nodes_prev.shape[0]
    n_last = len(nodes_last)
    nodes = np.zeros((n_prev * n_last, dims))
    weights = np.zeros(n_prev * n_last)
    idx = 0
    for i in range(n_prev):
        for j in range(n_last):
            nodes[idx, :-1] = nodes_prev[i]
            nodes[idx, -1] = nodes_last[j]
            weights[idx] = weights_prev[i] * weights_last[j]
            idx += 1
    return nodes, weights


def sparse_grid_cc_smolyak(dim_num, level_max):
    """
    Smolyak 稀疏网格构造（基于 Clenshaw-Curtis 规则）。

    索引集合：
        |ℓ|_1 = ℓ_1 + ... + ℓ_d <= L + d - 1
        其中 ℓ_i >= 1（或 0）

    权值组合（增量公式）：
        Q^{(d)}_L = Σ_{L <= |ℓ|_1 <= L+d-1} (-1)^{L+d-1-|ℓ|_1} * C(d-1, |ℓ|_1-L) * (Q^{(1)}_{ℓ_1} ⊗ ... ⊗ Q^{(1)}_{ℓ_d})
    """
    from combinatorial_stats import combination_lex_index

    # 生成所有满足 |ℓ|_1 <= L + d - 1 的多索引
    grids = []
    for total in range(level_max, level_max + dim_num):
        # 使用简单递归生成组合
        def generate(dim, remain, current):
            if dim == 1:
                yield current + [remain]
            else:
                for v in range(0, remain + 1):
                    yield from generate(dim - 1, remain - v, current + [v])

        for levels in generate(dim_num, total, []):
            # 确保每个 level >= 0
            if all(l >= 0 for l in levels):
                # 计算增量权值系数
                coeff = ((-1) ** (level_max + dim_num - 1 - total)) * math.comb(dim_num - 1, total - level_max)
                nodes_list = []
                weights_list = []
                for l in levels:
                    n, w = sparse_grid_cc_1d(l)
                    nodes_list.append(n)
                    weights_list.append(w)
                nodes, weights = tensor_product_grid(nodes_list, weights_list)
                grids.append((nodes, coeff * weights))

    # 合并重复节点（简化：直接拼接，对于小维度可接受）
    all_nodes = []
    all_weights = []
    for nodes, weights in grids:
        all_nodes.append(nodes)
        all_weights.append(weights)

    if len(all_nodes) == 0:
        return np.zeros((0, dim_num)), np.zeros(0)

    all_nodes = np.vstack(all_nodes)
    all_weights = np.concatenate(all_weights)

    # 去重（四舍五入到小数点后 12 位）
    rounded = np.round(all_nodes, 12)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    merged_weights = np.zeros(unique.shape[0])
    for i, w in enumerate(all_weights):
        merged_weights[inverse[i]] += w

    # 映射回 [-1, 1]^d（已在 [-1,1] 内）
    return unique, merged_weights


def integrate_sparse_grid(func, dim_num, level_max):
    """
    使用稀疏网格计算积分 ∫_{[-1,1]^d} f(x) dx。
    """
    nodes, weights = sparse_grid_cc_smolyak(dim_num, level_max)
    total = 0.0
    for i in range(nodes.shape[0]):
        total += weights[i] * func(nodes[i])
    return total
