"""
sparse_integrator.py
稀疏网格高维数值积分器

凝聚态物理应用：
在固体物理中，许多物理量需要在高维布里渊区上积分：
    I = \int_{BZ} f(k) d^3k / V_BZ

对于高维问题（如多体系统），规则网格的采样点数随维度指数增长（"维度灾难"）。
稀疏网格（Sparse Grid）通过Smolyak构造，在保持精度的同时大幅减少采样点数。

核心公式 - Smolyak稀疏网格：
    A(q,d) = sum_{q-d+1 <= |l|_1 <= q} (-1)^{q-|l|} * C(d-1, q-|l|) * (U^{l1} x ... x U^{ld})

其中U^l是一维Clenshaw-Curtis求积规则，使用Chebyshev-Gauss-Lobatto节点：
    x_j^l = cos(pi * j / 2^l),  j = 0, ..., 2^l

一维Clenshaw-Curtis权重（Waldvogel算法）：
    w_j = c_j / (N-1) * [1 - sum_{p=1}^{floor((N-1)/2)} b_p/(4p^2-1) * cos(2*pi*p*j/(N-1))]
    其中c_0 = c_{N-1} = 1/2, c_j = 1 否则
    b_p = 1/(4p^2-1) for p < (N-1)/2, b_p = 0 otherwise

基于种子项目1137_spquad（Greg von Winckel实现）。
"""

import numpy as np
from typing import Tuple, Callable
from itertools import product


def clenshaw_curtis_weights(N: int) -> np.ndarray:
    """
    计算一维Clenshaw-Curtis求积权重
    
    参考：Jörg Waldvogel, BIT 43 (2003)
    
    Parameters
    ----------
    N : int
        节点数
    
    Returns
    -------
    w : np.ndarray, shape (N,)
    """
    if N == 1:
        return np.array([2.0])
    
    n = N - 1
    # 构建向量c
    c = np.zeros(N)
    c[0::2] = 2.0 / np.arange(1, n + 2, 2)
    c[0] = 1.0
    if n % 2 == 0:
        c[-1] = 1.0 / (n + 1)
    else:
        c[-1] = 0.0
    
    # 使用FFT计算权重
    c_extended = np.concatenate([c, c[-2:0:-1]])
    f = np.fft.ifft(c_extended).real
    w = 2 * f[:N]
    w[0] *= 0.5
    w[-1] *= 0.5
    
    return w


def clenshaw_curtis_nodes(N: int) -> np.ndarray:
    """
    计算一维Clenshaw-Curtis节点（Chebyshev-Gauss-Lobatto点）
    
    x_j = cos(pi * j / (N-1)), j = 0, ..., N-1
    
    Parameters
    ----------
    N : int
    
    Returns
    -------
    x : np.ndarray, shape (N,)
    """
    if N == 1:
        return np.array([0.0])
    
    j = np.arange(N)
    x = np.cos(np.pi * j / (N - 1))
    return x


def difference_weights(level: int) -> np.ndarray:
    """
    计算差分权重（用于稀疏网格的层级构造）
    
    基于种子项目1137_spquad中的diffweight函数。
    
    dw_l = w_l - w_{l-1}（延拓到相同节点）
    
    Parameters
    ----------
    level : int
        层级（level=0对应中点规则）
    
    Returns
    -------
    dw : np.ndarray
    """
    if level == 0:
        return np.array([2.0])
    elif level == 1:
        # 3点CC规则减去中点规则
        w1 = clenshaw_curtis_weights(3)
        dw = np.zeros(3)
        dw[0] = w1[0]
        dw[1] = w1[1] - 2.0  # 中点规则权重为2
        dw[2] = w1[2]
        return dw
    else:
        N_prev = 2 ** (level - 1) + 1
        N_curr = 2 ** level + 1
        
        w_prev = clenshaw_curtis_weights(N_prev)
        w_curr = clenshaw_curtis_weights(N_curr)
        
        # 延拓w_prev到N_curr个节点（奇数位置）
        dw = w_curr.copy()
        for i in range(N_prev):
            dw[2 * i] -= w_prev[i]
        
        return dw


def generate_index_set(dim: int, max_level: int) -> np.ndarray:
    """
    生成稀疏网格的索引集合
    
    寻找所有满足 |l|_1 <= max_level 的多重指标 l = (l1, ..., ld)
    且每个 li >= 0。
    
    基于种子项目1137_spquad中的genindex函数。
    
    Parameters
    ----------
    dim : int
        维度
    max_level : int
        最大层级
    
    Returns
    -------
    indices : np.ndarray, shape (N, dim)
    """
    indices = []
    
    def recursive_gen(remaining_dim: int, remaining_level: int, current: list):
        if remaining_dim == 1:
            for l in range(remaining_level + 1):
                indices.append(current + [l])
        else:
            for l in range(remaining_level + 1):
                recursive_gen(remaining_dim - 1, remaining_level - l, current + [l])
    
    # Smolyak构造：|l|_1 <= max_level
    recursive_gen(dim, max_level, [])
    
    # 过滤掉 |l|_1 < max_level - dim + 1 的项
    result = []
    for idx in indices:
        if sum(idx) >= max_level - dim + 1:
            result.append(idx)
    
    return np.array(result, dtype=int)


def sparse_grid_quadrature(dim: int, max_level: int,
                            bounds: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建d维稀疏网格求积规则
    
    基于种子项目1137_spquad中的spquad和sparsegridnd函数。
    
    Parameters
    ----------
    dim : int
        维度
    max_level : int
        稀疏网格层级
    bounds : np.ndarray, shape (dim, 2)
        积分区域边界，默认[-1, 1]^dim
    
    Returns
    -------
    points : np.ndarray, shape (N, dim)
        求积节点
    weights : np.ndarray, shape (N,)
        求积权重
    """
    if bounds is None:
        bounds = np.array([[-1.0, 1.0]] * dim)
    
    if bounds.shape != (dim, 2):
        raise ValueError("bounds必须是(dim, 2)数组")
    
    # 一维变换参数
    lengths = bounds[:, 1] - bounds[:, 0]
    midpoints = 0.5 * (bounds[:, 0] + bounds[:, 1])
    
    index_set = generate_index_set(dim, max_level)
    
    # 收集所有子网格点
    all_points = []
    all_weights = []
    
    for idx in index_set:
        # 计算组合系数
        level_sum = np.sum(idx)
        coeff = ((-1) ** (max_level - level_sum)) * comb(max_level - level_sum, dim - 1)
        
        # 构建该子网格的笛卡尔积
        sub_points_list = []
        sub_weights_list = []
        
        for d in range(dim):
            level = idx[d]
            if level == 0:
                nodes = np.array([0.0])
                dw = np.array([2.0])
            else:
                N = 2 ** level + 1
                nodes = clenshaw_curtis_nodes(N)
                dw = difference_weights(level)
            
            # 变换到实际区间
            nodes = midpoints[d] + 0.5 * lengths[d] * nodes
            dw = dw * (lengths[d] / 2.0)
            
            sub_points_list.append(nodes)
            sub_weights_list.append(dw)
        
        # 笛卡尔积
        for point_tuple in product(*sub_points_list):
            all_points.append(point_tuple)
        for weight_tuple in product(*sub_weights_list):
            all_weights.append(coeff * np.prod(weight_tuple))
    
    if len(all_points) == 0:
        return np.zeros((0, dim)), np.zeros(0)
    
    points = np.array(all_points)
    weights = np.array(all_weights)
    
    # 合并重复点（节点凝聚）
    points_rounded = np.round(points, decimals=14)
    unique_points = []
    unique_weights = []
    
    visited = set()
    for i in range(len(points_rounded)):
        key = tuple(points_rounded[i])
        if key not in visited:
            visited.add(key)
            unique_points.append(points_rounded[i])
            # 合并所有相同点的权重
            w_sum = 0.0
            for j in range(len(points_rounded)):
                if tuple(points_rounded[j]) == key:
                    w_sum += weights[j]
            unique_weights.append(w_sum)
    
    points = np.array(unique_points)
    weights = np.array(unique_weights)
    
    return points, weights


def comb(n: int, k: int) -> int:
    """
    计算组合数 C(n, k)
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def integrate_sparse_grid(dim: int, max_level: int,
                           integrand: Callable[[np.ndarray], np.ndarray],
                           bounds: np.ndarray = None) -> float:
    """
    使用稀疏网格对高维函数进行数值积分
    
    I ≈ sum_i w_i * f(x_i)
    
    Parameters
    ----------
    dim : int
    max_level : int
    integrand : callable
        被积函数，输入shape (N, dim)，输出shape (N,)
    bounds : np.ndarray
    
    Returns
    -------
    result : float
    """
    points, weights = sparse_grid_quadrature(dim, max_level, bounds)
    
    if len(points) == 0:
        return 0.0
    
    values = integrand(points)
    result = np.dot(weights, values)
    return result
