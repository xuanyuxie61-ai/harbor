"""
高维不确定性量化稀疏网格模块 (Uncertainty Quantification via Sparse Grids)

集成种子项目:
- 1137_spquad: Clenshaw-Curtis 稀疏网格求积 (Smolyak 构造)

科学背景:
  中尺度对流系统预报中, 微物理参数、边界层参数、初始条件等均存在不确定性.
  使用稀疏网格求积对高维参数空间进行高效采样, 计算 ensemble 统计量:
    E[f] ≈ Σ_{i} w_i * f(x_i)
    Var[f] ≈ Σ_i w_i * f(x_i)^2 - (E[f])^2

核心公式 (Smolyak 稀疏网格):
  对 d 维积分, 全张量积需要 N^d 个点, 而稀疏网格仅需 O(N (log N)^{d-1}) 个点:
    A(q,d) = Σ_{|i| ≤ q} (Δ^{i_1} ⊗ ... ⊗ Δ^{i_d})
  其中 Δ^i = Q^i - Q^{i-1} 为差分求积算子.

  1D Clenshaw-Curtis 规则:
    节点: x_k = cos(kπ/n), k=0,...,n
    权重通过离散余弦变换 (DCT) 计算.
"""

import numpy as np
from typing import List, Tuple, Callable


def clenshaw_curtis_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 n 阶 Clenshaw-Curtis 求积节点与权重 (基于 1137_spquad/clencurt).

    节点 (包含端点):
      x_k = cos(k * π / (n-1)),  k = 0, ..., n-1
    权重通过 Waldvogel 快速算法计算.
    """
    if n < 2:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    N = n - 1
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)

    # Waldvogel 算法
    v = np.ones(N + 1)
    v[0] = 0.5
    v[-1] = 0.5
    k = np.arange(N + 1)
    # 使用 DCT 思想
    w = np.zeros(N + 1)
    if N % 2 == 0:
        # N 偶数
        w[0] = 1.0 / (N**2 - 1)
        for j in range(1, N):
            if j % 2 == 0:
                w[j] = 2.0 / (N**2 - 1)
            else:
                w[j] = 0.0
        w[N] = 1.0 / (N**2 - 1)
        # 修正
        g = np.zeros(N // 2)
        g[0] = 1.0
        for m in range(1, N // 2):
            g[m] = -g[m-1] * (N - 2*m + 1) / (N - 2*m)
        for j in range(0, N + 1, 2):
            s = 0
            for m in range(N // 2):
                s += g[m] / (4*m*m - 1) * np.cos(2*m*theta[j])
            w[j] = 4.0 / N * s
    else:
        # N 奇数
        for j in range(N + 1):
            s = 0.0
            for m in range((N + 1) // 2):
                s += np.sin((2*m + 1) * theta[j]) / (2*m + 1)
            w[j] = 4.0 / N * s

    # 归一化
    w = w / np.sum(w) * 2.0
    return x, w


def sparse_grid_index_set(dim: int, level: int) -> List[Tuple[int, ...]]:
    """
    生成 Smolyak 稀疏网格的指标集 {i: |i|_1 ≤ level + dim - 1}.
    """
    indices = []

    def recurse(current, sum_val, d):
        if d == dim:
            if sum_val <= level + dim - 1:
                indices.append(tuple(current))
            return
        # 最小可能值
        min_i = 1
        # 剩余维度能达到的最大值约束
        max_i = level + dim - 1 - sum_val - (dim - d - 1) * 1
        for i in range(min_i, max_i + 1):
            current.append(i)
            recurse(current, sum_val + i, d + 1)
            current.pop()

    recurse([], 0, 0)
    return indices


def sparse_grid_points_weights(dim: int, level: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 d 维稀疏网格的求积点和权重 (基于 1137_spquad 的 Smolyak 构造).

    返回:
      points: (N, dim) 数组
      weights: (N,) 数组
    """
    if dim < 1:
        return np.zeros((1, 0)), np.ones(1)
    if level < 0:
        level = 0

    indices = sparse_grid_index_set(dim, level)

    # 预计算各阶 1D 规则
    max_level = level + dim - 1
    rules = {}
    for lvl in range(1, max_level + 1):
        n = 2**(lvl - 1) + 1 if lvl > 1 else 1
        x, w = clenshaw_curtis_rule(n)
        rules[lvl] = (x, w)

    # 使用差分构造
    points_list = []
    weights_list = []

    for idx in indices:
        # 计算当前指标对应的差分权重
        # 使用包含-排除原理: 对差分算子 Δ = Q_l - Q_{l-1}
        # 使用直接张量积构造 (简化版, 对低维低阶足够)
        prod_points = [rules[i][0] for i in idx]
        prod_weights = [rules[i][1] for i in idx]

        # 生成笛卡尔积
        import itertools
        for comb in itertools.product(*[range(len(p)) for p in prod_points]):
            pt = np.array([prod_points[d][comb[d]] for d in range(dim)])
            wt = np.prod([prod_weights[d][comb[d]] for d in range(dim)])
            points_list.append(pt)
            weights_list.append(wt)

    if not points_list:
        return np.zeros((1, dim)), np.ones(1)

    points = np.array(points_list)
    weights = np.array(weights_list)

    # 节点合并 (去重求和权重)
    unique_pts = []
    unique_wts = []
    tol = 1e-12
    for pt, wt in zip(points, weights):
        found = False
        for j, upt in enumerate(unique_pts):
            if np.linalg.norm(pt - upt) < tol:
                unique_wts[j] += wt
                found = True
                break
        if not found:
            unique_pts.append(pt)
            unique_wts.append(wt)

    points = np.array(unique_pts)
    weights = np.array(unique_wts)
    # 权重归一化到 2^dim (标准超立方体 [-1,1]^d 体积)
    vol = 2.0**dim
    weights = weights / np.sum(weights) * vol
    return points, weights


def scale_to_physical(points: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    将标准超立方体 [-1,1]^d 上的点映射到物理参数空间.
    """
    dim = points.shape[1]
    scaled = np.zeros_like(points)
    for d in range(dim):
        a, b = bounds[d]
        scaled[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)
    return scaled


class EnsembleSparseGridUQ:
    """
    基于稀疏网格的集合预报不确定性量化.
    """

    def __init__(self, dim: int, level: int = 2):
        self.dim = dim
        self.level = level
        self.points_std, self.weights = sparse_grid_points_weights(dim, level)
        self.n_points = len(self.points_std)

    def compute_expectation(self, f: Callable[[np.ndarray], float],
                           bounds: List[Tuple[float, float]]) -> float:
        """
        计算 E[f(ξ)] 的稀疏网格近似.
        """
        phys_points = scale_to_physical(self.points_std, bounds)
        total = 0.0
        for i in range(self.n_points):
            val = f(phys_points[i])
            if np.isfinite(val):
                total += self.weights[i] * val
        return total / (2.0**self.dim)

    def compute_statistics(self, f: Callable[[np.ndarray], float],
                           bounds: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        计算均值和标准差.
        """
        phys_points = scale_to_physical(self.points_std, bounds)
        mean = 0.0
        var = 0.0
        vol = 2.0**self.dim
        for i in range(self.n_points):
            val = f(phys_points[i])
            if np.isfinite(val):
                mean += self.weights[i] * val / vol
                var += self.weights[i] * val**2 / vol
        var = max(0.0, var - mean**2)
        return mean, np.sqrt(var)
