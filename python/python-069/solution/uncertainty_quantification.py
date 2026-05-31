"""
不确定性量化模块：基于 sparse_grid_laguerre，
使用 Smolyak 稀疏网格对光合模型关键参数的不确定性进行传播分析。

核心数学：
  参数不确定性模型：假设 V_cmax 和 J_max 服从对数正态分布，
  等价于 x = ln(V_cmax) 服从指数型分布，可用 Laguerre 正交多项式展开。

  Smolyak 稀疏网格求积：
      Q_L^{(d)} f = sum_{L_min <= |l| <= L} (-1)^{L-|l|} * C(d-1, L-|l|) * (Q_{l1} x ... x Q_{ld}) f

  碳通量统计量：
      E[F] ≈ sum_i w_i * F(x_i)
      Var[F] ≈ sum_i w_i * F(x_i)^2 - E[F]^2
"""
import numpy as np
from scipy.special import comb


def laguerre_abscissa_and_weights(n):
    """
    Gauss-Laguerre 求积节点与权重。
    返回: x, w
    """
    x, w = np.polynomial.laguerre.laggauss(n)
    return x, w


def level_to_order_open(level_1d):
    """
    将 1D level 映射到求积阶数（open 规则）。
    level=0 -> order=1, level=1 -> order=3, level=2 -> order=7, ...
    """
    order = 2 ** (level_1d + 1) - 1
    return order


def _integer_compositions(n, k):
    """
    生成整数 n 的 k 部分非负整数组合。
    返回生成器，每次 yield 一个长度为 k 的 numpy 数组。
    """
    if k == 1:
        yield np.array([n], dtype=int)
        return
    # 使用递归生成组合
    for first in range(n + 1):
        for rest in _integer_compositions(n - first, k - 1):
            yield np.concatenate(([first], rest))


def sparse_grid_laguerre(dim_num, level_max):
    """
    构建基于 Laguerre 的稀疏网格（Smolyak 构造）。
    返回: points (dim_num, n_points), weights (n_points,)
    """
    level_min = max(0, level_max + 1 - dim_num)
    grid_points_list = []
    grid_weights_list = []

    for level in range(level_min, level_max + 1):
        coeff = ((-1) ** (level_max - level)) * int(comb(dim_num - 1, level_max - level))

        for level_1d in _integer_compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            order_nd = int(np.prod(order_1d))
            if order_nd == 0:
                continue

            # 生成笛卡尔积索引
            indices = np.zeros((order_nd, dim_num), dtype=int)
            for d in range(dim_num):
                if d == 0:
                    repeats = 1
                    tiles = order_nd // order_1d[d]
                else:
                    repeats = repeats * order_1d[d - 1]
                    tiles = order_nd // (repeats * order_1d[d])
                indices[:, d] = np.tile(np.repeat(np.arange(order_1d[d]), repeats), tiles)

            for pt in range(order_nd):
                pt_coords = np.zeros(dim_num, dtype=float)
                w = float(coeff)
                for d in range(dim_num):
                    n = order_1d[d]
                    x, wg = laguerre_abscissa_and_weights(n)
                    pt_coords[d] = x[indices[pt, d]]
                    w *= wg[indices[pt, d]]
                grid_points_list.append(pt_coords)
                grid_weights_list.append(w)

    if len(grid_points_list) == 0:
        return np.zeros((dim_num, 0)), np.zeros(0)
    grid_points = np.column_stack(grid_points_list)
    grid_weights = np.array(grid_weights_list, dtype=float)
    return grid_points, grid_weights


def propagate_uncertainty(model_func, dim_num, level_max,
                          param_means, param_stds):
    """
    对 model_func 进行不确定性传播。
    model_func: 接受 (n_samples, dim_num) 数组，返回 (n_samples,)
    返回: mean, variance, std
    """
    points, weights = sparse_grid_laguerre(dim_num, level_max)
    n = points.shape[1]
    # 将 Laguerre 节点（假设为指数型随机变量）映射到对数正态参数
    samples = np.zeros((dim_num, n), dtype=float)
    for d in range(dim_num):
        # x ~ Exp(lambda=1), 映射到对数正态: V = exp(mu + sigma * Phi^{-1}(1 - exp(-x)))
        # 简化：直接用 V = param_means[d] + param_stds[d] * (points[d, :] - 1.0)
        samples[d, :] = param_means[d] + param_stds[d] * (points[d, :] - 1.0)
        samples[d, :] = np.clip(samples[d, :], param_means[d] * 0.1, param_means[d] * 3.0)

    vals = model_func(samples.T)
    vals = np.asarray(vals, dtype=float)
    mean = np.sum(weights * vals)
    var = np.sum(weights * vals ** 2) - mean ** 2
    return mean, max(var, 0.0), np.sqrt(max(var, 0.0))
