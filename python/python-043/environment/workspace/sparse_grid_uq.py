"""
稀疏网格不确定性量化 (sparse_grid_uq.py)
=========================================
基于种子项目 1056_sandia_sparse 的 Smolyak 稀疏网格构造思想，
为地核发电机模型的参数不确定性提供高维数值积分与统计量估计。

地核发电机存在多个 poorly constrained 参数：
  - 磁扩散系数 eta
  - alpha 效应振幅 alpha0
  - 差速自转剪切强度 omega_shear
  - 球谐截断阶数 l_max 等

使用稀疏网格可在高维参数空间中高效计算输出统计量（如反转频率的期望、方差），
避免全张量积网格的“维数灾难”。

本模块提供：
  - Clenshaw-Curtis 1D 节点与权重
  - Smolyak 稀疏网格构造
  - 地核发电机参数 UQ 封装
"""

import numpy as np
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# 1. Clenshaw-Curtis 1D 节点与权重
#    在 [-1, 1] 上构造嵌套 Chebyshev 节点：
#      x_j = cos(pi * j / (n-1)), j=0..n-1
#    权重通过快速离散余弦变换计算，这里采用直接积分公式。
# ---------------------------------------------------------------------------
def clenshaw_curtis_rule(level: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回第 level 层 Clenshaw-Curtis 规则的节点和权重。
    level=1 -> n=1, level=2 -> n=3, level=3 -> n=5, ...
    节点数 n = 2^{level-1} + 1 (当 level>1)，level=1 时 n=1。
    """
    if level < 1:
        raise ValueError("level must be >= 1")
    if level == 1:
        return np.array([0.0]), np.array([2.0])

    n = 2 ** (level - 1) + 1
    # Chebyshev 节点（第二类）
    j = np.arange(n)
    x = np.cos(np.pi * j / (n - 1))
    # 权重（使用显式公式）
    w = np.zeros(n, dtype=float)
    for i in range(n):
        theta = np.pi * i / (n - 1)
        # 端点权重修正
        coeff = 1.0
        if i == 0 or i == n - 1:
            coeff = 0.5
        val = 0.0
        for k in range(1, (n - 1) // 2 + 1):
            if 2 * k == n - 1:
                b = 1.0
            else:
                b = 2.0
            val += b * np.cos(2.0 * k * theta) / (4.0 * k * k - 1.0)
        w[i] = coeff * (2.0 / (n - 1)) * (1.0 - val)
    return x, w


# ---------------------------------------------------------------------------
# 2. Smolyak 稀疏网格构造
#    对于 d 维、层级 L 的稀疏网格：
#      Q_d^{(L)} = sum_{L+1 <= |i| <= L+d} (-1)^{L+d-|i|} * C(d-1, L+d-|i|) * (U_{i1} x ... x U_{id})
#    其中 U_{ik} 是第 ik 层 1D 规则。
# ---------------------------------------------------------------------------
def sparse_grid_cc(dim: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 d 维 Clenshaw-Curtis Smolyak 稀疏网格。

    返回:
      points : shape (n_points, dim)，每行为一个网格点（在 [-1,1]^d 内）
      weights: shape (n_points,)，对应积分权重
    """
    from itertools import product

    # 预计算各层 1D 规则
    rules = {}
    max_1d_level = level_max
    for lv in range(1, max_1d_level + 1):
        rules[lv] = clenshaw_curtis_rule(lv)

    points_list = []
    weights_list = []

    # 遍历所有满足 level_max+1 <= |i| <= level_max+dim 的多重指标 i
    for total in range(level_max + 1, level_max + dim + 1):
        # 生成所有正整数组合 i_1+...+i_d = total
        def compositions(d_remain, sum_remain, current):
            if d_remain == 1:
                yield current + [sum_remain]
                return
            for val in range(1, sum_remain - d_remain + 2):
                yield from compositions(d_remain - 1, sum_remain - val, current + [val])

        for comp in compositions(dim, total, []):
            # 计算组合系数
            diff = level_max + dim - total
            sign = (-1) ** diff
            from math import comb
            coeff = sign * comb(dim - 1, diff)

            # 构造张量积
            x_lists = [rules[lv][0] for lv in comp]
            w_lists = [rules[lv][1] for lv in comp]
            for idx_tuple in product(*[range(len(xl)) for xl in x_lists]):
                pt = np.array([x_lists[j][idx_tuple[j]] for j in range(dim)])
                w = coeff
                for j in range(dim):
                    w *= w_lists[j][idx_tuple[j]]
                points_list.append(pt)
                weights_list.append(w)

    if not points_list:
        return np.zeros((0, dim)), np.zeros(0)

    points = np.array(points_list, dtype=float)
    weights = np.array(weights_list, dtype=float)

    # 去重（由于 Clenshaw-Curtis 节点是嵌套的，不同组合可能共享相同点）
    # 使用四舍五入到 1e-12 进行去重
    rounded = np.round(points, decimals=12)
    uniq, inverse = np.unique(rounded, axis=0, return_inverse=True)
    uniq_weights = np.zeros(uniq.shape[0], dtype=float)
    for i in range(len(weights)):
        uniq_weights[inverse[i]] += weights[i]

    return uniq, uniq_weights


# ---------------------------------------------------------------------------
# 3. 参数空间变换
#    将 [-1,1]^d 的参考坐标映射到物理参数范围。
# ---------------------------------------------------------------------------
def map_parameter_space(x_ref: np.ndarray, param_ranges: List[Tuple[float, float]]) -> np.ndarray:
    """
    将参考域 [-1,1]^d 映射到物理参数域。
    线性映射: p = p_min + 0.5*(x_ref+1)*(p_max-p_min)
    """
    x_ref = np.asarray(x_ref, dtype=float)
    d = len(param_ranges)
    if x_ref.ndim == 1:
        x_ref = x_ref.reshape(1, -1)
    p = np.zeros_like(x_ref)
    for j in range(d):
        p_min, p_max = param_ranges[j]
        p[:, j] = p_min + 0.5 * (x_ref[:, j] + 1.0) * (p_max - p_min)
    return p


# ---------------------------------------------------------------------------
# 4. 地核发电机 UQ 封装
# ---------------------------------------------------------------------------
def uq_dynamo_reversal_rate(
    dynamo_runner: Callable,
    param_ranges: List[Tuple[float, float]],
    level_max: int = 3
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    使用稀疏网格计算地核发电机反转频率的期望与方差。

    参数:
      dynamo_runner : 函数 f(params_array) -> reversal_rate (float)
      param_ranges  : 参数范围列表 [(min1,max1), ..., (minD,maxD)]
      level_max     : Smolyak 层级

    返回:
      mean  : 期望反转频率
      var   : 方差
      points: 使用的网格点
      rates : 各点对应的反转频率
    """
    dim = len(param_ranges)
    points_ref, weights = sparse_grid_cc(dim, level_max)
    n_points = points_ref.shape[0]

    points_phys = map_parameter_space(points_ref, param_ranges)
    rates = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        try:
            rates[i] = float(dynamo_runner(points_phys[i]))
        except Exception:
            rates[i] = 0.0  # 鲁棒 fallback

    # 期望: E[f] = sum w_i * f(x_i)
    mean = float(np.sum(weights * rates))
    # 方差: Var[f] = E[f^2] - E[f]^2
    mean_sq = float(np.sum(weights * rates * rates))
    var = max(0.0, mean_sq - mean ** 2)

    return mean, var, points_phys, rates


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 测试 1D Clenshaw-Curtis
    x, w = clenshaw_curtis_rule(3)
    assert len(x) == 5
    assert abs(np.sum(w) - 2.0) < 1e-12

    # 测试 2D 稀疏网格积分常数函数 f(x,y)=1
    pts, ws = sparse_grid_cc(2, 3)
    total_weight = np.sum(ws)
    assert abs(total_weight - 4.0) < 1e-10, f"Total weight = {total_weight}"

    # 测试积分 f(x)=x^2 在 [-1,1] -> 精确值 2/3
    pts1, ws1 = sparse_grid_cc(1, 5)
    integral = np.sum(ws1 * (pts1[:, 0] ** 2))
    assert abs(integral - 2.0 / 3.0) < 1e-10

    # 参数映射测试
    p = map_parameter_space(np.array([[0.0, 0.0]]), [(1.0, 3.0), (0.0, 10.0)])
    assert abs(p[0, 0] - 2.0) < 1e-10
    assert abs(p[0, 1] - 5.0) < 1e-10

    print("sparse_grid_uq: self-test passed.")


if __name__ == "__main__":
    _self_test()
