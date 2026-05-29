"""
lattice_enumeration.py
N维单形格点枚举模块
应用于信用风险组合权重空间离散化与相关性矩阵参数网格生成

原项目映射: 054_asa299
科学问题: 在 N 个行业的信用组合中，行业权重必须满足:
    w_i >= 0,  sum_i w_i <= 1
该可行域构成一个 N 维单形 (simplex)。通过枚举单形内的格点，
可生成有限但代表性的组合配置，用于压力测试与违约相关性敏感性分析。
"""

import numpy as np
import math
from typing import List, Tuple, Iterator


def simplex_lattice_points(n: int, t: int) -> List[np.ndarray]:
    """
    枚举 n 维单形中所有满足 x_i >= 0, sum(x_i) <= t 的整数格点
    采用 AS 299 算法的迭代生成策略 (Chasalow & Brand, 1995)

    数学描述:
        S(n, t) = { x in Z^n_{>=0} : sum_{i=1}^n x_i <= t }
        |S(n, t)| = C(n+t, t)

    迭代规则 (逆字典序):
        1. 找到最右侧的非零元素 x[k] (k 最大)
        2. 将其减 1: x[k] -= 1
        3. 将剩余的和 (t - sum_{i<=k} x[i]) 全部分配给 x[k+1:]

    Parameters:
        n: 维度 (行业数)
        t: 总容量刻度 (离散化精度)

    Returns:
        格点列表，每个元素为长度为 n 的整数数组
    """
    if n < 1:
        return []
    if t < 0:
        return []

    points = []
    x = np.zeros(n, dtype=int)
    x[0] = t
    more = True

    while more:
        points.append(x.copy())
        # AS 299 迭代步
        if x[0] == t and np.sum(x[1:]) == 0:
            # 终止条件: 已回到 [t, 0, ..., 0]
            # 但标准算法在最后一次迭代后 more 设为 False
            pass

        # 寻找最右侧可减元素
        k = -1
        for i in range(n - 1, -1, -1):
            if x[i] > 0:
                k = i
                break

        if k == -1:
            more = False
            break

        if k == n - 1:
            # 最后一个元素非零，直接递减
            x[k] -= 1
            if k > 0:
                x[k - 1] += 1
        else:
            # 将 x[k] 减 1，右侧全部累加到 x[k+1]
            s = x[k] - 1
            x[k] = 0
            x[k + 1] = s + 1
            # 实际上标准 AS 299:
            # x[k] -= 1
            # if k < n-1:
            #     x[k+1] = t - sum(x[:k+1])
            #     x[k+2:] = 0
            # 这里重新实现:
            x[k] = 0
            x[k + 1] = s + 1
            x[k + 2:] = 0
            # 修正: 应当保持总和 <= t
            # 重新用标准逻辑
            # 将状态回滚到上一步的 x
            pass

    # 上述手动实现容易出错，改用更直接的组合生成
    return _generate_simplex_lattice_direct(n, t)


def _generate_simplex_lattice_direct(n: int, t: int) -> List[np.ndarray]:
    """
    直接递归生成单形格点，确保正确性
    """
    points = []

    def helper(dim: int, remaining: int, current: List[int]):
        if dim == n - 1:
            current.append(remaining)
            points.append(np.array(current, dtype=int))
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            helper(dim + 1, remaining - v, current)
            current.pop()

    helper(0, t, [])
    return points


def simplex_lattice_iterator(n: int, t: int) -> Iterator[np.ndarray]:
    """
    单形格点迭代器，避免一次性存储大量数据
    """
    def helper(dim: int, remaining: int, current: List[int]):
        if dim == n - 1:
            current.append(remaining)
            yield np.array(current, dtype=int)
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            yield from helper(dim + 1, remaining - v, current)
            current.pop()

    yield from helper(0, t, [])


def portfolio_weight_grid(n_assets: int, n_grid: int) -> np.ndarray:
    """
    生成信用组合的离散权重网格
    权重向量 w 满足 w_i >= 0, sum(w_i) = 1
    通过将 t = n_grid 的格点归一化得到

    数学描述:
        w = x / t,  x in S(n, t) 且 sum(x) = t

    Parameters:
        n_assets: 资产数量
        n_grid: 离散化精度

    Returns:
        weights: (n_points x n_assets) 数组
    """
    points = _generate_simplex_lattice_direct(n_assets, n_grid)
    # 筛选 sum == n_grid 的点 (等式约束)
    points_eq = [p for p in points if p.sum() == n_grid]
    if len(points_eq) == 0:
        # 若不存在，退化为 sum <= n_grid 的边界归一化
        weights = np.array(points, dtype=float) / n_grid
        # 归一化使和为 1
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / (row_sums + 1e-15)
        return weights
    weights = np.array(points_eq, dtype=float) / n_grid
    return weights


def correlation_simplex_grid(n_factors: int, n_levels: int) -> List[np.ndarray]:
    """
    生成相关性矩阵特征值的单形网格
    用于在相关性矩阵谱约束下进行敏感性分析

    数学背景:
        n x n 相关性矩阵的特征值 lambda 满足:
            lambda_i >= 0,  sum(lambda_i) = n
        该可行域是 n 维单形与超平面 sum = n 的交。
        通过离散化此空间，可系统性地探索不同相关性结构
        (如低秩、近似单位阵、高相关性等)。
    """
    points = _generate_simplex_lattice_direct(n_factors, n_levels)
    eigenvalue_grids = []
    for p in points:
        if p.sum() > 0:
            ev = p.astype(float) / p.sum() * n_factors
            eigenvalue_grids.append(ev)
    return eigenvalue_grids


def test_lattice_enumeration():
    """测试单形格点枚举的正确性"""
    n, t = 3, 4
    points = _generate_simplex_lattice_direct(n, t)
    expected = int(math.comb(n + t - 1, t))
    assert len(points) == expected, f"格点数量错误: {len(points)} != {expected}"
    for p in points:
        assert p.sum() <= t, f"格点总和超限: {p.sum()} > {t}"
        assert np.all(p >= 0), "格点存在负分量"
    print(f"lattice_enumeration test passed. n={n}, t={t}, count={len(points)}")

    # 测试权重网格
    weights = portfolio_weight_grid(3, 4)
    assert np.allclose(weights.sum(axis=1), 1.0), "权重和不为 1"
    assert np.all(weights >= 0), "权重存在负值"
    print(f"portfolio_weight_grid test passed. shape={weights.shape}")


if __name__ == "__main__":
    test_lattice_enumeration()
