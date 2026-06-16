"""
稀疏网格构造模块 (Sparse Grid Construction)
=============================================
实现 Smolyak 稀疏网格算法, 用于高维随机空间的自适应求积。
核心公式:

  Smolyak 算子:
    A(q, D) = Σ_{q ≤ |i|₁ ≤ q+D-1} (-1)^{q+D-|i|₁} C(D-1, q+D-1-|i|₁) (Q^{i₁} ⊗ ... ⊗ Q^{i_D})

  其中 i = (i₁,...,i_D) ∈ ℕ^D, |i|₁ = Σ i_d,  Q^k 是 1D k 阶求积规则。

  稀疏网格点数: |Θ_{q,D}| = Σ_{q ≤ |i|₁ ≤ q+D-1} (-1)^{q+D-|i|₁} C(D-1, q+D-1-|i|₁) Π_{d=1}^D n_{i_d}

  对比全张量积: Π_{d=1}^D n_{i_d} — 维数灾难!

  Smolyak 代数精度: 对于每维精度 p, 需要 level q = p + D - 1

参考文献:
  Smolyak, S.A. (1963). Some quadrature and interpolation formulas for tensor product classes.
  Gerstner, T. & Griebel, M. (1998). Numerical integration using sparse grids.
"""

import numpy as np
from itertools import product as iterproduct
from typing import List, Tuple, Dict, Optional
import math


def multi_index_set_smolyak(D: int, level: int) -> List[Tuple[int, ...]]:
    """
    生成 Smolyak 多指标集:
      I_{q,D} = {i ∈ ℕ^D : q ≤ |i|₁ ≤ q + D - 1}

    其中 q = level, D = 随机维度。

    参数:
        D: 随机空间维度
        level: Smolyak level (q ≥ D)

    返回:
        多指标列表, 每个元素是长度为 D 的元组
    """
    if D <= 0:
        return [()]
    if level < D:
        level = D

    indices = []
    _enumerate_multi_indices(D, level, level + D - 1, [], indices)
    return indices


def _enumerate_multi_indices(
    remaining_dims: int,
    remaining_sum_min: int,
    remaining_sum_max: int,
    current: List[int],
    result: List[Tuple[int, ...]]
):
    """
    递归枚举多指标集 (回溯法)。

    约束: Σ i_d = s, 其中 q_min ≤ s ≤ q_max
    每个 i_d ≥ 1 (正整数指标)
    """
    if remaining_dims == 0:
        if remaining_sum_min <= 0 <= remaining_sum_max:
            result.append(tuple(current))
        return

    for v in range(max(1, remaining_sum_min - (remaining_dims - 1) * (remaining_sum_max)),
                   remaining_sum_max + 1):
        if v < 1:
            continue
        current.append(v)
        _enumerate_multi_indices(
            remaining_dims - 1,
            max(remaining_sum_min - v, 0),
            remaining_sum_max - v,
            current,
            result
        )
        current.pop()


def smolyak_coefficients(D: int, level: int, multi_indices: List[Tuple[int, ...]]) -> np.ndarray:
    """
    计算 Smolyak 组合系数:
      c_i = (-1)^{q+D-|i|₁} C(D-1, q+D-1-|i|₁)

    其中 C(n,k) = n! / (k!(n-k)!) 是二项式系数。

    这些系数确保不同 level 的张量积贡献互相抵消,
    仅保留必要的高阶交叉项。

    参数:
        D: 维度
        level: Smolyak level q
        multi_indices: 多指标列表

    返回:
        系数数组, shape (len(multi_indices),)
    """
    coeffs = np.zeros(len(multi_indices))
    q = level
    for idx, mi in enumerate(multi_indices):
        s = sum(mi)
        k = q + D - 1 - s
        if 0 <= k <= D - 1:
            sign = (-1) ** k
            binom = math.comb(D - 1, k)
            coeffs[idx] = sign * binom
    return coeffs


def n_points_1d(level_1d: int, rule_type: str = 'gauss') -> int:
    """
    1D 求积规则在给定 level 下的点数:
      level 1: 1 point  (中点规则)
      level k (k≥2): 2^{k-1} + 1 points (嵌套规则, 如 Clenshaw-Curtis)
      或非嵌套: 2k-1 points (Gauss)

    参数:
        level_1d: 1D level (正整数)
        rule_type: 'gauss' (非嵌套) 或 'cc' (嵌套 Clenshaw-Curtis)

    返回:
        该 level 下的 1D 求积点数
    """
    if rule_type == 'cc':
        if level_1d == 1:
            return 1
        return 2 ** (level_1d - 1) + 1
    else:  # Gauss 规则 (非嵌套)
        return max(1, 2 * level_1d - 1)


def sparse_grid_nodes_weights(
    D: int,
    level: int,
    rule_type: str = 'gauss',
    nodes_func=None,
    weights_func=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 Smolyak 稀疏网格的节点和权重。

    完整公式:
      A(q,D)[f] = Σ_{i ∈ I_{q,D}} c_i Σ_{j ∈ J_i} W_j f(Ξ_j)

    合并为: Σ_{k=1}^{N_sg} w_k f(ξ_k)

    其中 N_sg 是稀疏网格总点数 (去重后),
    ξ_k 是唯一的节点, w_k 是合并后的权重。

    参数:
        D: 随机空间维度
        level: Smolyak level
        rule_type: 'gauss' 或 'cc'
        nodes_func: 自定义 1D 节点函数 f(n) -> nodes
        weights_func: 自定义 1D 权重函数 f(n) -> weights

    返回:
        nodes: shape (N_sg, D)
        weights: shape (N_sg,)
    """
    if nodes_func is None:
        from polynomial_utils import gauss_legendre as _gl
        def nodes_func(n):
            return _gl(n)[0]

    if weights_func is None:
        from polynomial_utils import gauss_legendre as _gl
        def weights_func(n):
            return _gl(n)[1]

    multi_indices = multi_index_set_smolyak(D, level)
    coeffs = smolyak_coefficients(D, level, multi_indices)

    # 收集所有 (节点, 权重) 对
    all_nodes = []
    all_weights = []

    for mi, coeff in zip(multi_indices, coeffs):
        if abs(coeff) < 1e-15:
            continue

        # 构造该多指标的张量积
        nodes_1d_list = []
        weights_1d_list = []
        for d in range(D):
            n_d = n_points_1d(mi[d], rule_type)
            nd = nodes_func(n_d)
            wt = weights_func(n_d)
            nodes_1d_list.append(nd)
            weights_1d_list.append(wt)

        # 张量积
        grids = np.meshgrid(*nodes_1d_list, indexing='ij')
        w_grids = np.meshgrid(*weights_1d_list, indexing='ij')

        tp_nodes = np.column_stack([g.ravel() for g in grids])
        tp_weights = np.ones(tp_nodes.shape[0])
        for wg in w_grids:
            tp_weights *= wg.ravel()

        all_nodes.append(tp_nodes)
        all_weights.append(coeff * tp_weights)

    # 合并所有贡献
    if len(all_nodes) == 0:
        return np.zeros((0, D)), np.array([])

    combined_nodes = np.vstack(all_nodes)
    combined_weights = np.concatenate(all_weights)

    # 去重: 合并相同节点 (容差 1e-12)
    unique_nodes, unique_weights = _merge_duplicate_nodes(
        combined_nodes, combined_weights, tol=1e-12
    )

    return unique_nodes, unique_weights


def _merge_duplicate_nodes(
    nodes: np.ndarray, weights: np.ndarray, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并稀疏网格中的重复节点。

    算法:
    1. 对节点按字典序排序
    2. 扫描相邻节点, 如果距离 < tol, 合并权重

    这是 Smolyak 稀疏网格的关键步骤:
    不同多指标的张量积可能包含相同的节点,
    需要将其合并为一个节点, 权重为各贡献之和。
    """
    N = nodes.shape[0]
    if N == 0:
        return nodes, weights
    if N == 1:
        return nodes, weights

    # 四舍五入到容差级别以辅助去重
    rounded = np.round(nodes / tol) * tol

    # 用字典合并
    node_dict = {}
    for i in range(N):
        key = tuple(np.round(rounded[i], 10))
        if key in node_dict:
            node_dict[key] += weights[i]
        else:
            node_dict[key] = weights[i]

    unique_nodes = np.array(list(node_dict.keys()))
    unique_weights = np.array(list(node_dict.values()))

    # 过滤零权重节点
    mask = np.abs(unique_weights) > 1e-15
    return unique_nodes[mask], unique_weights[mask]


def estimate_sparse_grid_size(D: int, level: int, rule_type: str = 'gauss') -> int:
    """
    估计稀疏网格的点数 (不去重):
      N_raw = Σ_{i ∈ I_{q,D}} |c_i| Π_{d=1}^D n_{i_d}

    用于评估稀疏网格相对于全张量积的节省程度。
    """
    multi_indices = multi_index_set_smolyak(D, level)
    total = 0
    for mi in multi_indices:
        n_pts = 1
        for d in range(D):
            n_pts *= n_points_1d(mi[d], rule_type)
        total += n_pts
    return total


def estimate_full_grid_size(D: int, level: int, rule_type: str = 'gauss') -> int:
    """
    全张量积网格点数:
      N_full = Π_{d=1}^D n_{level}(d)

    与稀疏网格对比, 展示维数灾难的严重程度。
    """
    n = n_points_1d(level, rule_type)
    return n ** D


def anisotropic_sparse_grid(
    D: int,
    levels: List[int],
    rule_type: str = 'gauss',
    importance: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    各向异性稀疏网格 (Anisotropic Sparse Grid):

    当不同随机维度的重要性不同时, 使用各向异性 level 分配:
      level_d = importance[d] * q / Σ importance

    这比各向同性网格更高效, 因为高重要性维度获得更多求积点。

    参数:
        D: 维度
        levels: 各维度的 level 列表
        rule_type: 求积规则类型
        importance: 重要性权重 (可选)

    返回:
        nodes, weights
    """
    if importance is not None:
        total_imp = np.sum(importance)
        max_level = max(levels)
        levels = [max(1, int(round(importance[d] / total_imp * max_level * D)))
                  for d in range(D)]

    # 使用 level 列表构造非对称多指标集
    # 简化: 使用最大 level 的 Smolyak 网格, 但各维度使用不同 level
    multi_indices = multi_index_set_smolyak(D, max(levels))
    coeffs = smolyak_coefficients(D, max(levels), multi_indices)

    all_nodes = []
    all_weights = []

    from polynomial_utils import gauss_legendre as _gl

    for mi, coeff in zip(multi_indices, coeffs):
        if abs(coeff) < 1e-15:
            continue

        nodes_1d_list = []
        weights_1d_list = []
        for d in range(D):
            # 各维度使用 capped level
            effective_level = min(mi[d], levels[d])
            effective_level = max(1, effective_level)
            n_d = n_points_1d(effective_level, rule_type)
            nd, wt = _gl(n_d)
            nodes_1d_list.append(nd)
            weights_1d_list.append(wt)

        grids = np.meshgrid(*nodes_1d_list, indexing='ij')
        w_grids = np.meshgrid(*weights_1d_list, indexing='ij')

        tp_nodes = np.column_stack([g.ravel() for g in grids])
        tp_weights = np.ones(tp_nodes.shape[0])
        for wg in w_grids:
            tp_weights *= wg.ravel()

        all_nodes.append(tp_nodes)
        all_weights.append(coeff * tp_weights)

    if len(all_nodes) == 0:
        return np.zeros((0, D)), np.array([])

    combined_nodes = np.vstack(all_nodes)
    combined_weights = np.concatenate(all_weights)

    unique_nodes, unique_weights = _merge_duplicate_nodes(
        combined_nodes, combined_weights, tol=1e-12
    )

    return unique_nodes, unique_weights
