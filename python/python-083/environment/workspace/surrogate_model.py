"""
surrogate_model.py
==================
高维代理模型模块：Smolyak 稀疏网格多维插值。
整合自：
  - 1110_sparse_interp_nd：Smolyak 稀疏网格 + Clenshaw-Curtis 节点 + 多维 Lagrange 插值

物理背景：
  在多目标拓扑优化中，设计空间维度高（材料密度、晶格尺寸、工艺参数等）。
  全张量积网格遭遇"维度灾难"，Smolyak 稀疏网格通过组合不同维度的
  一维张量积规则，以多项式精度为代价大幅减少采样点数量。

核心公式：
  Smolyak 稀疏网格插值：
      A(q,d) = Σ_{q-d+1 ≤ |i| ≤ q} (-1)^{q-|i|} · C(d-1, q-|i|) · (U^{i1} ⊗ ... ⊗ U^{id})

  其中 U^{ik} 为一维差分插值算子，i = (i1, ..., id) 为层级多指标，
  |i| = i1 + ... + id，q 为总层级。

  一维 Clenshaw-Curtis 节点：
      x_j^k = cos(j·π / n_k),  j = 0, 1, ..., n_k
      n_k = 2^{k-1} + 1  (k ≥ 1),  n_0 = 1
"""

import numpy as np
from typing import Callable, Tuple, List


# =============================================================================
# 1. Clenshaw-Curtis 节点与权重
# =============================================================================

def clenshaw_curtis_points(level: int) -> np.ndarray:
    """
    计算一维 Clenshaw-Curtis 节点。
    level k: n_k = 2^{k-1} + 1  (k>=1), n_0 = 1
    节点：x_j = cos(j·π / (n_k - 1)), j = 0, ..., n_k-1
    """
    if level == 0:
        return np.array([0.5], dtype=np.float64)  # 中点映射到 [0,1]
    n = 2**(level - 1) + 1
    j = np.arange(n)
    # 标准 CC 节点在 [-1,1]
    x = np.cos(j * np.pi / (n - 1))
    # 线性映射到 [0,1]
    return 0.5 * (x + 1.0)


def clenshaw_curtis_weights(level: int) -> np.ndarray:
    """
    Clenshaw-Curtis 积分权重（基于 DCT 的显式公式）。
    对于 level=0 返回权重 1.0（单点中点规则）。
    """
    if level == 0:
        return np.array([1.0], dtype=np.float64)
    n = 2**(level - 1) + 1
    x = clenshaw_curtis_points(level)
    # 标准 CC 权重（在 [-1,1] 上）
    w = np.zeros(n, dtype=np.float64)
    if n == 2:
        w[:] = 0.5
    else:
        # 基于梯形+端点修正的 CC 权重
        w[0] = 1.0 / (n * (n - 2))
        w[-1] = w[0]
        for j in range(1, n - 1):
            if j % 2 == 0:
                w[j] = 2.0 / (1.0 - j * j)
            else:
                w[j] = 0.0
        # 归一化（在 [-1,1] 上积分常数 1 得 2）
        w = w / np.sum(w) * 2.0
    # 由于节点已映射到 [0,1]，权重需乘以 0.5
    return w * 0.5


# =============================================================================
# 2. 一维 Lagrange 基函数
# =============================================================================

def lagrange_basis_1d(xi: float, nodes: np.ndarray, k: int) -> float:
    """
    一维 Lagrange 基函数 L_k(ξ) = Π_{j≠k} (ξ - x_j) / (x_k - x_j)
    """
    n = len(nodes)
    if n == 1:
        return 1.0
    result = 1.0
    xk = nodes[k]
    for j in range(n):
        if j != k:
            denom = xk - nodes[j]
            if abs(denom) < 1e-14:
                continue
            result *= (xi - nodes[j]) / denom
    return result


def interpolate_1d(xi: float, nodes: np.ndarray, values: np.ndarray) -> float:
    """一维 Lagrange 插值。"""
    result = 0.0
    for k in range(len(nodes)):
        result += values[k] * lagrange_basis_1d(xi, nodes, k)
    return result


# =============================================================================
# 3. Smolyak 稀疏网格组合
# =============================================================================

def generate_multi_index_combinations(d: int, q: int) -> List[Tuple[int, ...]]:
    """
    生成满足 q-d+1 ≤ |i| ≤ q 的所有 d 维非负整数多指标 i。
    采用递归枚举。
    """
    results = []
    min_sum = max(0, q - d + 1)
    max_sum = q

    def backtrack(pos: int, current: List[int], current_sum: int):
        if pos == d:
            if min_sum <= current_sum <= max_sum:
                results.append(tuple(current))
            return
        # 剩余维度能达到的最小和
        remaining = d - pos - 1
        for val in range(0, max_sum - current_sum + 1):
            # 剪枝：即使后面全取0，也需满足 min_sum
            if current_sum + val + 0 > max_sum:
                break
            if current_sum + val + remaining < min_sum:
                continue
            current.append(val)
            backtrack(pos + 1, current, current_sum + val)
            current.pop()

    backtrack(0, [], 0)
    return results


def smolyak_coefficient(d: int, q: int, i_sum: int) -> int:
    """
    Smolyak 组合系数：(-1)^{q-|i|} · C(d-1, q-|i|)
    """
    from math import comb
    s = q - i_sum
    if s < 0 or s > d - 1:
        return 0
    sign = (-1)**s
    return sign * comb(d - 1, s)


class SmolyakSparseGrid:
    """
    d 维 Smolyak 稀疏网格插值器。
    """
    def __init__(self, d: int, q: int):
        self.d = d
        self.q = q
        self.multi_indices = generate_multi_index_combinations(d, q)
        self.nodes_dict = {}
        self.values_dict = {}
        self._build_grid()

    def _build_grid(self):
        """构建所有层级组合的节点张量积。"""
        for mi in self.multi_indices:
            key = tuple(mi)
            nodes_list = []
            for dim, level in enumerate(mi):
                nodes_list.append(clenshaw_curtis_points(level))
            # 张量积节点
            grids = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.column_stack([g.ravel() for g in grids])
            self.nodes_dict[key] = flat_nodes

    def sample_function(self, func: Callable[[np.ndarray], float]):
        """在所有网格点上采样函数值。"""
        for key, nodes in self.nodes_dict.items():
            vals = np.array([func(pt) for pt in nodes], dtype=np.float64)
            self.values_dict[key] = vals

    def interpolate(self, x: np.ndarray) -> float:
        """
        在点 x ∈ [0,1]^d 处求稀疏网格插值。
        """
        result = 0.0
        for mi in self.multi_indices:
            key = tuple(mi)
            coeff = smolyak_coefficient(self.d, self.q, sum(mi))
            if coeff == 0:
                continue
            nodes_list = []
            for dim, level in enumerate(mi):
                nodes_list.append(clenshaw_curtis_points(level))

            # 计算该层级张量积插值在 x 处的值
            flat_nodes = self.nodes_dict[key]
            flat_vals = self.values_dict.get(key)
            if flat_vals is None:
                continue

            # 一维一维地做 Lagrange 插值
            # 先找到 x 在各维的一维基值
            n_per_dim = [len(nl) for nl in nodes_list]
            # 遍历所有张量积组合
            val_sum = 0.0
            for idx_flat in range(len(flat_vals)):
                # 将 flat 索引转为多维索引
                idx_multi = []
                temp = idx_flat
                for dim in range(self.d - 1, -1, -1):
                    idx_multi.append(temp % n_per_dim[dim])
                    temp //= n_per_dim[dim]
                idx_multi = idx_multi[::-1]

                # 计算 Lagrange 基函数值
                basis_val = 1.0
                for dim in range(self.d):
                    basis_val *= lagrange_basis_1d(
                        x[dim], nodes_list[dim], idx_multi[dim])
                val_sum += flat_vals[idx_flat] * basis_val

            result += coeff * val_sum
        return result

    def get_total_points(self) -> int:
        """稀疏网格总采样点数。"""
        return sum(len(pts) for pts in self.nodes_dict.values())


# =============================================================================
# 4. 全张量积网格对比（用于展示稀疏网格优势）
# =============================================================================

def full_tensor_product_points(d: int, n_per_dim: int) -> np.ndarray:
    """
    d 维全张量积网格，每维 n_per_dim 个点（均匀分布在 [0,1]）。
    """
    x1d = np.linspace(0.0, 1.0, n_per_dim)
    grids = np.meshgrid(*([x1d] * d), indexing='ij')
    return np.column_stack([g.ravel() for g in grids])


def compare_grid_cardinality(d: int, q: int) -> Tuple[int, int]:
    """
    对比 Smolyak 稀疏网格与同等精度全张量积网格的采样点数。
    """
    sg = SmolyakSparseGrid(d, q)
    n_sparse = sg.get_total_points()
    # 全张量积：每维取最大节点数
    max_level = max(sg.nodes_dict.keys(), key=lambda k: max(k))
    n_max = max(len(clenshaw_curtis_points(level)) for level in max_level)
    n_full = n_max ** d
    return n_sparse, n_full


# =============================================================================
# 5. 应用：多目标优化响应面
# =============================================================================

def build_compliance_surrogate(design_sampler: Callable[[np.ndarray], float],
                                d: int, q: int = 4) -> SmolyakSparseGrid:
    """
    为柔度函数构建 Smolyak 稀疏网格代理模型。

    design_sampler: f(x) -> compliance，其中 x ∈ [0,1]^d 为归一化设计参数。
    """
    sg = SmolyakSparseGrid(d, q)
    sg.sample_function(design_sampler)
    return sg


def test_function_oscillatory(x: np.ndarray) -> float:
    """
    测试函数（极坐标振荡，源自 sparse_interp_nd 的测试函数 f12_f0_nd）：
        f(x) = cos(2π·||x||) · exp(-||x||²/2)
    """
    r = np.linalg.norm(x)
    return np.cos(2.0 * np.pi * r) * np.exp(-0.5 * r * r)
