"""
sparse_grid.py — 自适应稀疏网格求积模块
==========================================
实现 Smolyak 稀疏网格构造、自适应求积点加密、
以及误差估计。用于高维随机空间的高效积分。

核心公式:
  Smolyak 公式:
    A(q, d) = Σ_{(-1)^{q-|i|}} C(d-1, q-|i|) (Q^{i_1} ⊗ ... ⊗ Q^{i_d})

  自适应加密:
    对每个活跃节点, 估计局部积分误差
    ε_k = |w_k f(ξ_k)| / Σ_j |w_j f(ξ_j)|
    若 ε_k > tolerance, 在该节点周围加密

映射种子项目:
  - 932_pyramid_grid: 金字塔层级结构 → Smolyak 层级
  - 068_ball_integrals: 高维积分 → 概率空间积分
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from config import SparseGridConfig, MeasureConfig, GlobalConfig
from measure import gauss_quadrature
import itertools


class SparseGridNode:
    """稀疏网格中的单个节点"""
    __slots__ = ['coords', 'weight', 'level', 'active']

    def __init__(self, coords: np.ndarray, weight: float,
                 level: int):
        self.coords = coords
        self.weight = weight
        self.level = level
        self.active = True


class AdaptiveSparseGrid:
    """
    自适应 Smolyak 稀疏网格。

    核心思想: 使用稀疏网格代替全张量积网格,
    将求积点数从 O(n^d) 降低到 O(n * (log n)^{d-1})。

    映射 932_pyramid_grid: 利用金字塔式的层级结构
    管理不同精度的求积点。

    映射 068_ball_integrals: 高维概率空间积分
    通过稀疏网格高效实现。
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.sg_config = config.sparse_grid
        self.measures = config.measures
        self.n_dim = len(config.measures)

        # 节点列表
        self.nodes: List[SparseGridNode] = []
        self._built = False

        # 构建稀疏网格
        self._build_grid()

    def _build_grid(self):
        """构建 Smolyak 稀疏网格"""
        d = self.n_dim
        q = self.sg_config.level + d

        # 一维求积规则层级: m(i) = i (linear growth)
        all_nodes = []
        all_weights = []

        # 枚举 Smolyak 组合
        for level_sum in range(max(d, q - self.sg_config.level),
                               q + 1):
            sign = (-1) ** (q - level_sum)
            from math import comb
            coeff = comb(d - 1, q - level_sum)

            if coeff == 0:
                continue

            # 枚举 i_1 + ... + i_d = level_sum, i_k >= 1
            for mi in self._compositions(level_sum, d):
                # 构造此组合的张量积求积
                nodes_1d = []
                weights_1d = []
                for dim_idx in range(d):
                    n_pts = max(mi[dim_idx], 1)
                    nd, wt = gauss_quadrature(
                        self.measures[dim_idx], n_pts)
                    nodes_1d.append(nd)
                    weights_1d.append(wt)

                # 张量积
                grids = np.meshgrid(*nodes_1d, indexing='ij')
                pts = np.column_stack([g.ravel() for g in grids])

                w_grids = np.meshgrid(*weights_1d, indexing='ij')
                wts = np.ones(w_grids[0].size)
                for wg in w_grids:
                    wts *= wg.ravel()

                wts *= sign * coeff

                all_nodes.append(pts)
                all_weights.append(wts)

        if len(all_nodes) > 0:
            nodes_all = np.vstack(all_nodes)
            weights_all = np.concatenate(all_weights)

            # 合并重复节点
            self.nodes = self._merge_nodes(nodes_all, weights_all)
        else:
            # 退化情况: 使用单点求积
            mean_pt = np.array([m.mean for m in self.measures])
            self.nodes = [SparseGridNode(mean_pt, 1.0, 0)]

        self._built = True

    def _compositions(self, n: int, k: int) -> list:
        """生成 n 的 k-组合 (每个分量 >= 1)"""
        if k == 1:
            return [(n,)] if n >= 1 else []
        result = []
        for first in range(1, n - k + 2):
            for rest in self._compositions(n - first, k - 1):
                result.append((first,) + rest)
        return result

    def _merge_nodes(self, nodes: np.ndarray,
                     weights: np.ndarray,
                     tol: float = 1e-12) -> List[SparseGridNode]:
        """合并重复节点"""
        merged: Dict[tuple, float] = {}
        for i in range(len(nodes)):
            key = tuple(np.round(nodes[i] / tol) * tol)
            if key in merged:
                merged[key] += weights[i]
            else:
                merged[key] = weights[i]

        result = []
        for key, w in merged.items():
            if abs(w) > 1e-15:
                coords = np.array(key)
                result.append(SparseGridNode(coords, w, 0))

        return result

    @property
    def n_points(self) -> int:
        return len(self.nodes)

    def get_nodes_array(self) -> np.ndarray:
        """返回节点坐标, shape (n_points, d)"""
        return np.array([n.coords for n in self.nodes])

    def get_weights_array(self) -> np.ndarray:
        """返回求积权重, shape (n_points,)"""
        return np.array([n.weight for n in self.nodes])

    def integrate(self, func_values: np.ndarray) -> float:
        """
        计算积分 ∫ f(ξ) w(ξ) dξ ≈ Σ_k weight_k * f(ξ_k)

        参数:
            func_values: f 在各节点上的值, shape (n_points,)

        返回:
            积分近似值
        """
        weights = self.get_weights_array()
        return float(np.sum(weights * func_values))

    def estimate_quadrature_error(self,
                                  func_values: np.ndarray) -> float:
        """
        估计求积误差。

        使用 hierarchical 差分:
          误差 ≈ |A(q) - A(q-1)|

        简化版本: 使用权重的绝对值之和与积分值的差异。

        参数:
            func_values: f 在各节点上的值

        返回:
            误差估计
        """
        weights = self.get_weights_array()

        # 权重和应为 1 (概率测度)
        w_sum = np.sum(weights)
        w_error = abs(w_sum - 1.0)

        # 函数值变化度
        if len(func_values) > 1:
            f_range = np.max(func_values) - np.min(func_values)
            f_var = np.std(func_values)
            # 误差估计: O(h^{2*level}) * 函数变化
            h_factor = 2.0 ** (-2 * self.sg_config.level)
            quad_error = h_factor * f_var
        else:
            quad_error = 0.0

        return quad_error + w_error * abs(np.mean(func_values))

    def adaptively_refine(self, func_values: np.ndarray,
                          tolerance: float = 1e-6) -> int:
        """
        自适应加密稀疏网格。

        对贡献大的节点周围添加新节点。

        参数:
            func_values: 当前节点上的函数值
            tolerance:   加密容差

        返回:
            新增节点数
        """
        weights = self.get_weights_array()
        contributions = np.abs(weights * func_values)
        total = np.sum(contributions)

        if total < 1e-15:
            return 0

        # 找到贡献最大的节点
        relative_contrib = contributions / total
        candidates = np.where(relative_contrib > tolerance)[0]

        n_added = 0
        new_nodes = []

        for idx in candidates:
            node = self.nodes[idx]
            # 在每个维度上添加扰动节点
            for dim in range(self.n_dim):
                eps = 0.1 * (self.measures[dim].support[1] -
                             self.measures[dim].support[0])
                eps /= (self.sg_config.level + 1)

                # 正方向扰动
                coords_plus = node.coords.copy()
                coords_plus[dim] += eps
                support = self.measures[dim].support
                coords_plus[dim] = np.clip(
                    coords_plus[dim], support[0], support[1])
                new_w = node.weight * 0.1
                new_nodes.append(
                    SparseGridNode(coords_plus, new_w,
                                   node.level + 1))

                # 负方向扰动
                coords_minus = node.coords.copy()
                coords_minus[dim] -= eps
                coords_minus[dim] = np.clip(
                    coords_minus[dim], support[0], support[1])
                new_nodes.append(
                    SparseGridNode(coords_minus, -new_w * 0.5,
                                   node.level + 1))

                n_added += 2

        self.nodes.extend(new_nodes)
        return n_added


def compute_sparse_grid_statistics(
    sparse_grid: AdaptiveSparseGrid,
    pce_evaluator,
    n_samples_estimate: int = 1000
) -> dict:
    """
    使用稀疏网格计算 PCE 的统计量。

    参数:
        sparse_grid:     稀疏网格
        pce_evaluator:   PCE 求值函数, 输入节点返回函数值
        n_samples_estimate: MC 估计样本数 (用于对比)

    返回:
        stats: 统计量字典
    """
    nodes = sparse_grid.get_nodes_array()
    weights = sparse_grid.get_weights_array()

    # 计算函数值
    func_values = pce_evaluator(nodes)

    # 均值
    mean = sparse_grid.integrate(func_values)

    # 方差: E[f^2] - (E[f])^2
    f_squared = func_values ** 2
    mean_f2 = sparse_grid.integrate(f_squared)
    variance = max(mean_f2 - mean ** 2, 0.0)

    # 误差估计
    error = sparse_grid.estimate_quadrature_error(func_values)

    return {
        "mean": mean,
        "variance": variance,
        "std": np.sqrt(variance),
        "quadrature_error": error,
        "n_points": sparse_grid.n_points,
    }
