"""
自适应配置细化模块 (Adaptive Collocation Refinement)
=======================================================
基于后验误差估计的自适应稀疏网格细化策略。

核心思想:
  不是均匀地增加所有维度的精度, 而是根据解在各维度方向上的
  变化剧烈程度, 自适应地分配计算资源。

误差指示器:
  1. 层级差分指示器 (Hierarchical Surplus):
     Δu_i = |u(ξ_{fine}) - u(ξ_{coarse})|

  2. 局部误差指示器 (Local Error Indicator):
     η_T = ||u - Π_h u||_{L²(T)}

  3. 梯度指示器 (Gradient Indicator):
     g_d ≈ |∂u/∂ξ_d| ≈ |Δ_d u| / Δξ_d

细化策略:
  - Dörfler 标记: 选择贡献最大的指标集, 使得累计误差 > θ × 总误差
  - 最大策略: 标记误差最大的 N_mark 个指标
  - CVT 最优采样: 将新配置点放置在误差密度的 CVT 最优位置

收敛性:
  对于解析解, 自适应稀疏网格的收敛速率为:
    ε ~ N^{-2p/d} (各向同性)
    ε ~ N^{-2p/d_eff} (自适应, d_eff ≤ d)
  其中 d_eff 是有效维度。

CVT 细化 (源自 Lloyd's algorithm):
  给定误差密度函数 ρ(ξ) ∝ |∂³u/∂ξ³|:
  1. 初始化候选点集
  2. 计算 Voronoi 分割
  3. 移动每个点到其 Voronoi 单元的质心
  4. 重复直到收敛

  质心: c_i = ∫_{V_i} ξ ρ(ξ) dξ / ∫_{V_i} ρ(ξ) dξ
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from scipy.spatial import Voronoi


class AdaptiveRefinement:
    """
    自适应稀疏网格细化器。

    通过评估配置点处的解, 识别需要细化的区域,
    并添加新的配置点以提高精度。
    """

    def __init__(
        self,
        dimension: int,
        initial_level: int = 2,
        refinement_threshold: float = 1e-3,
        max_level: int = 8,
        dorfler_theta: float = 0.5
    ):
        """
        参数:
            dimension: 随机维度 D
            initial_level: 初始稀疏网格 level
            refinement_threshold: 细化阈值
            max_level: 最大 level
            dorfler_theta: Dörfler 标记参数 θ ∈ (0,1)
        """
        self.D = dimension
        self.level = initial_level
        self.threshold = refinement_threshold
        self.max_level = max_level
        self.theta = dorfler_theta

        # 存储层级信息
        self.level_history: List[Dict] = []
        self.error_history: List[float] = []

    def compute_hierarchical_surplus(
        self,
        solution_values: np.ndarray,
        nodes: np.ndarray,
        weights: np.ndarray,
        level_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算层级差分 (Hierarchical Surplus):

        对于嵌套网格, 层级差分 = 新添加点的贡献:
          Δu_k = u(ξ_k) - Π_{coarser} u(ξ_k)

        对于非嵌套网格 (如 Gauss), 使用差分近似:
          Δu_k ≈ Σ_{j ∈ neighbors} w_j (u_k - u_j) / ||ξ_k - ξ_j||

        这是后验误差估计的关键:
        |Δu_k| 大 ⟹ 该区域解变化剧烈 ⟹ 需要细化

        参数:
            solution_values: shape (N_config,)
            nodes: shape (N_config, D)
            weights: shape (N_config,)
            level_map: 每个点对应的 level (可选)

        返回:
            surplus: shape (N_config,)
        """
        N = nodes.shape[0]
        surplus = np.zeros(N)

        # 计算点对之间的距离
        for i in range(N):
            dists = np.linalg.norm(nodes - nodes[i], axis=1)
            dists[i] = np.inf  # 排除自身

            # 找最近邻
            n_neighbors = min(5, N - 1)
            nn_idx = np.argsort(dists)[:n_neighbors]

            # 加权差分
            s = 0.0
            w_total = 0.0
            for j in nn_idx:
                if dists[j] > 1e-14:
                    w = 1.0 / dists[j]
                    s += w * abs(solution_values[i] - solution_values[j])
                    w_total += w
            if w_total > 1e-14:
                surplus[i] = s / w_total

        return surplus

    def dorfler_marking(
        self, surplus: np.ndarray
    ) -> np.ndarray:
        """
        Dörfler 标记策略:
          找到最小指标集 M 使得:
            Σ_{k ∈ M} η_k ≥ θ Σ_k η_k

        其中 η_k = surplus[k] 是误差指示器。

        这保证了最少的细化操作获得最大的误差减小。

        参数:
            surplus: 误差指示器, shape (N,)

        返回:
            marked: boolean mask, shape (N,)
        """
        total = np.sum(surplus)
        target = self.theta * total

        # 降序排序
        sorted_idx = np.argsort(surplus)[::-1]

        marked = np.zeros(len(surplus), dtype=bool)
        cumulative = 0.0
        for idx in sorted_idx:
            marked[idx] = True
            cumulative += surplus[idx]
            if cumulative >= target:
                break

        return marked

    def compute_cvt_density(
        self,
        nodes: np.ndarray,
        surplus: np.ndarray
    ) -> np.ndarray:
        """
        计算 CVT 密度函数:
          ρ(ξ_k) ∝ |Δu_k|^α

        其中 α 是放大参数 (通常 α=1 或 2)。
        高密度区域将被分配更多的配置点。

        参数:
            nodes: shape (N, D)
            surplus: shape (N,)

        返回:
            density: shape (N,)
        """
        alpha = 2.0
        rho = np.abs(surplus) ** alpha
        total = np.sum(rho)
        if total > 1e-30:
            rho /= total
        else:
            rho = np.ones(len(rho)) / len(rho)
        return rho

    def lloyd_cvt_refinement(
        self,
        density_nodes: np.ndarray,
        density_weights: np.ndarray,
        n_new_points: int,
        n_iterations: int = 20
    ) -> np.ndarray:
        """
        Lloyd 算法计算 CVT 最优采样点:

        输入: 密度函数由 (节点, 权重) 离散表示
        输出: CVT 最优 generators, shape (n_new_points, D)

        算法 (1D 简化版):
        1. 初始化 generators (从密度加权随机采样)
        2. for iter = 1,...,n_iterations:
           a. 计算每个 generator 的 Voronoi 区域
           b. 计算每个 Voronoi 区域的质心 (密度加权)
           c. 移动 generator 到质心
        3. 返回收敛后的 generators

        对于 1D:
          Voronoi 区域 = 相邻 generator 之间的区间
          质心 = ∫_V ξ ρ(ξ) dξ / ∫_V ρ(ξ) dξ

        对于高维: 使用近似最近邻代替精确 Voronoi
        """
        D = density_nodes.shape[1] if density_nodes.ndim > 1 else 1
        density_nodes = np.atleast_2d(density_nodes)

        # 初始化: 从密度加权分布采样
        probs = np.abs(density_weights)
        probs_sum = np.sum(probs)
        if probs_sum > 1e-30:
            probs /= probs_sum
        else:
            probs = np.ones(len(probs)) / len(probs)

        rng = np.random.RandomState(42)
        gen_idx = rng.choice(len(probs), size=min(n_new_points, len(probs)),
                             replace=False if n_new_points <= len(probs) else True,
                             p=probs)
        generators = density_nodes[gen_idx].copy()

        n_gen = generators.shape[0]

        for iteration in range(n_iterations):
            # 分配每个密度节点到最近的 generator
            assignments = np.zeros(len(density_nodes), dtype=int)
            for i in range(len(density_nodes)):
                dists = np.linalg.norm(generators - density_nodes[i], axis=1)
                assignments[i] = np.argmin(dists)

            # 更新 generator 到密度加权质心
            new_gens = np.zeros_like(generators)
            for k in range(n_gen):
                mask = assignments == k
                if np.sum(mask) == 0:
                    new_gens[k] = generators[k]
                    continue
                w = np.abs(density_weights[mask])
                w_sum = np.sum(w)
                if w_sum > 1e-30:
                    new_gens[k] = np.sum(w[:, np.newaxis] * density_nodes[mask], axis=0) / w_sum
                else:
                    new_gens[k] = generators[k]

            generators = new_gens

        return generators

    def estimate_convergence_rate(
        self, n_points_list: List[int], error_list: List[float]
    ) -> float:
        """
        估计收敛速率:
          ε ~ N^{-r}
          r = -d(log ε) / d(log N)

        使用最小二乘拟合 log-log 图:
          log(ε) = -r log(N) + C
        """
        if len(n_points_list) < 2:
            return 0.0

        log_n = np.log(np.array(n_points_list, dtype=float))
        log_e = np.log(np.maximum(np.array(error_list, dtype=float), 1e-15))

        # 最小二乘: log_e = -r * log_n + C
        A = np.column_stack([log_n, np.ones(len(log_n))])
        result = np.linalg.lstsq(A, log_e, rcond=None)
        r = -result[0][0]

        return float(r)

    def should_refine(self, current_error: float) -> bool:
        """
        判断是否需要继续细化:
          - 当前误差 > 阈值
          - 当前 level < 最大 level
        """
        return (current_error > self.threshold and
                self.level < self.max_level)
