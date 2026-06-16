"""
multi_fidelity.py — 多保真度信息融合与自适应状态机
===================================================

本模块实现随机 PDE 的多保真度求解框架:
  1. 多保真度模型图: 不同精度模型的网络表示
  2. Dijkstra 最短路径: 最优保真度转换路径
  3. 状态机: 自适应加密/粗化决策

数学框架:
  多保真度模型族: {M₀, M₁, ..., M_L}
  M_ℓ: 第 ℓ 级保真度, 成本 c_ℓ, 精度 ε_ℓ
  最优估计: Q_MF = Q_L + Σ_{ℓ=0}^{L-1} (Q_ℓ - Q_{ℓ-1})
  方差: Var[Q_MF] = Σ Var[Q_ℓ - Q_{ℓ-1}]

映射种子项目:
  - 1276_IzzetYoung_ConsistentMIClientSimulator: 状态机 + Dijkstra + 主题图
  - 865_percolation_simulation: 渗流连通性判断模型间信息流通
"""

import numpy as np
from heapq import heappush, heappop


# ============================================================
# 第1部分: 多保真度模型定义
# ============================================================

class FidelityModel:
    """
    单保真度模型:
        Q_ℓ(ξ) = PDE 在第 ℓ 级网格/精度下的解

    属性:
        level: 保真度级别
        cost: 计算成本 (正比于 N_h^{d/2})
        mesh_size: 网格尺寸 h_ℓ = h₀ · 2^{-ℓ}
        accuracy: 理论精度 O(h_ℓ^p), p 为收敛阶
    """

    def __init__(self, level, base_mesh=8, convergence_order=2, base_cost=1.0):
        self.level = level
        self.mesh_size = base_mesh * (2 ** level)
        self.cost = base_cost * (2 ** (level * 2))  # 2D: O(h^{-2})
        self.accuracy_order = convergence_order
        self.h = 1.0 / self.mesh_size
        self.theoretical_accuracy = self.h ** convergence_order

    def __repr__(self):
        return (f"Fidelity(L={self.level}, h={self.h:.4f}, "
                f"cost={self.cost:.1f}, acc={self.theoretical_accuracy:.2e})")


class FidelityGraph:
    """
    多保真度模型图 (映射自 ConsistentMI 的 topic_graph):
        节点: 保真度模型 M_ℓ
        边权: 模型间差异度量 ||Q_ℓ - Q_k||

    用于:
      1. 最优保真度路径搜索 (Dijkstra)
      2. 信息融合权重计算
      3. 自适应加密决策
    """

    def __init__(self, n_levels=4, base_mesh=8, convergence_order=2):
        self.levels = []
        for l in range(n_levels):
            self.levels.append(FidelityModel(l, base_mesh, convergence_order))
        self.n_levels = n_levels
        # 边权矩阵 (模型间差异)
        self.edge_weights = np.zeros((n_levels, n_levels))
        self._initialize_edges()

    def _initialize_edges(self):
        """初始化边权: 相邻级别差异 ∝ h^p"""
        for i in range(self.n_levels):
            for j in range(self.n_levels):
                if i != j:
                    # 理论误差差
                    diff = abs(self.levels[i].theoretical_accuracy -
                               self.levels[j].theoretical_accuracy)
                    self.edge_weights[i, j] = diff + 1e-10
                    # 计算成本差
                    cost_ratio = max(self.levels[i].cost, self.levels[j].cost) / \
                                 min(self.levels[i].cost, self.levels[j].cost)
                    self.edge_weights[i, j] *= cost_ratio

    def dijkstra(self, source, target):
        """
        Dijkstra 最短路径 (映射自 ConsistentMI Client.dijkstra):
            在保真度图中搜索从 source 到 target 的最优转换路径。

        物理含义: 找到从低保真到目标保真度的最优"信息传递路径"。

        返回:
            path: 节点列表
            distance: 最短距离
        """
        n = self.n_levels
        dist = np.full(n, float('inf'))
        prev = np.full(n, -1, dtype=int)
        dist[source] = 0.0
        pq = [(0.0, source)]
        visited = set()

        while pq:
            d, u = heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                break
            for v in range(n):
                if v not in visited and v != u:
                    w = self.edge_weights[u, v]
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = u
                        heappush(pq, (dist[v], v))
        # 回溯路径
        path = []
        node = target
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, dist[target]


# ============================================================
# 第2部分: 自适应状态机
# (映射自 1276_IzzetYoung: Client state machine)
# ============================================================

class AdaptiveRefinementState:
    """
    自适应加密状态机:
        COARSE → EVALUATING → REFINING → VALIDATING → CONVERGED
                                         ↓
                                      COARSE (若不收敛)

    状态转移条件 (映射自 Client.update_state + select_action):
      - 误差 > tol → REFINING
      - 误差 < tol 且 成本合理 → CONVERGED
      - 成本超限 → COARSE (降级)
    """
    COARSE = 'COARSE'
    EVALUATING = 'EVALUATING'
    REFINING = 'REFINING'
    VALIDATING = 'VALIDATING'
    CONVERGED = 'CONVERGED'
    DEGRADED = 'DEGRADED'

    def __init__(self, error_tol=1e-3, max_level=5, budget=1000.0):
        self.state = self.COARSE
        self.current_level = 0
        self.error_tol = error_tol
        self.max_level = max_level
        self.budget = budget
        self.spent = 0.0
        self.history = []

    def update(self, current_error, model_cost):
        """
        状态转移 (映射自 Client.update_state + select_action):
            根据当前误差和成本决策下一步行动。

        返回:
            action: 'refine', 'keep', 'degrade', 'stop'
        """
        self.history.append({
            'state': self.state,
            'level': self.current_level,
            'error': current_error,
            'cost': model_cost
        })
        self.spent += model_cost

        # 预算检查
        if self.spent > self.budget:
            self.state = self.DEGRADED
            return 'stop'

        # 收敛检查
        if current_error < self.error_tol:
            self.state = self.CONVERGED
            return 'stop'

        # 可加密
        if self.current_level < self.max_level:
            if current_error > 10 * self.error_tol:
                self.state = self.REFINING
                self.current_level += 1
                return 'refine'
            elif current_error > self.error_tol:
                self.state = self.VALIDATING
                return 'keep'
            else:
                self.state = self.CONVERGED
                return 'stop'
        else:
            # 已达最大级别
            if current_error > self.error_tol:
                self.state = self.DEGRADED
                return 'stop'
            return 'stop'

    def get_receptivity(self):
        """
        自适应接受度 (映射自 Client.receptivity):
            基于历史误差变化趋势计算"接受新信息的程度"
        """
        if len(self.history) < 2:
            return 0.5
        errors = [h['error'] for h in self.history[-5:]]
        if len(errors) >= 2:
            improvement = (errors[0] - errors[-1]) / max(errors[0], 1e-15)
            return max(0.0, min(1.0, 0.5 + improvement))
        return 0.5


# ============================================================
# 第3部分: 多保真度融合估计
# ============================================================

def multi_fidelity_monte_carlo(model_evaluators, n_samples_per_level, seed=42):
    """
    多保真度 Monte Carlo (MFMC) 估计:
        Q_MF = Q_L^{MC} + Σ_{ℓ=0}^{L-1} β_ℓ (Q_ℓ^{MC} - Q_{ℓ-1}^{MC})

    最优样本分配:
        N_ℓ = N₀ · √(c₀/W_ℓ) · √(V_ℓ/W_ℓ) / Σ ...

    其中:
        c_ℓ: 第 ℓ 级单次成本
        V_ℓ: Var[Q_ℓ - Q_{ℓ-1}]
        W_ℓ: 权重因子

    参数:
        model_evaluators: list of callable, Q_ℓ(xi) → scalar
        n_samples_per_level: list of int, 每级样本数

    返回:
        estimate: MFMC 估计值
        variance: 估计方差
        cost_total: 总计算成本
    """
    rng = np.random.default_rng(seed)
    L = len(model_evaluators)
    if L == 0:
        return 0.0, 0.0, 0.0

    max_n = max(n_samples_per_level)
    xi_all = rng.standard_normal((max_n, 1))

    # 各级模型评估
    Q = []
    costs = []
    for l_idx in range(L):
        n_l = n_samples_per_level[l_idx]
        q_vals = np.array([model_evaluators[l_idx](xi_all[i]) for i in range(n_l)])
        Q.append(q_vals)
        costs.append(n_l * (2 ** (l_idx * 2)))

    # MFMC 估计
    if L == 1:
        estimate = np.mean(Q[0])
        variance = np.var(Q[0]) / max(1, len(Q[0]))
        return estimate, variance, costs[0]

    # Q_L 的 MC 估计
    estimate = np.mean(Q[-1])
    variance = np.var(Q[-1]) / max(1, len(Q[-1]))

    # 差分修正
    for l in range(L - 2, -1, -1):
        n_min = min(len(Q[l]), len(Q[l + 1]))
        if n_min > 1:
            diff = Q[l + 1][:n_min] - Q[l][:n_min]
            beta = 1.0  # 控制变量系数
            estimate -= beta * np.mean(diff)
            variance += beta ** 2 * np.var(diff) / n_min

    return estimate, max(variance, 1e-30), sum(costs)


# ============================================================
# 第4部分: 渗流连通性辅助决策
# (映射自 865_percolation_simulation)
# ============================================================

def information_percolation(adjacency_matrix, source_nodes, target_nodes, threshold=0.3):
    """
    信息渗流分析 (映射自 percolation_simulation):
        判断多保真度网络中信息能否从源节点渗流到目标节点。

    算法:
        1. 阈值化: A[i,j] > threshold → 连通
        2. BFS 洪水填充: 从源节点出发
        3. 检查是否到达目标节点

    返回:
        percolates: 是否渗流连通
        reachable: 从源可达的节点集合
    """
    n = len(adjacency_matrix)
    occupied = adjacency_matrix > threshold
    visited = set(source_nodes)
    queue = list(source_nodes)
    while queue:
        node = queue.pop(0)
        for neighbor in range(n):
            if occupied[node, neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    percolates = any(t in visited for t in target_nodes)
    return percolates, visited


def compute_fidelity_weights(error_estimates, costs):
    """
    最优保真度权重 (最小化方差给定预算):
        w_ℓ ∝ √(V_ℓ / c_ℓ) / Σ_k √(V_k / c_k)

    其中 V_ℓ 为误差估计, c_ℓ 为成本。
    """
    L = len(error_estimates)
    weights = np.zeros(L)
    for l in range(L):
        c = max(costs[l], 1e-10)
        weights[l] = np.sqrt(max(error_estimates[l], 0.0) / c)
    total = np.sum(weights)
    if total > 1e-30:
        weights /= total
    else:
        weights = np.ones(L) / L
    return weights
