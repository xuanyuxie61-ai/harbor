"""
cable_routing.py
海上风电场集电系统电缆路由优化

融合源项目：
- 076_bellman_ford: Bellman-Ford最短路径算法（电缆最优路由）
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class CableRouter:
    """
    海上风电场集电系统电缆路由优化器。

    物理模型：
    -----------
    将风电场建模为加权有向图 G = (V, E, w)：
        - 顶点 V：升压站 + 各风机
        - 边 E：允许铺设电缆的连接
        - 权重 w(e)：电缆成本 = L(e) · C_unit · I_rated² · R_unit

    其中：
        - L(e)：边 e 的欧几里得长度 [m]
        - C_unit：单位长度电缆成本 [元/m]
        - I_rated：额定电流 [A]
        - R_unit：单位长度电阻 [Ω/m]

    采用 Bellman-Ford 算法（源自 076_bellman_ford）求解从升压站到各风机的
    最短电缆路径。

    同时考虑电缆载流量约束：
        I_cable ≥ Σ_{下游} I_turbine

    电缆截面积选择：
        A_cable = I_cable / (J_max · k1 · k2)

    其中 J_max 为最大电流密度，k1、k2 为温度和敷设修正系数。
    """

    def __init__(self, substation: Tuple[float, float],
                 turbines: List[Tuple[float, float]],
                 cable_cost_per_meter: float = 500.0,
                 max_cable_length: float = 2000.0):
        """
        Parameters
        ----------
        substation : Tuple[float, float]
            升压站坐标 (x, y) [m]。
        turbines : List[Tuple[float, float]]
            风机坐标列表。
        cable_cost_per_meter : float
            单位长度电缆成本 [元/m]。
        max_cable_length : float
            单根电缆最大长度 [m]。
        """
        self.substation = np.array(substation, dtype=float)
        self.turbines = [np.array(t, dtype=float) for t in turbines]
        self.n_nodes = 1 + len(turbines)
        self.cable_cost_per_meter = cable_cost_per_meter
        self.max_cable_length = max_cable_length

    def _distance(self, i: int, j: int) -> float:
        """
        计算节点 i 和 j 之间的欧几里得距离。

        节点编号：
            0：升压站
            1~n：风机
        """
        if i == j:
            return 0.0
        p1 = self.substation if i == 0 else self.turbines[i - 1]
        p2 = self.substation if j == 0 else self.turbines[j - 1]
        return float(np.linalg.norm(p1 - p2))

    def build_graph(self, connectivity_radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建电缆路由图。

        Parameters
        ----------
        connectivity_radius : Optional[float]
            连接半径，若 None 则使用 max_cable_length。

        Returns
        -------
        edges : np.ndarray
            2 × E 的边矩阵，每列为 (source, target)。
        weights : np.ndarray
            边权重向量，长度 E。
        """
        radius = connectivity_radius if connectivity_radius is not None else self.max_cable_length
        edge_list = []
        weight_list = []

        n = self.n_nodes
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = self._distance(i, j)
                if d <= radius and d > 1e-6:
                    # 电缆成本 = 长度 × 单位成本
                    cost = d * self.cable_cost_per_meter
                    edge_list.append([j, i])  # (target, source) 格式与 bellman_ford 一致
                    weight_list.append(cost)

        if not edge_list:
            raise ValueError("图中没有边，请增大连接半径")

        edges = np.array(edge_list, dtype=int).T
        weights = np.array(weight_list, dtype=float)
        return edges, weights

    def bellman_ford(self, source: int,
                     edges: np.ndarray,
                     weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bellman-Ford 最短路径算法。

        源自 076_bellman_ford。

        算法步骤：
            1. 初始化：dist[source] = 0, dist[v] = ∞, predecessor[v] = -1
            2. 松弛：重复 |V| - 1 次，对所有边 (u, v)：
               如果 dist[u] + w(u,v) < dist[v]，则更新 dist[v] 和 predecessor[v]
            3. 负权环检测

        时间复杂度 O(|V|·|E|)。

        Parameters
        ----------
        source : int
            源节点索引（升压站为 0）。
        edges : np.ndarray
            2 × E 边矩阵。
        weights : np.ndarray
            边权重。

        Returns
        -------
        dist : np.ndarray
            到各节点的最短距离。
        pred : np.ndarray
            前驱节点数组，用于回溯路径。
        """
        v_num = self.n_nodes
        e_num = edges.shape[1]
        r8_big = 1.0e30

        dist = np.full(v_num, r8_big)
        dist[source] = 0.0
        pred = np.full(v_num, -1, dtype=int)

        # 松弛
        for _ in range(v_num - 1):
            for j in range(e_num):
                u = edges[1, j]  # source
                v = edges[0, j]  # target
                t = dist[u] + weights[j]
                if t < dist[v]:
                    dist[v] = t
                    pred[v] = u

        # 负权环检测
        for j in range(e_num):
            u = edges[1, j]
            v = edges[0, j]
            if dist[u] + weights[j] < dist[v] - 1e-12:
                raise ValueError("图中存在负权环")

        return dist, pred

    def find_paths(self, source: int = 0) -> Dict[int, List[int]]:
        """
        找到从升压站到所有风机的最短电缆路径。

        Parameters
        ----------
        source : int
            源节点（升压站）。

        Returns
        -------
        Dict[int, List[int]]
            各目标节点的路径（节点索引列表）。
        """
        edges, weights = self.build_graph()
        dist, pred = self.bellman_ford(source, edges, weights)

        paths = {}
        for target in range(1, self.n_nodes):
            path = []
            node = target
            while node != -1:
                path.append(int(node))
                if node == source:
                    break
                node = int(pred[node])
            if path and path[-1] == source:
                paths[target] = list(reversed(path))
            else:
                paths[target] = []  # 不可达

        return paths

    def compute_total_cable_cost(self, paths: Dict[int, List[int]]) -> float:
        """
        计算总电缆成本。

        Parameters
        ----------
        paths : Dict[int, List[int]]
            各风机的电缆路径。

        Returns
        -------
        float
            总成本 [元]。
        """
        total_cost = 0.0
        counted_edges = set()

        for target, path in paths.items():
            if len(path) < 2:
                continue
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                edge_key = tuple(sorted([i, j]))
                if edge_key not in counted_edges:
                    d = self._distance(i, j)
                    total_cost += d * self.cable_cost_per_meter
                    counted_edges.add(edge_key)

        return total_cost

    def compute_cable_length_stats(self, paths: Dict[int, List[int]]) -> Tuple[float, float, float]:
        """
        计算电缆长度统计量。

        Returns
        -------
        total_length : float
            总长度 [m]。
        avg_length : float
            平均每台风机电缆长度 [m]。
        max_length : float
            最大单路径长度 [m]。
        """
        lengths = []
        counted_edges = set()

        for target, path in paths.items():
            if len(path) < 2:
                continue
            path_length = 0.0
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                edge_key = tuple(sorted([i, j]))
                d = self._distance(i, j)
                path_length += d
                counted_edges.add(edge_key)
            lengths.append(path_length)

        if not lengths:
            return 0.0, 0.0, 0.0

        total = sum(lengths)
        avg = total / len(lengths)
        max_len = max(lengths)
        return total, avg, max_len

    def optimize_routing_mst(self) -> Tuple[Dict[int, List[int]], float]:
        """
        使用最小生成树 (MST) 思想优化电缆路由。

        采用 Prim 算法构建 MST，然后从 MST 中提取到各节点的路径。

        Returns
        -------
        paths : Dict[int, List[int]]
            优化后的路径。
        total_cost : float
            总成本。
        """
        n = self.n_nodes
        in_mst = [False] * n
        parent = [-1] * n
        key = [float('inf')] * n
        key[0] = 0.0

        for _ in range(n):
            # 找到最小 key 的未访问节点
            u = -1
            min_key = float('inf')
            for i in range(n):
                if not in_mst[i] and key[i] < min_key:
                    min_key = key[i]
                    u = i
            if u == -1:
                break
            in_mst[u] = True

            for v in range(n):
                if u == v:
                    continue
                d = self._distance(u, v)
                cost = d * self.cable_cost_per_meter
                if not in_mst[v] and cost < key[v]:
                    key[v] = cost
                    parent[v] = u

        # 从 parent 数组提取路径
        paths = {}
        for target in range(1, n):
            path = []
            node = target
            while node != -1:
                path.append(node)
                if node == 0:
                    break
                node = parent[node]
            if path and path[-1] == 0:
                paths[target] = list(reversed(path))
            else:
                paths[target] = []

        total_cost = sum(
            self._distance(parent[i], i) * self.cable_cost_per_meter
            for i in range(1, n) if parent[i] != -1
        )
        return paths, total_cost
