# -*- coding: utf-8 -*-
"""
topology_tracker.py
===================
界面拓扑变化检测与追踪模块，基于图论分析检测界面的分裂与合并事件。

融合原始项目:
  - 481_graph_adj: 图邻接矩阵、BFS 最短路径、连通分量分析

核心数学公式
------------
1. 界面连通分量提取:
   将零等值线附近的网格单元视为图节点，
   相邻单元间若法向量方向连续则建立边。
   图的连通分量个数 = 界面的拓扑分量个数。

2. 图邻接矩阵 A:
   A_{ij} = 1  if 节点 i 与 j 相邻
   A_{ij} = 0  otherwise
   连通分量可通过 A 的幂次检测:
   (A^k)_{ij} > 0  表示存在长度 ≤ k 的路径连接 i,j。

3. BFS 最短路径（源自 graph_adj_distance_from_node）:
   distance[node_k] = 0
   按层扩展邻居，distance 记录到源点的最短路径长度。
   时间复杂度 O(N + E)。

4. Euler 示性数（拓扑不变量）:
   χ = V - E + F
   对闭曲面，χ = 2 - 2g，其中 g 为亏格。
   对平面闭曲线，χ = 1 - n_holes。

5. 界面演化中的拓扑变化判据:
   - 分裂: 连通分量数 N_c(t+Δt) > N_c(t)
   - 合并: N_c(t+Δt) < N_c(t)
   - 消失: 某分量体积 V → 0
   - 生成: 新分量从内部成核
"""

import numpy as np


class TopologyTracker:
    """
    水平集界面拓扑追踪器，基于图的连通分量分析。
    """

    def __init__(self, levelset):
        self.ls = levelset
        self.history = {
            'num_components': [],
            'volumes': [],
            'euler_chars': []
        }

    def _build_interface_graph(self, band_width=None):
        """
        在零等值线附近带状区域内构建邻接图。
        节点：满足 |φ| < band_width 的网格单元。
        边：共享边的相邻单元。

        返回:
            adj : dict, 邻接表
            nodes : list, 节点列表 (i,j)
        """
        phi = self.ls.phi
        nx, ny = self.ls.nx, self.ls.ny
        if band_width is None:
            band_width = 2.0 * max(self.ls.dx, self.ls.dy)

        nodes = []
        node_index = {}
        idx = 0
        for i in range(nx):
            for j in range(ny):
                if np.abs(phi[i, j]) < band_width:
                    nodes.append((i, j))
                    node_index[(i, j)] = idx
                    idx += 1

        num_nodes = len(nodes)
        adj = {k: [] for k in range(num_nodes)}

        # 四邻域连接
        for k, (i, j) in enumerate(nodes):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in node_index:
                    adj[k].append(node_index[(ni, nj)])

        return adj, nodes

    def bfs_distance(self, adj, start_node, num_nodes):
        """
        从 start_node 出发进行 BFS，计算到所有可达节点的最短距离。
        源自 graph_adj_distance_from_node 的 BFS 思想。

        参数:
            adj       : dict, 邻接表
            start_node: int, 起始节点索引
            num_nodes : int, 总节点数
        返回:
            distance  : ndarray, 距离数组（不可达为 inf）
        """
        distance = np.full(num_nodes, np.inf)
        distance[start_node] = 0
        queue = [start_node]
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1
            d = distance[current]
            for neighbor in adj[current]:
                if distance[neighbor] == np.inf:
                    distance[neighbor] = d + 1
                    queue.append(neighbor)

        return distance

    def find_connected_components(self, band_width=None):
        """
        使用 BFS 查找界面带状区域内的连通分量。
        返回分量列表，每个分量为节点索引列表。
        """
        adj, nodes = self._build_interface_graph(band_width)
        num_nodes = len(nodes)
        if num_nodes == 0:
            return [], []

        visited = np.zeros(num_nodes, dtype=bool)
        components = []

        for start in range(num_nodes):
            if not visited[start]:
                dists = self.bfs_distance(adj, start, num_nodes)
                component = [i for i in range(num_nodes) if dists[i] < np.inf]
                for i in component:
                    visited[i] = True
                components.append(component)

        return components, nodes

    def compute_euler_characteristic_approx(self, band_width=None):
        """
        近似计算界面附近网格的 Euler 示性数。
        将网格单元视为面，共享边为边，角点为顶点。
        χ = V - E + F
        """
        phi = self.ls.phi
        nx, ny = self.ls.nx, self.ls.ny
        if band_width is None:
            band_width = 2.0 * max(self.ls.dx, self.ls.dy)

        # 统计满足条件的单元（面）
        face_count = 0
        edge_count = 0
        vertex_count = 0

        face_mask = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if np.abs(phi[i, j]) < band_width:
                    face_mask[i, j] = True
                    face_count += 1

        # 统计边（内部边只算一次）
        for i in range(nx - 1):
            for j in range(ny):
                if face_mask[i, j] and face_mask[i + 1, j]:
                    edge_count += 1
        for i in range(nx):
            for j in range(ny - 1):
                if face_mask[i, j] and face_mask[i, j + 1]:
                    edge_count += 1

        # 统计顶点
        for i in range(nx):
            for j in range(ny):
                if face_mask[i, j]:
                    vertex_count += 1

        chi = vertex_count - edge_count + face_count
        return chi

    def detect_topological_event(self, prev_components, curr_components):
        """
        检测拓扑变化事件。
        参数:
            prev_components : 前一时刻的连通分量数
            curr_components : 当前时刻的连通分量数
        返回:
            event : str, 事件类型描述
        """
        if curr_components > prev_components:
            return "SPLIT"
        elif curr_components < prev_components:
            return "MERGE"
        else:
            return "NO_CHANGE"

    def update_history(self):
        """
        记录当前时刻的拓扑统计信息。
        """
        components, nodes = self.find_connected_components()
        num_comp = len(components)
        self.history['num_components'].append(num_comp)

        # 计算每个分量的近似体积
        volumes = []
        dx, dy = self.ls.dx, self.ls.dy
        phi = self.ls.phi
        for comp in components:
            vol = 0.0
            for idx in comp:
                i, j = nodes[idx]
                if phi[i, j] < 0:
                    vol += dx * dy
            volumes.append(vol)
        self.history['volumes'].append(volumes)

        chi = self.compute_euler_characteristic_approx()
        self.history['euler_chars'].append(chi)

    def get_summary(self):
        """
        返回拓扑演化的摘要信息。
        """
        n_steps = len(self.history['num_components'])
        if n_steps == 0:
            return "No topology history recorded."

        summary = []
        summary.append(f"Topology evolution over {n_steps} time steps:")
        summary.append(f"  Initial components: {self.history['num_components'][0]}")
        summary.append(f"  Final components: {self.history['num_components'][-1]}")
        summary.append(f"  Max components: {max(self.history['num_components'])}")
        summary.append(f"  Min components: {min(self.history['num_components'])}")

        events = []
        for i in range(1, n_steps):
            event = self.detect_topological_event(
                self.history['num_components'][i - 1],
                self.history['num_components'][i]
            )
            if event != "NO_CHANGE":
                events.append(f"Step {i}: {event}")

        if events:
            summary.append("  Detected events:")
            for ev in events:
                summary.append(f"    {ev}")
        else:
            summary.append("  No topological events detected.")

        return "\n".join(summary)
