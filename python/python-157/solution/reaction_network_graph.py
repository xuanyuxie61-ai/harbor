"""
reaction_network_graph.py
化学反应网络图分析与连通性模块
融合来源：489_grf_display（GRF 图文件读取与节点/边数据结构）

用于分析复杂燃烧反应网络中的物种连通性、最短路径与反应循环检测。
"""
import numpy as np
from combustion_utils import check_positive, check_nonnegative


class ReactionNetwork:
    r"""
    化学反应网络图表示。

    节点：化学物种（如 H2, O2, H, O, OH, H2O, HO2, H2O2）
    边：基元反应连接两个物种

    存储格式采用 CSR（Compressed Sparse Row）风格的邻接表，
    融合 489_grf_display 的节点-边数据结构设计。
    """

    def __init__(self):
        self.node_names = []
        self.node_map = {}
        self.edge_list = []  # list of (i, j, rate_coeff)
        self.adjacency_built = False
        self.edge_pointer = None
        self.edge_data = None
        self.edge_weights = None

    def add_node(self, name):
        r"""添加物种节点"""
        if name not in self.node_map:
            self.node_map[name] = len(self.node_names)
            self.node_names.append(name)
        return self.node_map[name]

    def add_edge(self, name_i, name_j, rate_coeff=1.0):
        r"""添加反应边"""
        i = self.add_node(name_i)
        j = self.add_node(name_j)
        self.edge_list.append((i, j, float(rate_coeff)))
        self.adjacency_built = False

    def build_adjacency(self):
        r"""
        构建 CSR 风格邻接结构。
        融合 489_grf_display 的 grf_data_read 思想。
        """
        n_nodes = len(self.node_names)
        n_edges = len(self.edge_list)

        # 按源节点排序
        sorted_edges = sorted(self.edge_list, key=lambda e: e[0])

        self.edge_pointer = np.zeros(n_nodes + 1, dtype=int)
        self.edge_data = np.zeros(n_edges, dtype=int)
        self.edge_weights = np.zeros(n_edges, dtype=float)

        for idx, (i, j, w) in enumerate(sorted_edges):
            self.edge_data[idx] = j
            self.edge_weights[idx] = w

        # 构建 pointer
        pos = 0
        self.edge_pointer[0] = 0
        for i in range(n_nodes):
            count = sum(1 for e in sorted_edges if e[0] == i)
            pos += count
            self.edge_pointer[i + 1] = pos

        self.adjacency_built = True

    def neighbors(self, i):
        r"""返回节点 i 的邻接节点列表"""
        if not self.adjacency_built:
            self.build_adjacency()
        start = self.edge_pointer[i]
        end = self.edge_pointer[i + 1]
        return self.edge_data[start:end]

    def degree(self, i):
        r"""节点度数"""
        if not self.adjacency_built:
            self.build_adjacency()
        return self.edge_pointer[i + 1] - self.edge_pointer[i]

    def bfs_shortest_path(self, start_name, target_name):
        r"""
        广度优先搜索最短路径（无权图）。
        返回路径上的节点名称列表。
        """
        if start_name not in self.node_map or target_name not in self.node_map:
            return None
        start = self.node_map[start_name]
        target = self.node_map[target_name]

        visited = [False] * len(self.node_names)
        parent = [-1] * len(self.node_names)
        queue = [start]
        visited[start] = True

        while queue:
            u = queue.pop(0)
            if u == target:
                # 回溯路径
                path = []
                cur = target
                while cur != -1:
                    path.append(self.node_names[cur])
                    cur = parent[cur]
                return path[::-1]
            for v in self.neighbors(u):
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        return None

    def find_cycles(self, max_length=6):
        r"""
        检测反应循环（长度不超过 max_length 的简单环）。
        返回循环列表，每个循环为节点索引元组。
        """
        n = len(self.node_names)
        cycles = []

        def dfs(u, start, depth, path, visited_set):
            if depth > max_length:
                return
            for v in self.neighbors(u):
                if v == start and depth >= 2:
                    cycle = tuple(path + [v])
                    # 规范化循环表示（从最小索引开始）
                    min_idx = cycle.index(min(cycle))
                    norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                    if norm not in cycles:
                        cycles.append(norm)
                elif v not in visited_set and v > start:
                    visited_set.add(v)
                    path.append(v)
                    dfs(v, start, depth + 1, path, visited_set)
                    path.pop()
                    visited_set.remove(v)

        for i in range(n):
            dfs(i, i, 0, [i], {i})

        return cycles

    def network_statistics(self):
        r"""
        计算网络统计指标:
            - 平均度数
            - 图密度
            - 聚类系数近似
        """
        if not self.adjacency_built:
            self.build_adjacency()
        n = len(self.node_names)
        if n == 0:
            return {}

        degrees = [self.degree(i) for i in range(n)]
        avg_degree = np.mean(degrees)
        density = len(self.edge_list) / (n * (n - 1.0)) if n > 1 else 0.0

        # 近似局部聚类系数
        clustering = []
        for i in range(n):
            neighbors_i = set(self.neighbors(i))
            ki = len(neighbors_i)
            if ki < 2:
                clustering.append(0.0)
                continue
            edges_between = 0
            for u in neighbors_i:
                for v in neighbors_i:
                    if u < v and v in set(self.neighbors(u)):
                        edges_between += 1
            clustering.append(2.0 * edges_between / (ki * (ki - 1.0)))
        avg_clustering = np.mean(clustering)

        return {
            'n_nodes': n,
            'n_edges': len(self.edge_list),
            'avg_degree': avg_degree,
            'density': density,
            'avg_clustering': avg_clustering
        }


def build_hydrogen_oxygen_network():
    r"""
    构建 H2-O2 燃烧基元反应网络（简化 8 物种模型）。

    物种: H2, O2, H, O, OH, H2O, HO2, H2O2
    主要反应路径来自典型链式反应机理。
    """
    net = ReactionNetwork()

    # 链引发
    net.add_edge("H2", "H", 1.0e-3)
    net.add_edge("O2", "O", 5.0e-4)

    # 链分支
    net.add_edge("H", "O2", 2.0e10)
    net.add_edge("O", "H2", 5.0e4)
    net.add_edge("OH", "H2", 1.0e8)

    # 链传递
    net.add_edge("H", "OH", 1.0e10)
    net.add_edge("O", "OH", 3.0e9)
    net.add_edge("OH", "H2O", 5.0e9)

    # 链终止
    net.add_edge("H", "HO2", 1.0e10)
    net.add_edge("O", "HO2", 5.0e9)
    net.add_edge("OH", "HO2", 2.0e10)

    # 过氧化物路径
    net.add_edge("HO2", "H2O2", 1.0e6)
    net.add_edge("H2O2", "OH", 2.0e7)

    net.build_adjacency()
    return net
