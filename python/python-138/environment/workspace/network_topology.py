"""
微反应器网络拓扑设计与流路分析 (基于有向图与树图算法)
=====================================================
设计微反应器网络 (Microreactor Network, MRN) 的流路拓扑，确保：
    1. 流体分配均匀（欧拉回路/路径存在性）
    2. 网络无循环死区（生成树覆盖）
    3. 压降最小化（最短路径树）

数学框架：
    将 MRN 建模为有向图 G=(V,E)，其中 V 为微通道节点，E 为通道。
    对分配网络（distributor），要求从入口到每个反应器模块的流路唯一，
    即底层无向图为树结构。

    Eulerian 路径定理：
        有向图存在欧拉回路当且仅当：
        (1) 图弱连通
        (2) 对每个顶点 v，indegree(v) = outdegree(v)

    生成树计数（Cayley 公式）：
        对完全图 K_n，生成树数量为 n^{n-2}。
        对一般图，使用 Kirchhoff 矩阵树定理：
            τ(G) = (1/n²) det( n I - J + L )
        其中 L 为 Laplacian 矩阵，J 为全 1 矩阵。
"""

import numpy as np
from typing import Tuple, List, Optional, Set


class MicroreactorNetworkTopology:
    """
    微反应器网络拓扑分析器。
    """

    def __init__(self, n_nodes: int = 8):
        if n_nodes < 2:
            raise ValueError("节点数至少为 2")
        self.n = n_nodes
        self.adj = np.zeros((n_nodes, n_nodes), dtype=int)
        self.edges = []  # 有向边列表 [(u,v), ...]

    def add_edge(self, u: int, v: int, capacity: float = 1.0):
        """添加有向边 u -> v。"""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("节点索引越界")
        if u == v:
            raise ValueError("不允许自环")
        self.adj[u, v] += 1
        self.edges.append((u, v))

    def compute_degrees(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算各节点的入度与出度。
        返回 (indegree, outdegree)。
        """
        indegree = np.sum(self.adj, axis=0)
        outdegree = np.sum(self.adj, axis=1)
        return indegree, outdegree

    def is_eulerian(self) -> Tuple[bool, bool]:
        """
        判断有向图是否为 Eulerian。

        返回:
            (has_circuit, has_path)
            has_circuit: 存在闭合欧拉回路
            has_path:    存在开欧拉路径
        """
        indegree, outdegree = self.compute_degrees()
        diff = indegree - outdegree
        n_plus = np.sum(diff == 1)
        n_minus = np.sum(diff == -1)
        n_zero = np.sum(diff == 0)

        has_circuit = n_plus == 0 and n_minus == 0 and n_zero == self.n
        has_path = (n_plus == 1 and n_minus == 1 and n_zero == self.n - 2) or has_circuit
        return has_circuit, has_path

    def find_eulerian_path(self) -> Optional[List[int]]:
        """
        Hierholzer 算法找欧拉路径/回路。
        返回节点访问序列，若不存在则返回 None。
        """
        has_circuit, has_path = self.is_eulerian()
        if not has_circuit and not has_path:
            return None

        # 复制邻接矩阵用于删除边
        adj_copy = self.adj.copy()
        path = []
        stack = []

        # 找起始点
        indegree, outdegree = self.compute_degrees()
        if has_circuit:
            start = 0
        else:
            start = int(np.where(outdegree - indegree == 1)[0][0])

        current = start
        while True:
            has_out = False
            for v in range(self.n):
                if adj_copy[current, v] > 0:
                    stack.append(current)
                    adj_copy[current, v] -= 1
                    current = v
                    has_out = True
                    break
            if not has_out:
                path.append(current)
                if len(stack) == 0:
                    break
                current = stack.pop()

        path.reverse()
        return path

    def pruefer_to_tree(self, code: np.ndarray) -> List[Tuple[int, int]]:
        """
        Pruefer 序列解码：将长度为 n-2 的 Pruefer 码转换为 n 个节点的树。

        算法：
            1. 初始化 degree[i] = 1 + code 中 i 的出现次数
            2. 对 code 中每个元素 p：
               找到最小 degree==1 的节点 v
               添加边 (v, p)
               degree[v] -= 1, degree[p] -= 1
            3. 最后连接剩余两个 degree==1 的节点
        """
        code = np.asarray(code, dtype=int).flatten()
        n = len(code) + 2
        if n != self.n:
            # 允许动态调整
            pass

        degree = np.ones(n, dtype=int)
        for p in code:
            degree[p] += 1

        edges = []
        for p in code:
            v = int(np.where(degree == 1)[0][0])
            edges.append((v, p))
            degree[v] -= 1
            degree[p] -= 1

        # 最后两个节点
        remaining = np.where(degree == 1)[0]
        if len(remaining) >= 2:
            edges.append((int(remaining[0]), int(remaining[1])))

        return edges

    def tree_to_pruefer(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        将树编码为 Pruefer 序列（n 个节点需要 n-2 个码）。
        """
        n = self.n
        # 构建邻接表
        adj_list = [set() for _ in range(n)]
        for u, v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)

        degree = np.array([len(adj_list[i]) for i in range(n)])
        code = []
        for _ in range(n - 2):
            leaf = int(np.where(degree == 1)[0][0])
            neighbor = adj_list[leaf].pop()
            adj_list[neighbor].remove(leaf)
            code.append(neighbor)
            degree[leaf] -= 1
            degree[neighbor] -= 1

        return np.array(code, dtype=int)

    def laplacian_matrix(self) -> np.ndarray:
        """
        构建无向图 Laplacian 矩阵：
            L = D - A
        其中 D 为度矩阵，A 为邻接矩阵（对称化）。
        """
        A_sym = ((self.adj + self.adj.T) > 0).astype(int)
        D = np.diag(np.sum(A_sym, axis=1))
        L = D - A_sym
        return L

    def count_spanning_trees(self) -> int:
        """
        Kirchhoff 矩阵树定理：
            τ(G) = det(L_{n-1})
        其中 L_{n-1} 为删除最后一行一列的 Laplacian。
        """
        L = self.laplacian_matrix()
        if L.shape[0] < 2:
            return 1
        L_reduced = L[:-1, :-1]
        try:
            tau = int(round(np.linalg.det(L_reduced)))
        except np.linalg.LinAlgError:
            tau = 0
        return max(tau, 0)

    def generate_optimal_distribution_tree(self, root: int = 0) -> List[Tuple[int, int]]:
        """
        生成以 root 为根的最短路径树（BFS 树），用于微反应器流体分配网络。
        """
        visited = [False] * self.n
        parent = [-1] * self.n
        queue = [root]
        visited[root] = True
        edges = []

        while queue:
            u = queue.pop(0)
            for v in range(self.n):
                if (self.adj[u, v] > 0 or self.adj[v, u] > 0) and not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    edges.append((u, v))
                    queue.append(v)

        # 对未访问节点随机连接
        for v in range(self.n):
            if not visited[v] and v != root:
                edges.append((root, v))
        return edges

    def network_uniformity_index(self) -> float:
        """
        网络均匀性指数：各节点度数的变异系数。
            η_net = 1 - σ_d / μ_d
        """
        A_sym = ((self.adj + self.adj.T) > 0).astype(int)
        degrees = np.sum(A_sym, axis=1)
        mu = np.mean(degrees)
        if mu < 1.0e-12:
            return 0.0
        sigma = np.std(degrees)
        return max(0.0, 1.0 - sigma / mu)
