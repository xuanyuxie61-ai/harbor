"""
Graph-Based Anomaly Detection via Adjacency Matrix Analysis
===========================================================
源自种子项目 481_graph_adj (Graph algorithms on adjacency matrices)。

在 time series 异常检测中，将时间序列嵌入为图结构是重要手段：
- 节点：时间窗口或状态向量
- 边：相似度超过阈值的连接

核心数学：
1. 相似度图构造（k-NN 或 epsilon-ball）：
    A_{ij} = 1  if  ||x_i - x_j||_2 < epsilon
    A_{ij} = 0  otherwise

2. 图拉普拉斯矩阵：
    L = D - A,   D_{ii} = sum_j A_{ij}
    归一化拉普拉斯：L_sym = I - D^{-1/2} A D^{-1/2}

3. 谱聚类与连通性：
    - 若图不连通（存在多个连通分量），不同分量间的点具有结构性差异
    - Fiedler 值（L 的第二小特征值）> 0 当且仅当图连通

4. 异常得分（基于 PageRank / 随机游走）：
    稳态分布 pi 满足  pi^T = pi^T P,   P = D^{-1} A (行随机)
    低 PageRank 的节点可能是异常（与主流模式连接弱）。

5. 最短路径异常度：
    节点到图中心的平均距离异常大的点。
"""

import numpy as np
from typing import List, Tuple


class GraphAnomalyDetector:
    """
    基于图邻接矩阵的 time series 异常检测器。
    """

    def __init__(self, epsilon: float | None = None, k_neighbors: int = 5):
        self.epsilon = epsilon
        self.k = k_neighbors
        self.adjacency: np.ndarray | None = None
        self.distances: np.ndarray | None = None

    def _build_knn_graph(self, X: np.ndarray) -> np.ndarray:
        """
        构建 k-NN 无向图邻接矩阵。
        """
        n = X.shape[0]
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf  # 排除自身
            neighbors = np.argpartition(dists, self.k)[:self.k]
            adj[i, neighbors] = 1
            adj[neighbors, i] = 1  # 无向
        return adj

    def _build_epsilon_graph(self, X: np.ndarray) -> np.ndarray:
        """
        构建 epsilon-ball 图邻接矩阵。
        """
        n = X.shape[0]
        # 计算成对距离矩阵
        diff = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        self.distances = dists
        adj = (dists < self.epsilon).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    def build_graph(self, X: np.ndarray) -> np.ndarray:
        """
        根据数据构造相似度图。
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.epsilon is not None:
            self.adjacency = self._build_epsilon_graph(X)
        else:
            self.adjacency = self._build_knn_graph(X)
        return self.adjacency

    def connected_components(self) -> Tuple[int, np.ndarray]:
        """
        BFS 计算连通分量数与每个节点的分量标签。
        """
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1, dtype=int)
        comp_id = 0

        for start in range(n):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            labels[start] = comp_id
            while queue:
                u = queue.pop(0)
                neighbors = np.where(self.adjacency[u] == 1)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        labels[v] = comp_id
                        queue.append(v)
            comp_id += 1
        return comp_id, labels

    def graph_distance_from_node(self, source: int) -> np.ndarray:
        """
        BFS 计算从源点到所有其他节点的最短路径长度（无权图）。
        """
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        dist = np.full(n, -1, dtype=int)
        dist[source] = 0
        queue = [source]
        while queue:
            u = queue.pop(0)
            neighbors = np.where(self.adjacency[u] == 1)[0]
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def pagerank_anomaly_score(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        PageRank 异常得分：得分越低越异常。
        pi^{(t+1)} = alpha * P^T pi^{(t)} + (1-alpha) * v
        其中 v = 1/n 为 teleportation 向量。
        """
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        deg = np.sum(self.adjacency, axis=1)
        # 处理孤立点
        deg = np.where(deg == 0, 1, deg)
        P = self.adjacency / deg[:, None]  # 行随机

        pi = np.ones(n) / n
        v = np.ones(n) / n
        for _ in range(max_iter):
            pi_new = alpha * P.T @ pi + (1 - alpha) * v
            if np.linalg.norm(pi_new - pi, 1) < tol:
                break
            pi = pi_new

        # 异常得分：PageRank 越低越异常
        score = 1.0 / (pi + 1e-12)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score

    def spectral_anomaly_score(self) -> np.ndarray:
        """
        基于图拉普拉斯特征向量的异常得分。
        使用归一化拉普拉斯的 Fiedler 向量（第二小特征值对应的特征向量）
        进行一维嵌入，远离零的点为异常。
        """
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        deg = np.sum(self.adjacency, axis=1)
        D = np.diag(deg)
        L = D - self.adjacency

        # 正则化避免奇异
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt

        # 计算特征值/特征向量（小图可用稠密求解）
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Fiedler 向量（第二小）
        if n > 1:
            fiedler = eigvecs[:, 1]
        else:
            fiedler = np.zeros(n)

        score = np.abs(fiedler)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score

    def detect(self, X: np.ndarray, method: str = "pagerank") -> np.ndarray:
        """
        综合异常检测入口。
        """
        self.build_graph(X)
        if method == "pagerank":
            return self.pagerank_anomaly_score()
        elif method == "spectral":
            return self.spectral_anomaly_score()
        else:
            raise ValueError(f"Unknown method: {method}")
