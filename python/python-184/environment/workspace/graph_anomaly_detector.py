
import numpy as np
from typing import List, Tuple


class GraphAnomalyDetector:

    def __init__(self, epsilon: float | None = None, k_neighbors: int = 5):
        self.epsilon = epsilon
        self.k = k_neighbors
        self.adjacency: np.ndarray | None = None
        self.distances: np.ndarray | None = None

    def _build_knn_graph(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf
            neighbors = np.argpartition(dists, self.k)[:self.k]
            adj[i, neighbors] = 1
            adj[neighbors, i] = 1
        return adj

    def _build_epsilon_graph(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]

        diff = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        self.distances = dists
        adj = (dists < self.epsilon).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    def build_graph(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.epsilon is not None:
            self.adjacency = self._build_epsilon_graph(X)
        else:
            self.adjacency = self._build_knn_graph(X)
        return self.adjacency

    def connected_components(self) -> Tuple[int, np.ndarray]:
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
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        deg = np.sum(self.adjacency, axis=1)

        deg = np.where(deg == 0, 1, deg)
        P = self.adjacency / deg[:, None]

        pi = np.ones(n) / n
        v = np.ones(n) / n
        for _ in range(max_iter):
            pi_new = alpha * P.T @ pi + (1 - alpha) * v
            if np.linalg.norm(pi_new - pi, 1) < tol:
                break
            pi = pi_new


        score = 1.0 / (pi + 1e-12)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score

    def spectral_anomaly_score(self) -> np.ndarray:
        if self.adjacency is None:
            raise RuntimeError("Graph not built yet.")
        n = self.adjacency.shape[0]
        deg = np.sum(self.adjacency, axis=1)
        D = np.diag(deg)
        L = D - self.adjacency


        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt


        eigvals, eigvecs = np.linalg.eigh(L_sym)

        if n > 1:
            fiedler = eigvecs[:, 1]
        else:
            fiedler = np.zeros(n)

        score = np.abs(fiedler)
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score

    def detect(self, X: np.ndarray, method: str = "pagerank") -> np.ndarray:
        self.build_graph(X)
        if method == "pagerank":
            return self.pagerank_anomaly_score()
        elif method == "spectral":
            return self.spectral_anomaly_score()
        else:
            raise ValueError(f"Unknown method: {method}")
