
import numpy as np
from typing import Tuple, List, Optional, Set


class MicroreactorNetworkTopology:

    def __init__(self, n_nodes: int = 8):
        if n_nodes < 2:
            raise ValueError("节点数至少为 2")
        self.n = n_nodes
        self.adj = np.zeros((n_nodes, n_nodes), dtype=int)
        self.edges = []

    def add_edge(self, u: int, v: int, capacity: float = 1.0):
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("节点索引越界")
        if u == v:
            raise ValueError("不允许自环")
        self.adj[u, v] += 1
        self.edges.append((u, v))

    def compute_degrees(self) -> Tuple[np.ndarray, np.ndarray]:
        indegree = np.sum(self.adj, axis=0)
        outdegree = np.sum(self.adj, axis=1)
        return indegree, outdegree

    def is_eulerian(self) -> Tuple[bool, bool]:
        indegree, outdegree = self.compute_degrees()
        diff = indegree - outdegree
        n_plus = np.sum(diff == 1)
        n_minus = np.sum(diff == -1)
        n_zero = np.sum(diff == 0)

        has_circuit = n_plus == 0 and n_minus == 0 and n_zero == self.n
        has_path = (n_plus == 1 and n_minus == 1 and n_zero == self.n - 2) or has_circuit
        return has_circuit, has_path

    def find_eulerian_path(self) -> Optional[List[int]]:
        has_circuit, has_path = self.is_eulerian()
        if not has_circuit and not has_path:
            return None


        adj_copy = self.adj.copy()
        path = []
        stack = []


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
        code = np.asarray(code, dtype=int).flatten()
        n = len(code) + 2
        if n != self.n:

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


        remaining = np.where(degree == 1)[0]
        if len(remaining) >= 2:
            edges.append((int(remaining[0]), int(remaining[1])))

        return edges

    def tree_to_pruefer(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        n = self.n

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
        A_sym = ((self.adj + self.adj.T) > 0).astype(int)
        D = np.diag(np.sum(A_sym, axis=1))
        L = D - A_sym
        return L

    def count_spanning_trees(self) -> int:
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


        for v in range(self.n):
            if not visited[v] and v != root:
                edges.append((root, v))
        return edges

    def network_uniformity_index(self) -> float:
        A_sym = ((self.adj + self.adj.T) > 0).astype(int)
        degrees = np.sum(A_sym, axis=1)
        mu = np.mean(degrees)
        if mu < 1.0e-12:
            return 0.0
        sigma = np.std(degrees)
        return max(0.0, 1.0 - sigma / mu)
