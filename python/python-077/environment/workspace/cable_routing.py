
import numpy as np
from typing import List, Tuple, Dict, Optional


class CableRouter:

    def __init__(self, substation: Tuple[float, float],
                 turbines: List[Tuple[float, float]],
                 cable_cost_per_meter: float = 500.0,
                 max_cable_length: float = 2000.0):
        self.substation = np.array(substation, dtype=float)
        self.turbines = [np.array(t, dtype=float) for t in turbines]
        self.n_nodes = 1 + len(turbines)
        self.cable_cost_per_meter = cable_cost_per_meter
        self.max_cable_length = max_cable_length

    def _distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        p1 = self.substation if i == 0 else self.turbines[i - 1]
        p2 = self.substation if j == 0 else self.turbines[j - 1]
        return float(np.linalg.norm(p1 - p2))

    def build_graph(self, connectivity_radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
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

                    cost = d * self.cable_cost_per_meter
                    edge_list.append([j, i])
                    weight_list.append(cost)

        if not edge_list:
            raise ValueError("图中没有边，请增大连接半径")

        edges = np.array(edge_list, dtype=int).T
        weights = np.array(weight_list, dtype=float)
        return edges, weights

    def bellman_ford(self, source: int,
                     edges: np.ndarray,
                     weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v_num = self.n_nodes
        e_num = edges.shape[1]
        r8_big = 1.0e30

        dist = np.full(v_num, r8_big)
        dist[source] = 0.0
        pred = np.full(v_num, -1, dtype=int)


        for _ in range(v_num - 1):
            for j in range(e_num):
                u = edges[1, j]
                v = edges[0, j]
                t = dist[u] + weights[j]
                if t < dist[v]:
                    dist[v] = t
                    pred[v] = u


        for j in range(e_num):
            u = edges[1, j]
            v = edges[0, j]
            if dist[u] + weights[j] < dist[v] - 1e-12:
                raise ValueError("图中存在负权环")

        return dist, pred

    def find_paths(self, source: int = 0) -> Dict[int, List[int]]:
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
                paths[target] = []

        return paths

    def compute_total_cable_cost(self, paths: Dict[int, List[int]]) -> float:
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
        n = self.n_nodes
        in_mst = [False] * n
        parent = [-1] * n
        key = [float('inf')] * n
        key[0] = 0.0

        for _ in range(n):

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
