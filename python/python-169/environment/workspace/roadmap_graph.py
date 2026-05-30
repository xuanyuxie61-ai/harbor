
import numpy as np
from typing import List, Tuple, Optional, Callable






def profile_data() -> np.ndarray:
    return np.array([
        [2.0,  2.50], [3.0,  3.10], [4.0,  3.50], [5.0,  3.80],
        [6.0,  4.00], [7.0,  4.10], [8.0,  4.30], [9.0,  4.50],
        [10.0, 4.80], [11.0, 5.20], [12.0, 5.60], [13.0, 6.10],
        [14.0, 6.50], [15.0, 6.80], [16.0, 7.00], [17.0, 7.10],
        [18.0, 7.20], [19.0, 7.30], [20.0, 7.20], [21.0, 7.10],
        [22.0, 7.00], [23.0, 6.80], [24.0, 6.50], [25.0, 6.10],
        [26.0, 5.70], [27.0, 5.30], [28.0, 4.90], [29.0, 4.60],
        [30.0, 4.30], [31.0, 4.10], [32.0, 3.90], [33.0, 3.70],
        [34.0, 3.50], [35.0, 3.30], [36.0, 3.10],

        [37.0, 2.90], [38.0, 2.70], [39.0, 2.50], [40.0, 2.30],
        [41.0, 2.10],
    ], dtype=float)


def scale_profile_to_workspace(profile: np.ndarray,
                               workspace_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    w_min, w_max = workspace_bounds
    w_min = np.asarray(w_min)
    w_max = np.asarray(w_max)

    p_min = profile.min(axis=0)
    p_max = profile.max(axis=0)
    scale = (w_max[:2] - w_min[:2]) / (p_max - p_min + 1e-14)
    scaled = w_min[:2] + (profile - p_min) * scale

    z_val = (w_min[2] + w_max[2]) * 0.5
    result = np.zeros((scaled.shape[0], 3), dtype=float)
    result[:, :2] = scaled
    result[:, 2] = z_val
    return result






def hits_iteration(A: np.ndarray, max_iter: int = 100,
                   tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    auth = np.ones(n, dtype=float) / np.sqrt(n)
    hub = np.ones(m, dtype=float) / np.sqrt(m)
    for _ in range(max_iter):
        auth_new = A.T @ hub
        hub_new = A @ auth_new
        norm_a = np.linalg.norm(auth_new)
        norm_h = np.linalg.norm(hub_new)
        if norm_a < 1e-14:
            norm_a = 1e-14
        if norm_h < 1e-14:
            norm_h = 1e-14
        auth_new = auth_new / norm_a
        hub_new = hub_new / norm_h
        if np.linalg.norm(auth_new - auth) < tol and np.linalg.norm(hub_new - hub) < tol:
            break
        auth = auth_new
        hub = hub_new
    return auth, hub


def hits_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        m, n = A.shape
        U = np.eye(m, 1) / np.sqrt(m)
        Vt = np.eye(1, n) / np.sqrt(n)
    auth = np.abs(Vt[0, :])
    hub = np.abs(U[:, 0])

    if auth.sum() > 1e-14:
        auth = auth / auth.sum()
    if hub.sum() > 1e-14:
        hub = hub / hub.sum()
    return auth, hub






class RoadmapGraph:

    def __init__(self, n_dof: int = 7):
        self.n_dof = n_dof
        self.nodes = []
        self.edges = []
        self.adj_list = {}

    def add_node(self, q: np.ndarray) -> int:
        idx = len(self.nodes)
        self.nodes.append(np.asarray(q, dtype=float).reshape(-1))
        self.adj_list[idx] = []
        return idx

    def add_edge(self, i: int, j: int, cost: float):
        if i == j:
            return
        self.edges.append((i, j, cost))
        self.adj_list[i].append((j, cost))
        self.adj_list[j].append((i, cost))

    def knn_edges(self, k: int = 5, radius: float = 2.0):
        n = len(self.nodes)
        if n < 2:
            return
        nodes_arr = np.array(self.nodes)
        for i in range(n):
            diffs = nodes_arr - nodes_arr[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf

            mask = dists < radius
            valid_idx = np.where(mask)[0]
            if valid_idx.size == 0:
                continue
            sorted_idx = valid_idx[np.argsort(dists[valid_idx])]
            for j in sorted_idx[:k]:
                self.add_edge(i, j, dists[j])

    def hits_ranking(self) -> np.ndarray:
        n = len(self.nodes)
        if n == 0:
            return np.array([])


        m = len(self.edges)
        if m == 0:
            return np.ones(n) / n
        A = np.zeros((m, n), dtype=float)
        for e_idx, (i, j, _) in enumerate(self.edges):
            A[e_idx, i] = 1.0
            A[e_idx, j] = 1.0
        auth, hub = hits_svd(A)
        return auth

    def dijkstra(self, start_idx: int, goal_idx: int) -> Tuple[List[int], float]:
        import heapq
        n = len(self.nodes)
        dist = {i: np.inf for i in range(n)}
        prev = {i: -1 for i in range(n)}
        dist[start_idx] = 0.0
        visited = set()
        pq = [(0.0, start_idx)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal_idx:
                break
            for v, cost in self.adj_list.get(u, []):
                if v not in visited:
                    alt = d + cost
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        heapq.heappush(pq, (alt, v))
        if dist[goal_idx] == np.inf:
            return [], np.inf

        path = []
        u = goal_idx
        while u != -1:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path, dist[goal_idx]


def build_prm_roadmap(sampler, collision_checker: Callable,
                      n_samples: int = 200, k: int = 5,
                      radius: float = 2.0, seed: int = 42) -> RoadmapGraph:
    rng = np.random.default_rng(seed)
    graph = RoadmapGraph()
    samples = sampler(n_samples)
    valid_indices = []
    for i in range(samples.shape[0]):
        if not collision_checker(samples[i]):
            idx = graph.add_node(samples[i])
            valid_indices.append(idx)
    graph.knn_edges(k=k, radius=radius)


    return graph
