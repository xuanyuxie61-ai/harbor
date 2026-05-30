
import numpy as np
from typing import Tuple, List, Optional


def construct_social_network(n_nodes: int,
                             community_structure: bool = True,
                             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if community_structure:
        k_communities = max(2, n_nodes // 20)
        p_in = 0.35
        p_out = 0.05
    else:
        k_communities = 1
        p_in = 0.15
        p_out = 0.15


    community_ids = np.arange(n_nodes) % k_communities

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if community_ids[i] == community_ids[j]:
                prob = p_in
            else:
                prob = p_out

            if rng.random() < prob:
                weight = rng.uniform(0.1, 1.0)
                adj[i, j] = weight
                adj[j, i] = weight


    if not is_connected(adj):
        adj = ensure_connected(adj, rng)

    return adj


def is_connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    count = 1

    while queue:
        u = queue.pop(0)
        neighbors = np.where(adj[u, :] > 0)[0]
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)

    return count == n


def ensure_connected(adj: np.ndarray, rng) -> np.ndarray:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if not visited[start]:
            comp = []
            queue = [start]
            visited[start] = True
            while queue:
                u = queue.pop(0)
                comp.append(u)
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)

    for i in range(len(components) - 1):
        u = rng.choice(components[i])
        v = rng.choice(components[i + 1])
        adj[u, v] = 0.5
        adj[v, u] = 0.5

    return adj


def floyd_warshall(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    INF = np.inf

    dist = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)


    mask = adj > 0
    dist[mask] = adj[mask]


    for k in range(n):

        dk = dist[:, k][:, np.newaxis] + dist[k, :][np.newaxis, :]
        dist = np.minimum(dist, dk)

    return dist


def network_efficiency(dist: np.ndarray) -> float:
    n = dist.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / dist
    np.fill_diagonal(inv_dist, 0.0)
    efficiency = np.sum(inv_dist) / (n * (n - 1))
    return efficiency


def betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    cb = np.zeros(n, dtype=np.float64)

    for s in range(n):


        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        d = np.full(n, -1, dtype=np.int32)
        d[s] = 0
        Q = [s]

        while Q:
            v = Q.pop(0)
            S.append(v)
            neighbors = np.where(adj[v, :] > 0)[0]
            for w in neighbors:
                if d[w] < 0:
                    d[w] = d[v] + 1
                    Q.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = np.zeros(n, dtype=np.float64)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]


    if n > 2:
        cb /= ((n - 1) * (n - 2))

    return cb


def power_method_eigenvector(adj: np.ndarray,
                             max_iter: int = 1000,
                             tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    n = adj.shape[0]


    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    M = adj / row_sums[:, np.newaxis]


    alpha = 0.85
    P = alpha * M + (1 - alpha) / n * np.ones((n, n), dtype=np.float64)

    y = np.random.rand(n)
    y /= np.linalg.norm(y)

    lambda_old = 0.0

    for it in range(max_iter):
        z = P @ y
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-15:
            break
        y_new = z / z_norm


        lambda_new = float(y_new @ (P @ y_new))


        diff_lambda = abs(lambda_new - lambda_old)
        cos_angle = np.clip(float(y @ y_new), -1.0, 1.0)
        sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))

        y = y_new
        lambda_old = lambda_new

        if diff_lambda < tol and sin_angle < tol:
            break

    return lambda_old, y


def clustering_coefficient(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    C = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neighbors = np.where(adj[i, :] > 0)[0]
        k = len(neighbors)
        if k < 2:
            C[i] = 0.0
            continue

        e_count = 0
        for j_idx in range(k):
            for l_idx in range(j_idx + 1, k):
                u = neighbors[j_idx]
                v = neighbors[l_idx]
                if adj[u, v] > 0:
                    e_count += 1

        C[i] = (2.0 * e_count) / (k * (k - 1))

    return C


def degree_distribution(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    degrees = np.sum(adj > 0, axis=1)
    max_deg = int(degrees.max())
    pk = np.zeros(max_deg + 1, dtype=np.float64)

    for d in degrees:
        pk[int(d)] += 1.0

    pk /= len(degrees)

    return degrees, pk
