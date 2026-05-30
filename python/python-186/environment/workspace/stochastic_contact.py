
import numpy as np
from typing import Tuple, List


def circle_positive_distance_monte_carlo(n_samples: int = 10000,
                                         seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)


    theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    x = np.abs(np.cos(theta))
    y = np.abs(np.sin(theta))


    n_pairs = min(n_samples // 2, 5000)
    indices = rng.choice(n_samples, size=(n_pairs, 2), replace=False)

    dx = x[indices[:, 0]] - x[indices[:, 1]]
    dy = y[indices[:, 0]] - y[indices[:, 1]]
    distances = np.sqrt(dx**2 + dy**2)

    mean_dist = float(np.mean(distances))
    var_dist = float(np.var(distances, ddof=1))

    return mean_dist, var_dist


def geometric_contact_rate(network_size: int,
                           activity_distribution: str = 'uniform',
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)


    r = np.sqrt(rng.uniform(0.0, 1.0, network_size))
    theta = rng.uniform(0.0, 2.0 * np.pi, network_size)
    pos_x = r * np.cos(theta)
    pos_y = r * np.sin(theta)


    if activity_distribution == 'uniform':
        activity = np.ones(network_size, dtype=np.float64)
    elif activity_distribution == 'power_law':

        gamma = 2.5
        u = rng.uniform(0.0, 1.0, network_size)
        activity = (1.0 - u)**(-1.0 / (gamma - 1.0))
        activity = np.clip(activity, 0.1, 10.0)
    elif activity_distribution == 'exponential':
        activity = rng.exponential(1.0, network_size)
    else:
        activity = np.ones(network_size, dtype=np.float64)


    d0 = 0.3
    contact_rates = np.zeros((network_size, network_size), dtype=np.float64)

    for i in range(network_size):
        for j in range(i + 1, network_size):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            d = np.sqrt(dx**2 + dy**2)


            rate = np.sqrt(activity[i] * activity[j]) * np.exp(-d / d0)
            contact_rates[i, j] = rate
            contact_rates[j, i] = rate

    return contact_rates


def magic4_test_matrix(n: int = 8) -> np.ndarray:
    if n % 4 != 0:
        n = (n // 4) * 4
        if n < 4:
            n = 4

    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            k = i * n + j + 1
            m1 = abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                A[i, j] = n * n + 1 - k
            else:
                A[i, j] = k

    return A


def percolation_threshold_estimate(network_size: int,
                                   n_realizations: int = 50,
                                   seed: int = 42) -> float:
    rng = np.random.default_rng(seed)


    p_base = 0.1
    adj = np.zeros((network_size, network_size), dtype=np.float64)
    for i in range(network_size):
        for j in range(i + 1, network_size):
            if rng.random() < p_base:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    thresholds = []

    for _ in range(n_realizations):

        edges = []
        for i in range(network_size):
            for j in range(i + 1, network_size):
                if adj[i, j] > 0:
                    edges.append((i, j))

        rng.shuffle(edges)
        n_edges = len(edges)
        adj_temp = adj.copy()

        for step, (i, j) in enumerate(edges):
            adj_temp[i, j] = 0.0
            adj_temp[j, i] = 0.0

            max_comp = largest_component_size(adj_temp)
            if max_comp < network_size / 2.0:
                p_remaining = 1.0 - (step + 1.0) / n_edges
                thresholds.append(p_remaining)
                break

        if len(thresholds) <= _:
            thresholds.append(0.0)

    return float(np.mean(thresholds))


def largest_component_size(adj: np.ndarray) -> int:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    max_size = 0

    for start in range(n):
        if not visited[start]:
            size = 0
            queue = [start]
            visited[start] = True
            while queue:
                u = queue.pop(0)
                size += 1
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            max_size = max(max_size, size)

    return max_size
