
import numpy as np
from typing import Tuple, List


def gr_adjacency_matrix(
    node_num: int,
    node_coordinates: np.ndarray,
    edge_num: int,
    edge_nodes: np.ndarray,
) -> np.ndarray:
    adj = np.zeros((node_num, node_num), dtype=int)
    for e in range(edge_num):
        n1 = int(edge_nodes[0, e]) - 1
        n2 = int(edge_nodes[1, e]) - 1
        if 0 <= n1 < node_num and 0 <= n2 < node_num:
            adj[n1, n2] += 1
            adj[n2, n1] += 1
    return adj


def build_distance_graph(
    positions: np.ndarray,
    connect_threshold: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = positions.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    adj = (dist <= connect_threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return dist, adj


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    cost = 0.0
    for i in range(n):
        i1 = p[i]
        i2 = p[(i + 1) % n]
        cost += distance[i1, i2]
    return float(cost)


def path_greedy(n: int, distance: np.ndarray, start: int) -> np.ndarray:
    p = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    p[0] = start
    visited[start] = True

    current = start
    for step in range(1, n):

        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j] and distance[current, j] < min_dist:
                min_dist = distance[current, j]
                next_node = j
        if next_node < 0:

            unvisited = np.where(~visited)[0]
            next_node = unvisited[0] if len(unvisited) > 0 else 0
        p[step] = next_node
        visited[next_node] = True
        current = next_node

    return p


def tsp_greedy_optimize(
    distance: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n = distance.shape[0]
    if n < 2:
        return np.array([0]), 0.0

    best_cost = np.inf
    best_path = np.arange(n)

    for start in range(n):
        p = path_greedy(n, distance, start)
        cost = path_cost(n, distance, p)
        if cost < best_cost:
            best_cost = cost
            best_path = p.copy()

    return best_path, best_cost


def compute_graph_metrics(adjacency: np.ndarray) -> dict:
    n = adjacency.shape[0]
    degrees = np.sum(adjacency, axis=1)
    avg_degree = float(np.mean(degrees))


    clustering = 0.0
    for i in range(n):
        ki = degrees[i]
        if ki < 2:
            continue
        neighbors = np.where(adjacency[i, :] > 0)[0]
        edges_between = 0
        for j in neighbors:
            for k in neighbors:
                if j < k and adjacency[j, k] > 0:
                    edges_between += 1
        clustering += 2.0 * edges_between / (ki * (ki - 1))
    clustering /= max(n, 1)


    dist = np.where(adjacency > 0, adjacency.astype(float), np.inf)
    np.fill_diagonal(dist, 0.0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    diameter = float(np.max(dist[np.isfinite(dist)])) if np.any(np.isfinite(dist)) else 0.0

    return {
        "average_degree": avg_degree,
        "clustering_coefficient": clustering,
        "diameter": diameter,
    }


def optimize_maintenance_route(
    positions: np.ndarray,
    depot_position: np.ndarray = None,
) -> Tuple[np.ndarray, float, dict]:
    n = positions.shape[0]
    if depot_position is None:
        depot_position = np.mean(positions, axis=0)


    all_pos = np.vstack([depot_position.reshape(1, -1), positions])
    dist, adj = build_distance_graph(all_pos, connect_threshold=300.0)


    best_path, best_cost = tsp_greedy_optimize(dist)
    metrics = compute_graph_metrics(adj)
    metrics["total_route_distance"] = best_cost

    return best_path, best_cost, metrics
