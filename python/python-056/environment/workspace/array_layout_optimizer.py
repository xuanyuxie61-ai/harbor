"""
array_layout_optimizer.py
================================================================================
涡轮阵列布局与图网络优化模块 (来源于 1365_tsp_greedy + 484_graph_representation 项目)
================================================================================
本模块融合旅行商问题 (TSP) 贪心算法与图论网络表示，用于优化潮汐能
提取系统中涡轮阵列的布局和维护路径。将涡轮位置视为图中的节点，
通过邻接矩阵和贪心TSP路径规划最小化维护航行距离，同时保证阵列
的能量捕获效率。

核心公式:
    图邻接矩阵:
        A_{ij} = 1  若节点 i 与 j 之间存在边（距离小于阈值）
        A_{ij} = 0  否则

    TSP 贪心算法:
        从起点出发，每次选择最近的未访问节点，直到遍历所有节点。

    维护路径成本:
        C = Σ_{k=1}^{N} d_{p_k, p_{k+1}}
        其中 p 为访问顺序，p_{N+1} = p_1

    阵列拓扑度量:
        - 平均度: <k> = (2|E|)/|V|
        - 聚类系数: 衡量局部连接紧密程度
        - 直径: 图中任意两节点最短路径的最大值
"""

import numpy as np
from typing import Tuple, List


def gr_adjacency_matrix(
    node_num: int,
    node_coordinates: np.ndarray,
    edge_num: int,
    edge_nodes: np.ndarray,
) -> np.ndarray:
    """
    计算图的邻接矩阵。

    参数:
        node_num: 节点数量
        node_coordinates: 节点坐标 (2, node_num)
        edge_num: 边数量
        edge_nodes: 边定义 (2, edge_num)

    返回:
        adjacency_matrix: (node_num, node_num)
    """
    adj = np.zeros((node_num, node_num), dtype=int)
    for e in range(edge_num):
        n1 = int(edge_nodes[0, e]) - 1  # 1-based to 0-based
        n2 = int(edge_nodes[1, e]) - 1
        if 0 <= n1 < node_num and 0 <= n2 < node_num:
            adj[n1, n2] += 1
            adj[n2, n1] += 1
    return adj


def build_distance_graph(
    positions: np.ndarray,
    connect_threshold: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据涡轮位置构建距离图。

    参数:
        positions: 涡轮位置 (N, 2)
        connect_threshold: 连接阈值距离 (m)

    返回:
        (distance_matrix, adjacency_matrix)
    """
    n = positions.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    adj = (dist <= connect_threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return dist, adj


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    """
    计算 TSP 路径的总成本。

    参数:
        n: 城市数
        distance: 距离矩阵 (n, n)
        p: 访问顺序 (n,)

    返回:
        总路径长度
    """
    cost = 0.0
    for i in range(n):
        i1 = p[i]
        i2 = p[(i + 1) % n]
        cost += distance[i1, i2]
    return float(cost)


def path_greedy(n: int, distance: np.ndarray, start: int) -> np.ndarray:
    """
    从指定起点构造贪心 TSP 路径。

    算法:
        1. 初始化访问集合
        2. 当前节点 = start
        3. 重复: 选择距离当前节点最近的未访问节点

    参数:
        n: 城市数
        distance: 距离矩阵
        start: 起点索引 (0-based)

    返回:
        访问顺序 (n,)
    """
    p = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    p[0] = start
    visited[start] = True

    current = start
    for step in range(1, n):
        # 找到最近的未访问节点
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j] and distance[current, j] < min_dist:
                min_dist = distance[current, j]
                next_node = j
        if next_node < 0:
            # 回退到第一个未访问节点
            unvisited = np.where(~visited)[0]
            next_node = unvisited[0] if len(unvisited) > 0 else 0
        p[step] = next_node
        visited[next_node] = True
        current = next_node

    return p


def tsp_greedy_optimize(
    distance: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    对所有可能的起点执行贪心 TSP，选择最优路径。

    参数:
        distance: 距离矩阵 (n, n)

    返回:
        (best_path, best_cost)
    """
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
    """
    计算图拓扑度量。

    参数:
        adjacency: 邻接矩阵

    返回:
        包含 average_degree, clustering_coefficient, diameter 的字典
    """
    n = adjacency.shape[0]
    degrees = np.sum(adjacency, axis=1)
    avg_degree = float(np.mean(degrees))

    # 局部聚类系数
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

    # 直径 (Floyd-Warshall 近似)
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
    """
    优化潮汐涡轮阵列的维护航行路径。

    参数:
        positions: 涡轮位置 (N, 2)
        depot_position: 维护基地位置，默认域中心

    返回:
        (route, total_distance, metrics)
    """
    n = positions.shape[0]
    if depot_position is None:
        depot_position = np.mean(positions, axis=0)

    # 包含基地作为额外节点
    all_pos = np.vstack([depot_position.reshape(1, -1), positions])
    dist, adj = build_distance_graph(all_pos, connect_threshold=300.0)

    # TSP 贪心路径
    best_path, best_cost = tsp_greedy_optimize(dist)
    metrics = compute_graph_metrics(adj)
    metrics["total_route_distance"] = best_cost

    return best_path, best_cost, metrics
