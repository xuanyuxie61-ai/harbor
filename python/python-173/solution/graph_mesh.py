"""
网格图论分析模块

融合自:
- 955_quadrilateral_mesh_rcm: 网格的 Reverse Cuthill-McKee 重排序
- 481_graph_adj: 图的邻接结构、连通性分析、BFS/DFS
- 544_hits: HITS 算法用于网格单元重要性排序

核心概念:
1. 网格图的邻接矩阵 A，其中 A[i,j] = 1 当且仅当节点 i 和 j 共享一条边。
2. RCM 重排序通过最小化矩阵带宽来加速稀疏线性求解器。
3. HITS 算法通过 authority/hub 分数识别网格中的关键节点。
"""

import numpy as np
from collections import deque


def build_mesh_adjacency(n_nodes, triangles):
    """
    构建网格节点的邻接矩阵（稀疏表示）。
    
    Parameters
    ----------
    n_nodes : int
        节点总数
    triangles : ndarray, shape (n_tri, 3)
        三角形顶点索引
    
    Returns
    -------
    adj_list : list of list
        每个节点的邻居节点列表
    adj_row : ndarray
        CSR 格式的行指针
    adj : ndarray
        CSR 格式的列索引
    """
    adj_set = [set() for _ in range(n_nodes)]

    for tri in triangles:
        i, j, k = tri
        adj_set[i].add(j)
        adj_set[i].add(k)
        adj_set[j].add(i)
        adj_set[j].add(k)
        adj_set[k].add(i)
        adj_set[k].add(j)

    # 包含自身
    for i in range(n_nodes):
        adj_set[i].add(i)

    adj_list = [sorted(list(s)) for s in adj_set]

    # CSR 格式
    adj_row = np.zeros(n_nodes + 1, dtype=int)
    for i in range(n_nodes):
        adj_row[i + 1] = adj_row[i] + len(adj_list[i])

    adj = np.zeros(adj_row[-1], dtype=int)
    idx = 0
    for i in range(n_nodes):
        for j in adj_list[i]:
            adj[idx] = j
            idx += 1

    return adj_list, adj_row, adj


def adj_bandwidth(n_nodes, adj_row, adj):
    """
    计算邻接矩阵的带宽。
    
    带宽定义:
        B = max_{i,j: A[i,j] ≠ 0} |i - j| + 1
    
    Parameters
    ----------
    n_nodes : int
    adj_row : ndarray
    adj : ndarray
    
    Returns
    -------
    bandwidth : int
    """
    band_lo = 0
    band_hi = 0
    for i in range(n_nodes):
        for j_idx in range(adj_row[i], adj_row[i + 1]):
            col = adj[j_idx]
            band_lo = max(band_lo, i - col)
            band_hi = max(band_hi, col - i)
    return band_lo + 1 + band_hi


def rcm_reordering(n_nodes, adj_row, adj):
    """
    Reverse Cuthill-McKee (RCM) 重排序算法。
    
    算法步骤:
        1. 选择度数最小的节点作为根节点
        2. 执行 BFS，每层节点按度数升序排列
        3. 将得到的排序反转
    
    RCM 排序后，矩阵带宽通常显著降低。
    
    Parameters
    ----------
    n_nodes : int
    adj_row : ndarray
    adj : ndarray
    
    Returns
    -------
    perm : ndarray
        排列向量，新索引 = perm[旧索引]
    perm_inv : ndarray
        逆排列，旧索引 = perm_inv[新索引]
    """
    # 计算每个节点的度数
    degree = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        degree[i] = adj_row[i + 1] - adj_row[i] - 1  # 排除自身

    mask = np.ones(n_nodes, dtype=bool)
    perm = np.zeros(n_nodes, dtype=int)
    perm_inv = np.zeros(n_nodes, dtype=int)
    perm_idx = n_nodes - 1

    while perm_idx >= 0:
        # 找未处理节点中度数最小的
        min_deg = n_nodes + 1
        root = -1
        for i in range(n_nodes):
            if mask[i] and degree[i] < min_deg:
                min_deg = degree[i]
                root = i

        if root == -1:
            break

        # BFS
        queue = deque([root])
        mask[root] = False
        bfs_order = [root]

        while queue:
            node = queue.popleft()
            neighbors = []
            for j_idx in range(adj_row[node], adj_row[node + 1]):
                neighbor = adj[j_idx]
                if neighbor != node and mask[neighbor]:
                    neighbors.append(neighbor)

            # 按度数升序排列
            neighbors.sort(key=lambda x: degree[x])
            for neighbor in neighbors:
                if mask[neighbor]:
                    mask[neighbor] = False
                    queue.append(neighbor)
                    bfs_order.append(neighbor)

        # 反转 BFS 顺序并放入 perm
        for node in reversed(bfs_order):
            perm[node] = perm_idx
            perm_inv[perm_idx] = node
            perm_idx -= 1

    return perm, perm_inv


def apply_rcm_to_mesh(nodes, triangles, adj_row, adj):
    """
    对网格应用 RCM 重排序。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    adj_row : ndarray
    adj : ndarray
    
    Returns
    -------
    reordered_nodes : ndarray
    reordered_triangles : ndarray
    perm : ndarray
    bandwidth_before : int
    bandwidth_after : int
    """
    n_nodes = len(nodes)
    perm, perm_inv = rcm_reordering(n_nodes, adj_row, adj)

    bandwidth_before = adj_bandwidth(n_nodes, adj_row, adj)

    # 重排节点
    reordered_nodes = nodes[perm_inv]

    # 重排三角形中的节点索引
    reordered_triangles = perm[triangles]

    # 重新计算带宽
    _, new_adj_row, new_adj = build_mesh_adjacency(n_nodes, reordered_triangles)
    bandwidth_after = adj_bandwidth(n_nodes, new_adj_row, new_adj)

    return reordered_nodes, reordered_triangles, perm, bandwidth_before, bandwidth_after


def graph_distance_from_node(adj_list, source):
    """
    BFS 计算从源节点到所有其他节点的最短距离。
    
    对应原 graph_adj_distance_from_node.m 的核心功能。
    
    Parameters
    ----------
    adj_list : list of list
        邻接列表
    source : int
        源节点索引
    
    Returns
    -------
    distance : ndarray
        距离数组，不可达为 -1
    """
    n = len(adj_list)
    distance = np.full(n, -1, dtype=int)
    distance[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor != node and distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance


def graph_is_connected(adj_list):
    """
    判断图是否连通。
    
    对应原 graph_adj_is_nodewise_connected_breadth.m 的核心功能。
    
    Parameters
    ----------
    adj_list : list of list
    
    Returns
    -------
    connected : bool
    """
    n = len(adj_list)
    if n == 0:
        return True

    visited = np.zeros(n, dtype=bool)
    queue = deque([0])
    visited[0] = True
    count = 1

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor != node and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1

    return count == n


def hits_ranking(n_nodes, adj_row, adj, max_iter=50, tol=1e-6):
    """
    HITS (Hyperlink-Induced Topic Search) 算法用于网格节点重要性排序。
    
    对应原 hits_iteration.m 的核心功能。
    
    对于网格图，authority 分数高的节点是信息汇聚点，
    hub 分数高的节点是信息发散点。
    
    HITS 迭代:
        a^{(k+1)} = A^T h^{(k)}   (authority update)
        h^{(k+1)} = A a^{(k+1)}   (hub update)
    每次更新后归一化:
        a = a / ||a||_2
        h = h / ||h||_2
    
    Parameters
    ----------
    n_nodes : int
    adj_row : ndarray
    adj : ndarray
    max_iter : int
    tol : float
    
    Returns
    -------
    authority : ndarray
        每个节点的 authority 分数
    hub : ndarray
        每个节点的 hub 分数
    """
    a = np.ones(n_nodes) / np.sqrt(n_nodes)
    h = np.ones(n_nodes) / np.sqrt(n_nodes)

    # 构建邻接矩阵的转置作用
    def adj_multiply(v, transpose=False):
        result = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if i != j:
                    if transpose:
                        result[i] += v[j]
                    else:
                        result[i] += v[j]
        return result

    for _ in range(max_iter):
        # Authority update: a = A^T h
        a_new = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if j != i:
                    a_new[j] += h[i]

        norm_a = np.linalg.norm(a_new)
        if norm_a > 1e-14:
            a_new /= norm_a

        # Hub update: h = A a
        h_new = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if j != i:
                    h_new[i] += a_new[j]

        norm_h = np.linalg.norm(h_new)
        if norm_h > 1e-14:
            h_new /= norm_h

        if np.linalg.norm(a_new - a) < tol and np.linalg.norm(h_new - h) < tol:
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    return a, h


def compute_element_adjacency(n_triangles, triangles):
    """
    构建三角形单元之间的邻接关系（共享边）。
    
    Parameters
    ----------
    n_triangles : int
    triangles : ndarray, shape (n_tri, 3)
    
    Returns
    -------
    element_neighbors : list of list
        每个三角形的邻居三角形索引列表
    """
    edge_to_tri = {}
    for t in range(n_triangles):
        for e in range(3):
            i = triangles[t, e]
            j = triangles[t, (e + 1) % 3]
            edge = tuple(sorted([i, j]))
            if edge not in edge_to_tri:
                edge_to_tri[edge] = []
            edge_to_tri[edge].append(t)

    element_neighbors = [[] for _ in range(n_triangles)]
    for edge, tri_list in edge_to_tri.items():
        if len(tri_list) == 2:
            t1, t2 = tri_list
            element_neighbors[t1].append(t2)
            element_neighbors[t2].append(t1)

    return element_neighbors
