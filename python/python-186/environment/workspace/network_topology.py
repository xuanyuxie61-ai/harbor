"""
network_topology.py
社交网络拓扑分析模块

基于以下种子项目融合:
- 441_floyd: Floyd-Warshall全源最短路径
- 796_neighbors_to_metis_graph: 网格邻居图转换
- 902_power_method: 幂法特征值计算

核心科学问题: 复杂社交网络的拓扑结构分析、社区检测与节点中心性量化。
"""

import numpy as np
from typing import Tuple, List, Optional


def construct_social_network(n_nodes: int,
                             community_structure: bool = True,
                             seed: int = 42) -> np.ndarray:
    """
    构建具有社区结构的社交网络邻接矩阵。

    使用随机块模型(Stochastic Block Model, SBM)的变体:
    网络包含 k 个社区，社区内部连接概率 p_in，社区间连接概率 p_out。

    数学模型:
        A_ij ~ Bernoulli(p_in)  若 i,j 属于同一社区
        A_ij ~ Bernoulli(p_out) 若 i,j 属于不同社区

    参数:
        n_nodes: 节点总数
        community_structure: 是否启用社区结构
        seed: 随机种子

    返回:
        adj: 对称的加权邻接矩阵 (n_nodes x n_nodes)
    """
    rng = np.random.default_rng(seed)

    if community_structure:
        k_communities = max(2, n_nodes // 20)
        p_in = 0.35
        p_out = 0.05
    else:
        k_communities = 1
        p_in = 0.15
        p_out = 0.15

    # 分配社区标签
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

    # 确保连通性: 若图不连通，添加最小生成树边
    if not is_connected(adj):
        adj = ensure_connected(adj, rng)

    return adj


def is_connected(adj: np.ndarray) -> bool:
    """BFS检查图连通性"""
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
    """添加边确保图连通"""
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
    """
    Floyd-Warshall全源最短路径算法。

    对于加权图 G=(V,E,W)，计算距离矩阵 D:
        D_{ij}^{(0)} = w_{ij}  若 (i,j) in E
        D_{ij}^{(0)} = inf     否则
        D_{ij}^{(k)} = min(D_{ij}^{(k-1)}, D_{ik}^{(k-1)} + D_{kj}^{(k-1)})

    时间复杂度: O(n^3)
    空间复杂度: O(n^2)

    参数:
        adj: 加权邻接矩阵

    返回:
        dist: 全源最短距离矩阵
    """
    n = adj.shape[0]
    INF = np.inf

    dist = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    # 初始化
    mask = adj > 0
    dist[mask] = adj[mask]

    # 动态规划递推
    for k in range(n):
        # 向量化优化
        dk = dist[:, k][:, np.newaxis] + dist[k, :][np.newaxis, :]
        dist = np.minimum(dist, dk)

    return dist


def network_efficiency(dist: np.ndarray) -> float:
    """
    计算网络全局效率 (Latora-Marchiori效率)。

    公式:
        E(G) = (1 / (n * (n-1))) * sum_{i != j} (1 / d_{ij})

    其中 d_{ij} 为节点 i,j 之间的最短路径距离。
    """
    n = dist.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / dist
    np.fill_diagonal(inv_dist, 0.0)
    efficiency = np.sum(inv_dist) / (n * (n - 1))
    return efficiency


def betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """
    计算节点介数中心性 (Brandes算法)。

    介数中心性定义:
        C_B(v) = sum_{s != v != t} (sigma_{st}(v) / sigma_{st})

    其中 sigma_{st} 为 s 到 t 的最短路径总数，
    sigma_{st}(v) 为经过 v 的最短路径数。
    """
    n = adj.shape[0]
    cb = np.zeros(n, dtype=np.float64)

    for s in range(n):
        # BFS for unweighted; Dijkstra-like for weighted
        # 使用简化版本: 基于邻接的BFS层次遍历
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

    # 归一化
    if n > 2:
        cb /= ((n - 1) * (n - 2))

    return cb


def power_method_eigenvector(adj: np.ndarray,
                             max_iter: int = 1000,
                             tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    幂法计算主特征值和主特征向量。

    用于PageRank-like的中心性分析。

    算法:
        y_{k+1} = A * y_k / ||A * y_k||
        lambda_{k+1} = y_{k+1}^T * A * y_{k+1}  (Rayleigh商)

    收敛判定:
        |lambda_{k+1} - lambda_k| < tol
        sin(theta) = sqrt(1 - (y_k^T * y_{k+1})^2) < tol

    参数:
        adj: 邻接矩阵
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        lambda_max: 主特征值
        y: 主特征向量 (归一化)
    """
    n = adj.shape[0]

    # 构造行随机矩阵 (PageRank style)
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # 处理悬空节点
    M = adj / row_sums[:, np.newaxis]

    # 添加teleportation
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

        # Rayleigh商
        lambda_new = float(y_new @ (P @ y_new))

        # 收敛检查
        diff_lambda = abs(lambda_new - lambda_old)
        cos_angle = np.clip(float(y @ y_new), -1.0, 1.0)
        sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))

        y = y_new
        lambda_old = lambda_new

        if diff_lambda < tol and sin_angle < tol:
            break

    return lambda_old, y


def clustering_coefficient(adj: np.ndarray) -> np.ndarray:
    """
    计算局部聚类系数 (Watts-Strogatz)。

    公式:
        C_i = (2 * E_i) / (k_i * (k_i - 1))

    其中 E_i 为节点 i 的邻居之间的实际边数，
    k_i 为节点 i 的度数。
    """
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
    """
    计算度分布 P(k)。

    返回:
        degrees: 各节点的度
        pk: 度为 k 的概率质量函数
    """
    degrees = np.sum(adj > 0, axis=1)
    max_deg = int(degrees.max())
    pk = np.zeros(max_deg + 1, dtype=np.float64)

    for d in degrees:
        pk[int(d)] += 1.0

    pk /= len(degrees)

    return degrees, pk
