"""
概率路线图（PRM）与图排序模块
==============================
基于种子项目:
  - 544_hits       : HITS算法的权威值/枢纽值排序
  - 920_profile_data : 2D轮廓曲线作为演示轨迹目标

核心数学模型:
  1. 概率路线图（PRM）:
     (a) 在自由构型空间 C_free 中随机采样 N 个构型 q_i
     (b) 对每个 q_i，在半径 r 内连接 k 个最近邻
     (c) 边 (q_i, q_j) 存在当且仅当局部路径无碰撞

  2. HITS (Hyperlink-Induced Topic Search):
     给定二分图邻接矩阵 A（行=边/约束，列=节点/构型），
     迭代计算权威向量 a 和枢纽向量 h:
       a^{(k+1)} = A^T h^{(k)} / ||A^T h^{(k)}||_2
       h^{(k+1)} = A a^{(k+1)} / ||A a^{(k+1)}||_2
     收敛后，a_i 衡量节点 i 被多少高枢纽节点指向（权威），
     h_i 衡量节点 i 指向多少高权威节点（枢纽）。

  3. 在轨迹规划中:
     - 用HITS排序识别 roadmap 中的瓶颈节点（高权威+高枢纽）
     - 这些节点是连接不同自由空间区域的关键通道

  4. Profile数据作为2D末端轨迹目标:
     将2D面部轮廓映射到工作空间中的目标跟踪曲线，
     机械臂末端需沿此轮廓运动（焊接/抛光任务）。
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


# ---------------------------------------------------------------------------
# Profile数据（920_profile_data）
# ---------------------------------------------------------------------------

def profile_data() -> np.ndarray:
    r"""
    返回40个2D面部轮廓采样点，作为机械臂末端目标轨迹。
    原始数据来自MATLAB的 profile_data 函数。
    """
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
        # 扩展至40点：颈部延续
        [37.0, 2.90], [38.0, 2.70], [39.0, 2.50], [40.0, 2.30],
        [41.0, 2.10],
    ], dtype=float)


def scale_profile_to_workspace(profile: np.ndarray,
                               workspace_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    r"""
    将2D轮廓数据缩放到机械臂工作空间范围内。
    workspace_bounds = (min_bound, max_bound)，各为3维向量。
    这里将轮廓映射到XY平面，Z固定为 workspace 中值。
    """
    w_min, w_max = workspace_bounds
    w_min = np.asarray(w_min)
    w_max = np.asarray(w_max)
    # 缩放X
    p_min = profile.min(axis=0)
    p_max = profile.max(axis=0)
    scale = (w_max[:2] - w_min[:2]) / (p_max - p_min + 1e-14)
    scaled = w_min[:2] + (profile - p_min) * scale
    # Z坐标
    z_val = (w_min[2] + w_max[2]) * 0.5
    result = np.zeros((scaled.shape[0], 3), dtype=float)
    result[:, :2] = scaled
    result[:, 2] = z_val
    return result


# ---------------------------------------------------------------------------
# HITS算法（544_hits）
# ---------------------------------------------------------------------------

def hits_iteration(A: np.ndarray, max_iter: int = 100,
                   tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    HITS幂迭代算法。
    A: 邻接矩阵（m×n），这里取 roadmap 的边-节点关联矩阵。
    返回 (auth, hub)，均为归一化后的向量。
    """
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
    r"""
    HITS的SVD精确解：主右奇异向量=权威向量，主左奇异向量=枢纽向量。
    A = U Σ V^T
      auth = |V[:,0]|
      hub  = |U[:,0]|
    """
    A = np.asarray(A, dtype=float)
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        m, n = A.shape
        U = np.eye(m, 1) / np.sqrt(m)
        Vt = np.eye(1, n) / np.sqrt(n)
    auth = np.abs(Vt[0, :])
    hub = np.abs(U[:, 0])
    # 归一化
    if auth.sum() > 1e-14:
        auth = auth / auth.sum()
    if hub.sum() > 1e-14:
        hub = hub / hub.sum()
    return auth, hub


# ---------------------------------------------------------------------------
# 概率路线图（PRM）
# ---------------------------------------------------------------------------

class RoadmapGraph:
    r"""
    概率路线图图结构，用于高维构型空间的路径规划。
    """

    def __init__(self, n_dof: int = 7):
        self.n_dof = n_dof
        self.nodes = []          # 列表，每个元素是构型向量
        self.edges = []          # 列表，每个元素是 (i, j, cost)
        self.adj_list = {}       # 邻接表: node_idx -> [(neighbor_idx, cost)]

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
        r"""
        为每个节点连接k个最近邻（在半径范围内）。
        """
        n = len(self.nodes)
        if n < 2:
            return
        nodes_arr = np.array(self.nodes)
        for i in range(n):
            diffs = nodes_arr - nodes_arr[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            # 在radius内取最近k个
            mask = dists < radius
            valid_idx = np.where(mask)[0]
            if valid_idx.size == 0:
                continue
            sorted_idx = valid_idx[np.argsort(dists[valid_idx])]
            for j in sorted_idx[:k]:
                self.add_edge(i, j, dists[j])

    def hits_ranking(self) -> np.ndarray:
        r"""
        使用HITS算法对节点进行重要性排序。
        构造边-节点关联矩阵 A (|E| × |V|)。
        """
        n = len(self.nodes)
        if n == 0:
            return np.array([])
        # 简化：直接用邻接矩阵作为 A 的近似（方阵情况）
        # 构造无向图的关联矩阵：每行代表一条边，每列代表一个节点
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
        r"""
        Dijkstra最短路径算法。
        返回 (path_nodes, total_cost)。
        """
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
        # 回溯路径
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
    r"""
    构建PRM路线图。
    sampler: 返回 (n_samples, n_dof) 数组的采样函数
    collision_checker: q -> bool (True表示碰撞/无效)
    """
    rng = np.random.default_rng(seed)
    graph = RoadmapGraph()
    samples = sampler(n_samples)
    valid_indices = []
    for i in range(samples.shape[0]):
        if not collision_checker(samples[i]):
            idx = graph.add_node(samples[i])
            valid_indices.append(idx)
    graph.knn_edges(k=k, radius=radius)
    # 碰撞检测过滤边（简化：这里假设若端点有效则边有效）
    # 实际应做局部路径碰撞检测
    return graph
