"""
graph_sparsity.py — 图结构稀疏性分析模块
=========================================
来源项目映射:
  - 076_bellman_ford → 单源最短路径算法 (处理负权边)

科学背景:
  许多稀疏优化问题具有内在图结构:
    (1) 图总变差 (Graph TV):  min ||x||_1 + λ ∑_{(i,j)∈E} |x_i - x_j|
    (2) 簇稀疏 (clustered sparsity): 支撑集形成连通子图
    (3) 层次稀疏: 系数按偏序关系依赖

  本模块用 Bellman-Ford 算法:
    - 计算稀疏支撑集的最短依赖路径
    - 识别支撑集的连通分量
    - 在负权图上检测稀疏结构中的负环
      (对应于目标函数的无界下降方向)

核心公式:
  Bellman-Ford 松弛:
    d[v] = min(d[v], d[u] + w(u,v))   ∀ (u,v) ∈ E
  负环检测:
    若第 n-1 轮后仍可松弛, 则存在负环.
  图 TV 正则化:
    ||x||_{GTV} = ∑_{(i,j)∈E} w_{ij} |x_i - x_j|
"""
import numpy as np
from collections import defaultdict, deque


# ----------------------------------------------------------------------
# Bellman-Ford (源自 076_bellman_ford)
# ----------------------------------------------------------------------
def bellman_ford(n_vertices, edges, source):
    """Bellman-Ford 单源最短路径 (支持负权).

    输入:
        n_vertices : 顶点数
        edges      : list of (u, v, w)  有向边
        source     : 源点索引
    返回:
        dist       : (n_vertices,) 最短距离
        prev       : (n_vertices,) 前驱节点
        neg_cycle  : bool 是否含负环
    """
    INF = float('inf')
    dist = np.full(n_vertices, INF)
    prev = np.full(n_vertices, -1, dtype=int)
    dist[source] = 0.0

    edges = np.asarray(edges)
    if edges.size == 0:
        return dist, prev, False

    for iteration in range(n_vertices):
        updated = False
        for (u, v, w) in edges:
            u, v = int(u), int(v)
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break
    # 负环检测: 第 n 轮仍可松弛
    neg_cycle = False
    for (u, v, w) in edges:
        u, v = int(u), int(v)
        if dist[u] + w < dist[v] - 1e-12:
            neg_cycle = True
            break
    return dist, prev, neg_cycle


def reconstruct_path(prev, target):
    """从 prev 数组回溯路径."""
    path = []
    cur = target
    visited = set()
    while cur >= 0 and cur not in visited:
        path.append(cur)
        visited.add(cur)
        cur = prev[cur]
    if cur == -1:
        return path[::-1]
    return None  # 含环


# ----------------------------------------------------------------------
# 支撑集图结构
# ----------------------------------------------------------------------
def build_sparsity_graph(indices, threshold=0.5):
    """根据多指标的邻接关系构建图.

    两个多指标 α, β 相邻当且仅当 ||α - β||_1 = 1.
    边权: 基于指标差值的某种度量 (如阶差).
    """
    M = len(indices)
    edges = []
    adj = defaultdict(list)
    for i in range(M):
        for j in range(i + 1, M):
            diff = np.sum(np.abs(indices[i] - indices[j]))
            if diff == 1:
                w = 1.0
                edges.append((i, j, w))
                edges.append((j, i, w))
                adj[i].append(j)
                adj[j].append(i)
    return edges, adj


def support_connected_components(support, adj):
    """求支撑集的连通分量."""
    visited = set()
    components = []
    for s in support:
        if s in visited:
            continue
        comp = set()
        queue = deque([s])
        while queue:
            v = queue.popleft()
            if v in visited or v not in support:
                continue
            visited.add(v)
            comp.add(v)
            for nb in adj.get(v, []):
                if nb in support and nb not in visited:
                    queue.append(nb)
        components.append(comp)
    return components


def graph_total_variation(x, adj, weights=None):
    """计算图总变差 ||x||_{GTV} = ∑_{(i,j)∈E} w_{ij} |x_i - x_j|."""
    gtv = 0.0
    for i, nbs in adj.items():
        for j in nbs:
            if j > i:
                w = 1.0 if weights is None else weights.get((i, j), 1.0)
                gtv += w * abs(x[i] - x[j])
    return float(gtv)


def shortest_dependency_path(support, adj, source=None):
    """在支撑集内计算依赖路径 (最短路径).

    若未指定 source, 选支撑集中系数最大的指标.
    """
    if not support:
        return []
    if source is None:
        source = max(support, key=lambda i: abs(i))
    n = max(max(adj.keys()) if adj else 0, max(support)) + 1
    # 支撑集内的边
    sub_edges = []
    for u in support:
        for v in adj.get(u, []):
            if v in support:
                sub_edges.append((u, v, 1.0))
    if not sub_edges:
        return [source]
    dist, prev, _ = bellman_ford(n, sub_edges, source)
    # 返回所有可达节点的 BFS 顺序
    reachable = [v for v in support if dist[v] < float('inf')]
    return sorted(reachable, key=lambda v: dist[v])


# ----------------------------------------------------------------------
# 层次稀疏性
# ----------------------------------------------------------------------
def hierarchical_ordering(indices):
    """按 |α|_1 对多指标分层.

    返回:
        levels : dict {level: [indices in that level]}
    """
    levels = defaultdict(list)
    for i, alpha in enumerate(indices):
        levels[int(np.sum(alpha))].append(i)
    return dict(levels)


def check_downward_closed(support, indices):
    """检查支撑集是否向下封闭 (hereditary).

    向下封闭: α ∈ supp 且 β ≤ α 分量 ⇒ β ∈ supp.
    这是 hierarchical basis 的重要性质.
    """
    for i in support:
        alpha = indices[i]
        # 枚举所有 β ≤ α
        for j in range(len(indices)):
            beta = indices[j]
            if np.all(beta <= alpha) and j not in support:
                return False, (i, j)
    return True, None
