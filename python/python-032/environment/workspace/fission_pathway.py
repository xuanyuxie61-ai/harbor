"""
裂变路径网络分析与动力学图论模型
==================================
融合原始项目:
  - 286_digraph_arc: 有向图弧处理与欧拉回路检测

科学背景:
---------
核裂变过程可以抽象为在势能面上的"行走"过程，
从复合核（基态极小值）出发，翻越鞍点，最终到达断裂构型。
这一路径可以建模为有向图 G = (V, E)：

- 顶点 V_i: 势能面上的稳定构型（局部极小值）
- 有向边 E_{ij}: 从构型 i 到构型 j 的演化通道，
  权重 w_{ij} = exp( -ΔV_{ij} / T ) 为跃迁概率

裂变路径的搜索转化为图论问题：
1. 最短路径：寻找从基态到断裂点的最小能量路径 (Minimum Energy Path, MEP)
2. 欧拉回路：若存在闭合回路，表示裂变可逆（复合）
3. 连通性：判断是否存在从基态到所有断裂构型的路径

本模块实现裂变网络的图论分析，包括：
- 有向图表示与度分析
- 欧拉性质检测（判断是否存在哈密顿或欧拉路径）
- 基于 Dijkstra 的最小能量路径搜索
- 路径上的势能面插值
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class FissionPathwayGraph:
    """
    裂变路径网络图.
    
    节点：势能面局部极小值与鞍点
    边：有向跃迁通道，权重为跃迁概率
    """
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.adjacency = {i: [] for i in range(n_nodes)}
        self.edge_weights = {}
        self.node_positions = {}  # 节点在集体坐标空间中的位置
    
    def add_edge(self, i: int, j: int, weight: float):
        """添加有向边 i -> j."""
        if 0 <= i < self.n_nodes and 0 <= j < self.n_nodes:
            if j not in self.adjacency[i]:
                self.adjacency[i].append(j)
            self.edge_weights[(i, j)] = weight
    
    def set_node_position(self, node: int, position: np.ndarray):
        """设置节点在集体坐标空间中的位置."""
        self.node_positions[node] = position.copy()
    
    def out_degree(self, node: int) -> int:
        """出度."""
        return len(self.adjacency.get(node, []))
    
    def in_degree(self, node: int) -> int:
        """入度."""
        count = 0
        for i in range(self.n_nodes):
            if node in self.adjacency.get(i, []):
                count += 1
        return count
    
    def degree_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回所有节点的 (入度, 出度) 序列."""
        indeg = np.zeros(self.n_nodes, dtype=int)
        outdeg = np.zeros(self.n_nodes, dtype=int)
        for i in range(self.n_nodes):
            outdeg[i] = self.out_degree(i)
            indeg[i] = self.in_degree(i)
        return indeg, outdeg


def digraph_is_eulerian(graph: FissionPathwayGraph) -> int:
    """
    判断有向图是否为 Eulerian (改编自 digraph_arc_is_eulerian.m).
    
    返回:
        0: 非 Eulerian
        1: 存在开放 Euler 路径（起点终点不同）
        2: 存在闭合 Euler 回路
    
    在裂变物理中，闭合 Euler 回路意味着裂变-复合循环，
    这在高激发能下可能出现（多次穿越势垒）。
    """
    indeg, outdeg = graph.degree_sequence()
    n_plus = 0
    n_minus = 0
    
    for i in range(graph.n_nodes):
        if indeg[i] == outdeg[i]:
            continue
        elif n_plus == 0 and indeg[i] == outdeg[i] + 1:
            n_plus = 1
        elif n_minus == 0 and indeg[i] == outdeg[i] - 1:
            n_minus = 1
        else:
            return 0
    
    if n_plus == 0 and n_minus == 0:
        return 2
    elif n_plus == 1 and n_minus == 1:
        return 1
    else:
        return 0


def dijkstra_min_energy_path(
    graph: FissionPathwayGraph,
    source: int,
    target: int,
) -> Tuple[List[int], float]:
    """
    Dijkstra 算法搜索最小能量路径.
    
    边的代价定义为 -ln(w_{ij}) = ΔV_{ij}/T，
    因此最短路径对应最大跃迁概率路径。
    """
    n = graph.n_nodes
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    
    dist[source] = 0.0
    
    for _ in range(n):
        # 选择未访问的最小距离节点
        u = -1
        min_dist = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        
        if u == -1:
            break
        visited[u] = True
        
        for v in graph.adjacency.get(u, []):
            if not visited[v]:
                w = graph.edge_weights.get((u, v), 0.0)
                if w <= 0:
                    cost = np.inf
                else:
                    cost = -np.log(w)
                alt = dist[u] + cost
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    
    # 重构路径
    if dist[target] == np.inf:
        return [], np.inf
    
    path = []
    u = target
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return path, float(dist[target])


def build_fission_network_from_pes(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_grid_beta2: int = 15,
    n_grid_beta3: int = 15,
) -> FissionPathwayGraph:
    """
    从势能面采样构建裂变路径网络.
    
    在 (β₂, β₃) 网格上识别局部极小值作为节点，
    相邻节点间若能量单调则建立有向边。
    """
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    beta2_min, beta2_max = -0.3, 2.5
    beta3_min, beta3_max = -1.2, 1.2
    
    b2_grid = np.linspace(beta2_min, beta2_max, n_grid_beta2)
    b3_grid = np.linspace(beta3_min, beta3_max, n_grid_beta3)
    
    # 计算势能面
    V_grid = np.zeros((n_grid_beta2, n_grid_beta3))
    for i, b2 in enumerate(b2_grid):
        for j, b3 in enumerate(b3_grid):
            q = np.array([b2, b3, 0.0, 0.0, 0.0])
            V_grid[i, j] = potential_energy(q, mass_number, charge_number)
    
    # 识别局部极小值
    nodes = []
    node_indices = {}
    for i in range(1, n_grid_beta2 - 1):
        for j in range(1, n_grid_beta3 - 1):
            V_center = V_grid[i, j]
            neighbors = [
                V_grid[i - 1, j], V_grid[i + 1, j],
                V_grid[i, j - 1], V_grid[i, j + 1],
            ]
            if all(V_center <= v for v in neighbors):
                idx = len(nodes)
                nodes.append((i, j))
                node_indices[(i, j)] = idx
    
    graph = FissionPathwayGraph(len(nodes))
    
    for idx, (i, j) in enumerate(nodes):
        pos = np.array([b2_grid[i], b3_grid[j]])
        graph.set_node_position(idx, pos)
    
    # 建立边：相邻极小值之间，若能量单调则建立有向边
    for idx, (i, j) in enumerate(nodes):
        V_i = V_grid[i, j]
        # 检查四邻域
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in node_indices:
                nidx = node_indices[(ni, nj)]
                V_j = V_grid[ni, nj]
                # 双向都建立，权重由能量差决定
                delta_V = V_j - V_i
                w = np.exp(-delta_V / T)
                w = np.clip(w, 1e-10, 1.0)
                graph.add_edge(idx, nidx, w)
    
    return graph


def pathway_entropy(graph: FissionPathwayGraph, source: int) -> np.ndarray:
    """
    计算从源点出发的路径熵分布.
    
    S_i = -Σ_j p_{ij} ln p_{ij}
    其中 p_{ij} = w_{ij} / Σ_k w_{ik} 为归一化转移概率.
    """
    n = graph.n_nodes
    entropy = np.zeros(n)
    for i in range(n):
        neighbors = graph.adjacency.get(i, [])
        if not neighbors:
            entropy[i] = 0.0
            continue
        total_w = sum(graph.edge_weights.get((i, j), 0.0) for j in neighbors)
        if total_w <= 0:
            entropy[i] = 0.0
            continue
        S = 0.0
        for j in neighbors:
            w = graph.edge_weights.get((i, j), 0.0)
            if w > 0:
                p = w / total_w
                S -= p * np.log(p)
        entropy[i] = S
    return entropy
