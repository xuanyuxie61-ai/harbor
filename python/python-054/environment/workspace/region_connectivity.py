"""
region_connectivity.py
================================================================================
海洋区域连通性图网络表示与管理

融合项目：
    - 490_grf_io : GRF 图文件 I/O 与连通性表示

核心科学问题：
    将海洋划分为若干生物地球化学功能区域（如上升流区、涡旋、陆架区、
    深海平原），构建区域连通性图，分析碳和营养物质在不同区域间的
    传输路径与滞留时间。

科学背景：
    图 G = (V, E) 表示：
        V = {v₁, v₂, ..., v_n} : 海洋功能区节点
        E = {(vᵢ, vⱼ, wᵢⱼ)}    : 区域间连通边，权重 wᵢⱼ 表示交换通量
    
    节点属性：
        - xy[i] = (lonᵢ, latᵢ) : 区域中心坐标
        - area[i]               : 区域面积
        - DIC_inventory[i]      : 区域碳库存
    
    边权重：
        wᵢⱼ = kᵢⱼ · Aᵢⱼ · |Δρ| / Lᵢⱼ
    
    其中 kᵢⱼ 为交换系数，Aᵢⱼ 为接触面积，Δρ 为密度差，Lᵢⱼ 为特征距离。
    
    图的邻接表表示（CSR 风格）：
        edge_pointer[i] : 第 i 个节点的邻居在 edge_data 中的起始索引
        edge_data       : 邻居节点编号序列

================================================================================
"""

import numpy as np


class OceanRegionGraph:
    """
    海洋区域连通性图。
    
    支持：
        - 节点和边的增删
        - 邻接表构建
        - 基本图算法（连通分量、最短路径）
        - GRF 格式读写
    """
    
    def __init__(self):
        self.node_xy = []       # list of (x, y)
        self.node_attrs = {}    # dict of attr_name -> list of values
        self.edges = []         # list of (i, j, weight)
        self.n_nodes = 0
        self.n_edges = 0
    
    def add_node(self, x, y, **attrs):
        """添加节点并返回节点编号。"""
        node_id = self.n_nodes
        self.node_xy.append((x, y))
        for key, val in attrs.items():
            if key not in self.node_attrs:
                self.node_attrs[key] = [None] * self.n_nodes
            self.node_attrs[key].append(val)
        # 确保所有属性列表长度一致
        for key in self.node_attrs:
            while len(self.node_attrs[key]) < self.n_nodes + 1:
                self.node_attrs[key].append(None)
        self.n_nodes += 1
        return node_id
    
    def add_edge(self, i, j, weight=1.0):
        """添加无向边。"""
        if i < 0 or i >= self.n_nodes or j < 0 or j >= self.n_nodes:
            raise ValueError("节点编号越界")
        if i == j:
            return
        # 保证 i < j
        if i > j:
            i, j = j, i
        self.edges.append((i, j, float(weight)))
        self.n_edges += 1
    
    def build_adjacency(self):
        """
        构建 CSR 风格的邻接表。
        
        返回:
            edge_pointer : ndarray, shape (n_nodes+1,)
            edge_data    : ndarray, 邻居节点编号
            edge_weights : ndarray, 边权重
        """
        # 按源节点排序
        adj_dict = {i: [] for i in range(self.n_nodes)}
        for i, j, w in self.edges:
            adj_dict[i].append((j, w))
            adj_dict[j].append((i, w))
        
        edge_pointer = np.zeros(self.n_nodes + 1, dtype=int)
        edge_data = []
        edge_weights = []
        
        for i in range(self.n_nodes):
            neighbors = sorted(adj_dict[i], key=lambda t: t[0])
            edge_pointer[i] = len(edge_data)
            for j, w in neighbors:
                edge_data.append(j)
                edge_weights.append(w)
        
        edge_pointer[self.n_nodes] = len(edge_data)
        
        return edge_pointer, np.array(edge_data, dtype=int), np.array(edge_weights)
    
    def connected_components(self):
        """
        使用 DFS 找出连通分量。
        
        返回:
            list of list: 每个连通分量的节点列表
        """
        edge_pointer, edge_data, _ = self.build_adjacency()
        visited = [False] * self.n_nodes
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            start = edge_pointer[node]
            end = edge_pointer[node + 1]
            for idx in range(start, end):
                neighbor = edge_data[idx]
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(self.n_nodes):
            if not visited[i]:
                comp = []
                dfs(i, comp)
                components.append(comp)
        
        return components
    
    def dijkstra_shortest_path(self, source):
        """
        Dijkstra 最短路径算法。
        
        距离定义为边权重的倒数（权重越大，"距离"越短，表示传输越容易）。
        
        参数:
            source : int, 源节点
        
        返回:
            dist : ndarray, 到各节点的最短距离
            prev : ndarray, 前驱节点
        """
        edge_pointer, edge_data, edge_weights = self.build_adjacency()
        
        # 距离 = 1/weight
        dist = np.full(self.n_nodes, np.inf)
        prev = np.full(self.n_nodes, -1, dtype=int)
        dist[source] = 0.0
        
        unvisited = set(range(self.n_nodes))
        
        while unvisited:
            # 找距离最小的未访问节点
            u = min(unvisited, key=lambda x: dist[x])
            unvisited.remove(u)
            
            if dist[u] == np.inf:
                break
            
            start = edge_pointer[u]
            end = edge_pointer[u + 1]
            for idx in range(start, end):
                v = edge_data[idx]
                w = edge_weights[idx]
                if w > 1e-15:
                    d = dist[u] + 1.0 / w
                    if d < dist[v]:
                        dist[v] = d
                        prev[v] = u
        
        return dist, prev
    
    def to_grf_string(self):
        """
        将图序列化为 GRF 格式字符串。
        
        每行格式：
            node_index  x  y  neighbor1  neighbor2  ...
        """
        edge_pointer, edge_data, _ = self.build_adjacency()
        lines = []
        for i in range(self.n_nodes):
            x, y = self.node_xy[i]
            start = edge_pointer[i]
            end = edge_pointer[i + 1]
            neighbors = edge_data[start:end]
            line = f"{i+1}  {x:.6f}  {y:.6f}  " + "  ".join(str(n+1) for n in neighbors)
            lines.append(line)
        return "\n".join(lines)
    
    @classmethod
    def from_grf_string(cls, grf_str):
        """从 GRF 格式字符串解析图。"""
        graph = cls()
        lines = grf_str.strip().split("\n")
        
        # 第一遍：收集所有节点
        temp_nodes = []
        temp_edges = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            node_idx = int(parts[0]) - 1  # 转为 0-based
            x = float(parts[1])
            y = float(parts[2])
            neighbors = [int(p) - 1 for p in parts[3:]]
            temp_nodes.append((node_idx, x, y))
            temp_edges.append(neighbors)
        
        # 按索引排序添加节点
        temp_nodes.sort(key=lambda t: t[0])
        for idx, x, y in temp_nodes:
            graph.add_node(x, y)
        
        # 添加边
        for idx, neighbors in enumerate(temp_edges):
            for nb in neighbors:
                if idx < nb:  # 避免重复
                    graph.add_edge(idx, nb)
        
        return graph


def create_ocean_basin_graph(n_regions=12, basin_radius=500.0, seed=None):
    """
    创建一个示例海洋盆地连通性图。
    
    节点按功能区域分布：
        - 陆架区 (shelf)
        - 上升流区 (upwelling)
        - 涡旋区 (eddy)
        - 深海平原 (abyssal)
    
    参数:
        n_regions    : int, 区域数量
        basin_radius : float, 盆地半径 (km)
        seed         : int
    
    返回:
        OceanRegionGraph 实例
    """
    if seed is not None:
        np.random.seed(seed)
    
    graph = OceanRegionGraph()
    
    # 区域类型与参数
    region_types = ['shelf', 'upwelling', 'eddy', 'abyssal']
    type_colors = {'shelf': 0, 'upwelling': 1, 'eddy': 2, 'abyssal': 3}
    
    for i in range(n_regions):
        angle = 2.0 * np.pi * i / n_regions
        r = basin_radius * (0.3 + 0.7 * np.random.rand())
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        rtype = region_types[i % len(region_types)]
        area = 1000.0 + 5000.0 * np.random.rand()
        dic_inv = 1e6 + 5e6 * np.random.rand()
        
        graph.add_node(x, y,
                       region_type=rtype,
                       area=area,
                       DIC_inventory=dic_inv)
    
    # 添加边：相邻区域（按角度）+ 随机长程连接
    for i in range(n_regions):
        j = (i + 1) % n_regions
        dist = np.linalg.norm(np.array(graph.node_xy[i]) - np.array(graph.node_xy[j]))
        weight = max(0.1, 1000.0 / (dist + 1.0))
        graph.add_edge(i, j, weight)
        
        # 随机添加一条额外边
        if np.random.rand() < 0.3:
            k = np.random.randint(0, n_regions)
            if k != i and k != j:
                dist2 = np.linalg.norm(np.array(graph.node_xy[i]) - np.array(graph.node_xy[k]))
                weight2 = max(0.1, 500.0 / (dist2 + 1.0))
                graph.add_edge(i, k, weight2)
    
    return graph


def carbon_transport_path_analysis(graph, source_region, sink_region):
    """
    分析从源区到汇区的最优碳传输路径。
    
    参数:
        graph          : OceanRegionGraph
        source_region  : int, 源区编号
        sink_region    : int, 汇区编号
    
    返回:
        dict: 最短路径、路径长度、路径上的节点列表
    """
    dist, prev = graph.dijkstra_shortest_path(source_region)
    
    if dist[sink_region] == np.inf:
        return {
            'path_exists': False,
            'path_length': np.inf,
            'path_nodes': [],
        }
    
    # 回溯路径
    path = []
    node = sink_region
    while node != -1:
        path.append(node)
        node = prev[node]
    path = path[::-1]
    
    return {
        'path_exists': True,
        'path_length': dist[sink_region],
        'path_nodes': path,
    }
