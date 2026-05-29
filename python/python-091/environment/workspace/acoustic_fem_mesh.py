"""
声学有限元网格生成与RCM重排序模块

基于种子项目 1349_triangulation_rcm 的核心算法，为超声层析成像
提供优化的三角网格生成与稀疏矩阵带宽最小化。

核心算法:
- Reverse Cuthill-McKee (RCM) 算法：通过BFS层级遍历和反向编号
  最小化稀疏矩阵的半带宽，显著降低直接求解器的计算复杂度。
- 伪外围节点(Pseudo-peripheral node)选取：从最小度节点出发执行BFS，
  取最深层节点中度最小者。

科学背景:
在声学Helmholtz方程的有限元离散中，刚度矩阵 K 和质矩阵 M 均为大型
稀疏矩阵。RCM重排序可将半带宽从 O(N) 降低到 O(√N)，使得
Cholesky分解的浮点运算量从 O(N³) 降至 O(N^{1.5})。
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from collections import deque


def build_adjacency_structure(nodes: np.ndarray, triangles: np.ndarray) -> Dict[int, Set[int]]:
    """从三角网格构建节点邻接结构（图表示）。
    
    对于有限元网格，邻接图定义为：若两节点共享至少一个三角形单元，
    则它们之间存在一条边。
    
    参数:
        nodes: (N, 2) 节点坐标
        triangles: (M, 3) 三角形单元（节点索引）
    
    返回:
        adjacency: 邻接字典，adjacency[i] = 节点i的邻接节点集合
    """
    n_nodes = nodes.shape[0]
    adjacency = {i: set() for i in range(n_nodes)}
    
    for tri in triangles:
        i, j, k = tri
        adjacency[i].add(j)
        adjacency[i].add(k)
        adjacency[j].add(i)
        adjacency[j].add(k)
        adjacency[k].add(i)
        adjacency[k].add(j)
    
    return adjacency


def bfs_levels(adjacency: Dict[int, Set[int]], start: int) -> Tuple[Dict[int, int], int, int]:
    """从起点执行BFS，返回各节点层级、最深层和该层节点数。
    
    BFS层级定义了图的离心结构，是RCM算法的基础。
    
    参数:
        adjacency: 邻接字典
        start: 起始节点索引
    
    返回:
        levels: 节点到层级的映射
        max_level: 最大层级深度
        nodes_at_max: 最深层节点数
    """
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=bool)
    levels = {}
    
    queue = deque([start])
    visited[start] = True
    levels[start] = 0
    max_level = 0
    
    while queue:
        node = queue.popleft()
        current_level = levels[node]
        
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                levels[neighbor] = current_level + 1
                max_level = max(max_level, current_level + 1)
                queue.append(neighbor)
    
    nodes_at_max = sum(1 for v in levels.values() if v == max_level)
    return levels, max_level, nodes_at_max


def find_pseudo_peripheral(adjacency: Dict[int, Set[int]]) -> int:
    """寻找伪外围节点（pseudo-peripheral node）。
    
    算法步骤:
    1. 选取最小度节点作为起始点
    2. 执行BFS，取最深层中度最小的节点
    3. 重复步骤2，直到最深层节点数不再增加
    
    伪外围节点的离心率近似于图的直径，从该节点出发的BFS能
    产生最长的层级结构，从而最大化RCM的带宽缩减效果。
    """
    n_nodes = len(adjacency)
    
    # 步骤1：选取最小度节点
    degrees = {i: len(adjacency[i]) for i in range(n_nodes)}
    min_degree = min(degrees.values())
    candidates = [i for i, d in degrees.items() if d == min_degree]
    start = candidates[0]
    
    while True:
        levels, max_level, nodes_at_max = bfs_levels(adjacency, start)
        
        # 选取最深层中度最小的节点
        deepest_nodes = [i for i, lev in levels.items() if lev == max_level]
        deepest_degrees = {i: degrees[i] for i in deepest_nodes}
        min_deg = min(deepest_degrees.values())
        next_candidates = [i for i, d in deepest_degrees.items() if d == min_deg]
        next_start = next_candidates[0]
        
        # 从新起点执行BFS
        _, new_max_level, new_nodes_at_max = bfs_levels(adjacency, next_start)
        
        # 终止条件：最深层节点数不再增加
        if new_nodes_at_max <= nodes_at_max:
            return next_start
        
        start = next_start


def rcm_reorder(adjacency: Dict[int, Set[int]]) -> np.ndarray:
    """执行Reverse Cuthill-Mckee (RCM)重排序。
    
    算法步骤:
    1. 寻找伪外围节点 R
    2. 从 R 执行BFS，记录各节点层级
    3. 按层级分组，每层内部按度排序（升序）
    4. 将分组结果反转，得到新编号
    
    RCM的时间复杂度: O(N + E)，其中 E 为边数。
    对于2D有限元网格，E ≈ 3N。
    
    参数:
        adjacency: 邻接字典
    
    返回:
        reorder: 新到旧的映射数组，reorder[new_index] = old_index
    """
    n_nodes = len(adjacency)
    
    # 步骤1：寻找伪外围节点
    start = find_pseudo_peripheral(adjacency)
    
    # 步骤2：BFS层级遍历
    levels, max_level, _ = bfs_levels(adjacency, start)
    
    # 步骤3：按层级分组，每层按度排序
    level_groups = [[] for _ in range(max_level + 1)]
    degrees = {i: len(adjacency[i]) for i in range(n_nodes)}
    
    for node, lev in levels.items():
        level_groups[lev].append(node)
    
    # 每层内部按度升序排序
    for lev in range(max_level + 1):
        level_groups[lev].sort(key=lambda x: degrees[x])
    
    # 步骤4：展平并按RCM规则反转
    # Cuthill-McKee顺序：从伪外围节点开始，逐层展开
    cm_order = []
    for lev in range(max_level + 1):
        cm_order.extend(level_groups[lev])
    
    # Reverse：反转得到RCM顺序
    rcm_order = cm_order[::-1]
    
    reorder = np.array(rcm_order, dtype=int)
    return reorder


def apply_reorder_to_mesh(nodes: np.ndarray, triangles: np.ndarray,
                          reorder: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将RCM重排序应用到网格。
    
    参数:
        nodes: 原始节点坐标
        triangles: 原始三角形索引
        reorder: RCM新到旧映射
    
    返回:
        new_nodes: 重排序后的节点坐标
        new_triangles: 重排序后的三角形索引
        old_to_new: 旧到新的映射数组
    """
    n_nodes = nodes.shape[0]
    old_to_new = np.zeros(n_nodes, dtype=int)
    for new_idx, old_idx in enumerate(reorder):
        old_to_new[old_idx] = new_idx
    
    new_nodes = nodes[reorder]
    new_triangles = old_to_new[triangles]
    
    return new_nodes, new_triangles, old_to_new


def generate_acoustic_domain(nx: int = 41, ny: int = 41,
                             xlim: Tuple[float, float] = (0.0, 0.1),
                             ylim: Tuple[float, float] = (0.0, 0.1)) -> Tuple[np.ndarray, np.ndarray]:
    """生成规则矩形域的三角网格，用于声学模拟。
    
    物理参数:
    - 模拟域: 0.1m × 0.1m（典型医学超声成像区域）
    - 网格密度: nx × ny 个节点
    
    网格生成采用"左下-右上"对角线分割每个矩形单元，
    保证所有三角形均为锐角或直角三角形，避免病态单元。
    
    参数:
        nx: x方向节点数
        ny: y方向节点数
        xlim: x方向范围 (m)
        ylim: y方向范围 (m)
    
    返回:
        nodes: (N, 2) 节点坐标，单位 m
        triangles: (M, 3) 三角形索引
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"nx和ny必须至少为2，当前nx={nx}, ny={ny}")
    
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    
    # 生成节点
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([x[i], y[j]])
    nodes = np.array(nodes)
    
    # 生成三角形单元
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + i
            n4 = (j + 1) * nx + (i + 1)
            
            # 左下-右上对角线分割
            triangles.append([n1, n2, n4])
            triangles.append([n1, n4, n3])
    
    triangles = np.array(triangles)
    
    return nodes, triangles


def generate_optimized_acoustic_mesh(nx: int = 41, ny: int = 41,
                                     xlim: Tuple[float, float] = (0.0, 0.1),
                                     ylim: Tuple[float, float] = (0.0, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成声学模拟域并进行RCM优化。
    
    返回:
        nodes: 优化后的节点坐标
        triangles: 优化后的三角形索引
        reorder: RCM新到旧映射
        old_to_new: 旧到新映射
    """
    nodes, triangles = generate_acoustic_domain(nx, ny, xlim, ylim)
    adjacency = build_adjacency_structure(nodes, triangles)
    reorder = rcm_reorder(adjacency)
    new_nodes, new_triangles, old_to_new = apply_reorder_to_mesh(nodes, triangles, reorder)
    
    return new_nodes, new_triangles, reorder, old_to_new
