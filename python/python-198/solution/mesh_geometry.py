"""
mesh_geometry.py
================
三角剖分几何与域分解模块（融合 1332_triangulation_boundary_edges + 379_fem_to_medit）

功能：
- 在单位圆盘上生成结构化三角剖分
- 提取边界边并排序成闭合环
- 计算三角形面积、重心坐标、网格质量指标
- 提供域分解所需的邻接图

数学公式：
- 三角形面积：A = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
- 内切圆半径：r_in = 4*A*R / (a+b+c) （这里简化为基于面积的高宽比）
"""

import numpy as np


def generate_disk_triangulation(n_r=8, n_theta=16):
    """
    在单位圆盘上生成结构化三角剖分。
    
    参数:
        n_r: 径向层数
        n_theta: 角度方向分段数
    
    返回:
        nodes: (N,2) 节点坐标
        elements: (M,3) 三角形单元（0-based索引）
        boundary_mask: (N,) 边界节点掩码
    """
    if n_r < 2 or n_theta < 3:
        raise ValueError("n_r must be >=2 and n_theta >=3")
    
    nodes = []
    boundary_mask = []
    
    # 中心点
    nodes.append([0.0, 0.0])
    boundary_mask.append(0)
    
    for i in range(1, n_r + 1):
        r = i / n_r
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])
            boundary_mask.append(1 if i == n_r else 0)
    
    nodes = np.array(nodes, dtype=float)
    boundary_mask = np.array(boundary_mask, dtype=int)
    
    elements = []
    # 第一层：中心到第一环
    for j in range(n_theta):
        j1 = j
        j2 = (j + 1) % n_theta
        elements.append([0, 1 + j1, 1 + j2])
    
    # 外层环
    for i in range(1, n_r):
        base_prev = 1 + (i - 1) * n_theta
        base_curr = 1 + i * n_theta
        for j in range(n_theta):
            j1 = j
            j2 = (j + 1) % n_theta
            # 每个四边形分成两个三角形
            elements.append([base_prev + j1, base_curr + j1, base_curr + j2])
            elements.append([base_prev + j1, base_curr + j2, base_prev + j2])
    
    elements = np.array(elements, dtype=int)
    return nodes, elements, boundary_mask


def triangle_area(nodes, elements):
    """
    计算每个三角形的面积。
    A_k = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    """
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    area = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return area


def compute_element_quality(nodes, elements):
    """
    计算三角形的质量指标 q = 4*sqrt(3)*A / (a^2+b^2+c^2)
    q ∈ (0,1]，1表示等边三角形。
    """
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    
    a2 = np.sum((p2 - p3) ** 2, axis=1)
    b2 = np.sum((p1 - p3) ** 2, axis=1)
    c2 = np.sum((p1 - p2) ** 2, axis=1)
    
    area = triangle_area(nodes, elements)
    quality = 4.0 * np.sqrt(3.0) * area / (a2 + b2 + c2 + 1e-15)
    quality = np.clip(quality, 0.0, 1.0)
    return quality


def extract_boundary_edges(elements):
    """
    提取三角剖分的边界边。
    基于 1332_triangulation_boundary_edges 的核心思想：
    内部边恰好属于两个三角形，边界边只属于一个。
    
    返回:
        boundary_edges: (B,2) 边界边节点索引，按顺序排列成环
    """
    edge_count = {}
    for tri in elements:
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1
    
    # 边界边只出现一次
    boundary = [e for e, c in edge_count.items() if c == 1]
    if not boundary:
        return np.zeros((0, 2), dtype=int)
    
    # 排序成闭合环
    boundary = list(boundary)
    ordered = [boundary.pop(0)]
    
    while boundary:
        last = ordered[-1][1]
        found = False
        for i, e in enumerate(boundary):
            if e[0] == last:
                ordered.append(e)
                boundary.pop(i)
                found = True
                break
            elif e[1] == last:
                ordered.append((e[1], e[0]))
                boundary.pop(i)
                found = True
                break
        if not found:
            # 无法连成单环，直接返回未排序的
            return np.array(ordered, dtype=int)
    
    return np.array(ordered, dtype=int)


def build_node_adjacency(elements, n_nodes):
    """
    构建节点邻接图（用于域分解和通信图）。
    """
    adj = [set() for _ in range(n_nodes)]
    for tri in elements:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = tri[i], tri[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj


def domain_decomposition(nodes, elements, n_parts):
    """
    使用递归坐标二分法(RCB)对三角剖分进行域分解。
    返回每个节点所属的分区编号。
    """
    n_nodes = nodes.shape[0]
    partition = np.zeros(n_nodes, dtype=int)
    
    def rcb(idx, part_id, remaining_parts):
        if remaining_parts <= 1 or len(idx) <= 1:
            partition[idx] = part_id
            return part_id + 1
        
        # 选择最长方向分割
        coords = nodes[idx]
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
        split_dim = int(np.argmax(ranges))
        median = np.median(coords[:, split_dim])
        
        left = idx[coords[:, split_dim] < median]
        right = idx[coords[:, split_dim] >= median]
        
        if len(left) == 0 or len(right) == 0:
            partition[idx] = part_id
            return part_id + 1
        
        next_id = rcb(left, part_id, remaining_parts // 2)
        next_id = rcb(right, next_id, remaining_parts - remaining_parts // 2)
        return next_id
    
    rcb(np.arange(n_nodes), 0, n_parts)
    return partition


def compute_interface_nodes(elements, partition):
    """
    计算域分解后的界面节点（属于不同分区的三角形共享的节点）。
    """
    n_nodes = len(partition)
    interface = np.zeros(n_nodes, dtype=int)
    
    for tri in elements:
        parts = {partition[tri[i]] for i in range(3)}
        if len(parts) > 1:
            for nid in tri:
                interface[nid] = 1
    return interface
