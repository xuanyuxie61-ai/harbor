"""
ocean_mesh.py
================================================================================
海洋有限元网格生成与拓扑分析

融合项目：
    - 953_quadrilateral_mesh : Q4 四边形网格生成、邻接分析、面积计算
    - 752_mesh_bandwidth    : 稀疏矩阵带宽分析

核心科学问题：
    为二维浅海碳输送模型生成结构化/非结构化四边形网格，并分析其拓扑
    结构（邻接关系、边界边、稀疏矩阵带宽），为后续有限元离散化做准备。

科学背景：
    二维海洋区域 Ω ⊂ ℝ² 被离散化为 N_e 个四边形单元、N_n 个节点。
    每个单元 e 有 4 个局部节点映射到全局节点编号：
        element_node[e, :] = [n₁, n₂, n₃, n₄]
    
    节点坐标为 xy[2, n]。单元面积通过拆分为两个三角形计算：
        A_e = A_tri(n₁,n₂,n₃) + A_tri(n₁,n₃,n₄)
        A_tri = 0.5·|x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
    
    刚度矩阵 K 的稀疏结构由节点连通性决定：
        K_{ij} ≠ 0  ⇔  节点 i 与 j 属于同一单元
    
    矩阵带宽：
        μ = max_{i<j, K_{ij}≠0} (j - i)   (上带宽)
        λ = max_{i>j, K_{ij}≠0} (i - j)   (下带宽)
        M = λ + 1 + μ                      (总带宽)
================================================================================
"""

import numpy as np


# =============================================================================
# 四边形单元基础运算
# =============================================================================

def triangle_area(x1, y1, x2, y2, x3, y3):
    """
    计算三角形有向面积（带符号）。
    
    A = 0.5 · |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
    """
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def quadrilateral_area(xy_quad):
    """
    计算四边形面积，拆分为两个三角形。
    
    参数:
        xy_quad : ndarray, shape (4, 2), 四节点坐标 (逆时针)
    """
    a1 = triangle_area(*xy_quad[0], *xy_quad[1], *xy_quad[2])
    a2 = triangle_area(*xy_quad[0], *xy_quad[2], *xy_quad[3])
    return a1 + a2


def q4_shape_functions(r, s):
    """
    四边形等参元形函数（在参考单元 [-1,1]×[-1,1] 上）。
    
        ψ₁ = (1-r)(1-s)/4
        ψ₂ = (1+r)(1-s)/4
        ψ₃ = (1+r)(1+s)/4
        ψ₄ = (1-r)(1+s)/4
    """
    psi = np.array([
        0.25 * (1.0 - r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 + s),
        0.25 * (1.0 - r) * (1.0 + s),
    ])
    dpsi_dr = np.array([
        -0.25 * (1.0 - s),  0.25 * (1.0 - s),
         0.25 * (1.0 + s), -0.25 * (1.0 + s),
    ])
    dpsi_ds = np.array([
        -0.25 * (1.0 - r), -0.25 * (1.0 + r),
         0.25 * (1.0 + r),  0.25 * (1.0 - r),
    ])
    return psi, dpsi_dr, dpsi_ds


def reference_to_physical_q4(xy_nodes, r, s):
    """
    将参考坐标 (r,s) 映射到物理坐标 (x,y)。
    
    x = Σ ψᵢ(r,s) · xᵢ
    y = Σ ψᵢ(r,s) · yᵢ
    """
    psi, _, _ = q4_shape_functions(r, s)
    x = np.dot(psi, xy_nodes[:, 0])
    y = np.dot(psi, xy_nodes[:, 1])
    return x, y


# =============================================================================
# 矩形海洋区域网格生成
# =============================================================================

def generate_ocean_rectangle_mesh(Lx, Ly, nx, ny, x0=0.0, y0=0.0):
    """
    在矩形区域 [x0, x0+Lx] × [y0, y0+Ly] 上生成均匀四边形网格。
    
    参数:
        Lx, Ly : float, 区域尺寸 (km)
        nx, ny : int, x, y 方向单元数
        x0, y0 : float, 左下角原点
    
    返回:
        node_xy      : ndarray, shape (node_num, 2), 节点坐标
        element_node : ndarray, shape (element_num, 4), 单元-节点连接 (0-based)
        nx, ny       : int, 网格分辨率
    """
    node_num_x = nx + 1
    node_num_y = ny + 1
    node_num = node_num_x * node_num_y
    element_num = nx * ny
    
    dx = Lx / nx
    dy = Ly / ny
    
    node_xy = np.zeros((node_num, 2), dtype=float)
    for j in range(node_num_y):
        for i in range(node_num_x):
            idx = j * node_num_x + i
            node_xy[idx, 0] = x0 + i * dx
            node_xy[idx, 1] = y0 + j * dy
    
    element_node = np.zeros((element_num, 4), dtype=int)
    for j in range(ny):
        for i in range(nx):
            e_idx = j * nx + i
            n1 = j * node_num_x + i
            n2 = n1 + 1
            n3 = n2 + node_num_x
            n4 = n1 + node_num_x
            element_node[e_idx, :] = [n1, n2, n3, n4]
    
    return node_xy, element_node, nx, ny


def generate_ocean_semicircle_mesh(R, nx, ny):
    """
    在半圆区域（海底半圆形海山/海沟）上生成非均匀四边形网格。
    使用径向-角度映射，在近边界加密。
    
    参数:
        R      : float, 半圆半径 (km)
        nx, ny : int, 径向和角度方向单元数
    """
    node_num = (nx + 1) * (ny + 1)
    element_num = nx * ny
    
    node_xy = np.zeros((node_num, 2), dtype=float)
    element_node = np.zeros((element_num, 4), dtype=int)
    
    # 角度范围 [0, π]
    for j in range(ny + 1):
        theta = np.pi * j / ny
        for i in range(nx + 1):
            # 径向网格加密：靠近边界更密
            t = i / nx
            r = R * (t**1.5)  # 非均匀分布
            idx = j * (nx + 1) + i
            node_xy[idx, 0] = r * np.cos(theta)
            node_xy[idx, 1] = r * np.sin(theta)
    
    for j in range(ny):
        for i in range(nx):
            e_idx = j * nx + i
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            element_node[e_idx, :] = [n1, n2, n3, n4]
    
    return node_xy, element_node


# =============================================================================
# 邻接分析与带宽计算
# =============================================================================

def compute_adjacency(element_node, node_num):
    """
    从单元-节点连接表构建节点邻接表（稀疏矩阵结构）。
    
    对每个单元，其所有节点对互为邻居：
        adjacency[i] = {j | 存在单元 e, 使得 i,j ∈ element_node[e, :]}
    
    参数:
        element_node : ndarray, shape (element_num, 4)
        node_num     : int, 节点总数
    
    返回:
        adjacency : list of sets, 每个节点的邻居集合
    """
    adjacency = [set() for _ in range(node_num)]
    element_num = element_node.shape[0]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for i in range(4):
            for j in range(4):
                if i != j:
                    adjacency[nodes[i]].add(nodes[j])
    
    # 包含自身（对角元）
    for i in range(node_num):
        adjacency[i].add(i)
    
    return adjacency


def mesh_bandwidth(element_node, node_num):
    """
    计算有限元网格对应的稀疏矩阵带宽。
    
    数学定义：
        λ = max_{e} max_{i<j} (global_node[j] - global_node[i])_negative
        μ = max_{e} max_{i<j} (global_node[j] - global_node[i])_positive
        M = λ + 1 + μ
    
    参数:
        element_node : ndarray, shape (element_num, 4)
        node_num     : int, 节点总数
    
    返回:
        dict: {'ml': 下带宽, 'mu': 上带宽, 'm': 总带宽}
    """
    ml = 0
    mu = 0
    element_num = element_node.shape[0]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for i in range(4):
            for j in range(4):
                gi = nodes[i]
                gj = nodes[j]
                if gi < gj:
                    mu = max(mu, gj - gi)
                elif gi > gj:
                    ml = max(ml, gi - gj)
    
    m = ml + 1 + mu
    return {'ml': ml, 'mu': mu, 'm': m}


def compute_element_areas(node_xy, element_node):
    """
    计算所有单元的面积。
    
    返回:
        areas : ndarray, shape (element_num,)
        total_area : float
    """
    element_num = element_node.shape[0]
    areas = np.zeros(element_num)
    for e in range(element_num):
        nodes = element_node[e, :]
        xy_quad = node_xy[nodes, :]
        areas[e] = quadrilateral_area(xy_quad)
    return areas, np.sum(areas)


def compute_boundary_edges(element_node):
    """
    识别网格边界边。
    
    算法：统计每条边被多少个单元共享。若只出现一次，则为边界边。
    
    四边形局部边定义：(0,1), (1,2), (2,3), (3,0)
    
    返回:
        boundary_edges : list of tuple, 全局节点对 (i,j) 且 i<j
        boundary_num   : int, 边界边数
    """
    edge_count = {}
    element_num = element_node.shape[0]
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    for e in range(element_num):
        nodes = element_node[e, :]
        for le in local_edges:
            n1, n2 = nodes[le[0]], nodes[le[1]]
            key = (min(n1, n2), max(n1, n2))
            edge_count[key] = edge_count.get(key, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges, len(boundary_edges)


def sample_q4_mesh(node_xy, element_node, n_samples):
    """
    在 Q4 网格上按面积加权随机采样 n_samples 个点。
    
    算法：
        1. 计算每个单元的面积 A_e
        2. 按面积比例随机选择单元
        3. 在选中单元内均匀采样
    
    参数:
        node_xy      : ndarray, shape (node_num, 2)
        element_node : ndarray, shape (element_num, 4)
        n_samples    : int, 采样点数
    
    返回:
        samples : ndarray, shape (n_samples, 2)
    """
    areas, _ = compute_element_areas(node_xy, element_node)
    element_num = element_node.shape[0]
    
    # 面积归一化概率
    probs = areas / np.sum(areas)
    chosen_elements = np.random.choice(element_num, size=n_samples, p=probs)
    
    samples = np.zeros((n_samples, 2))
    for k in range(n_samples):
        e = chosen_elements[k]
        nodes = element_node[e, :]
        xy_quad = node_xy[nodes, :]
        
        # 在四边形参考单元上均匀采样，再映射
        # 简单方法：拆分为两个三角形，按面积选择三角形，再在其中均匀采样
        a1 = triangle_area(*xy_quad[0], *xy_quad[1], *xy_quad[2])
        a2 = triangle_area(*xy_quad[0], *xy_quad[2], *xy_quad[3])
        if np.random.rand() < a1 / (a1 + a2):
            # 三角形 0-1-2
            p = xy_quad[0]
            q = xy_quad[1]
            r = xy_quad[2]
        else:
            # 三角形 0-2-3
            p = xy_quad[0]
            q = xy_quad[2]
            r = xy_quad[3]
        
        # 三角形内均匀采样
        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        samples[k, :] = p + u * (q - p) + v * (r - p)
    
    return samples
