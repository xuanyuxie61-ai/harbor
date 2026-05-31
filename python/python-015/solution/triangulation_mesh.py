"""
triangulation_mesh.py
Fermi面三角剖分与网格拓扑分析

凝聚态物理应用：
在能带理论中，Fermi面是E(k) = E_F的等能面。
通过将Fermi面三角剖分为三角形网格，可以：
1. 计算Fermi面面积：A_FS = sum_i A_i
2. 分析拓扑连通性（是否包围Weyl节点）
3. 计算Berry相位沿Fermi面边界的贡献

三角剖分使用Delaunay三角化，确保最小角最大化。

核心公式：
三角形面积（2D投影）：
    A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|

基于种子项目1330_triangulation的多个核心函数：
- triangulation_order3_boundary_node（边界节点检测）
- triangulation_order3_adj_count（邻接关系计数）
- triangle_area_2d（三角形面积计算）
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, List, Optional


def delaunay_triangulate_2d(points: np.ndarray) -> np.ndarray:
    """
    对二维点集进行Delaunay三角剖分
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
    
    Returns
    -------
    triangles : np.ndarray, shape (M, 3)
        每个三角形由三个顶点的索引组成
    """
    if points.shape[1] != 2:
        raise ValueError("点必须是二维的")
    
    if len(points) < 3:
        return np.zeros((0, 3), dtype=int)
    
    tri = Delaunay(points)
    return tri.simplices.astype(int)


def triangle_area_2d(t: np.ndarray) -> float:
    """
    计算二维三角形面积
    
    基于种子项目1330_triangulation中的triangle_area_2d。
    
    公式：
        A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    
    Parameters
    ----------
    t : np.ndarray, shape (2, 3) 或 (3, 2)
        三角形顶点坐标
    
    Returns
    -------
    area : float
    """
    t = np.asarray(t, dtype=float)
    
    if t.shape == (2, 3):
        x1, x2, x3 = t[0, 0], t[0, 1], t[0, 2]
        y1, y2, y3 = t[1, 0], t[1, 1], t[1, 2]
    elif t.shape == (3, 2):
        x1, x2, x3 = t[0, 0], t[1, 0], t[2, 0]
        y1, y2, y3 = t[0, 1], t[1, 1], t[2, 1]
    else:
        raise ValueError(f"t形状不支持: {t.shape}")
    
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def triangulation_boundary_nodes(node_num: int, triangles: np.ndarray) -> np.ndarray:
    """
    检测三角剖分中的边界节点
    
    基于种子项目1330_triangulation中的triangulation_order3_boundary_node。
    
    原理：
    - 内部边被两个三角形共享，出现两次
    - 边界边只出现一次
    
    Parameters
    ----------
    node_num : int
        节点总数
    triangles : np.ndarray, shape (M, 3)
    
    Returns
    -------
    is_boundary : np.ndarray, shape (node_num,)
        布尔数组，True表示该节点在边界上
    """
    triangle_num = triangles.shape[0]
    
    # 构建边列表
    edges = []
    for tri in triangles:
        edges.append(tuple(sorted([tri[0], tri[1]])))
        edges.append(tuple(sorted([tri[1], tri[2]])))
        edges.append(tuple(sorted([tri[2], tri[0]])))
    
    # 统计每条边出现的次数
    edge_count = {}
    for e in edges:
        edge_count[e] = edge_count.get(e, 0) + 1
    
    # 边界节点：至少属于一条边界边
    is_boundary = np.zeros(node_num, dtype=bool)
    for e, count in edge_count.items():
        if count == 1:  # 边界边
            is_boundary[e[0]] = True
            is_boundary[e[1]] = True
    
    return is_boundary


def triangulation_adjacency_count(node_num: int, triangles: np.ndarray,
                                   triangle_neighbor: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
    """
    计算三角剖分中的节点邻接关系数量
    
    基于种子项目1330_triangulation中的triangulation_order3_adj_count。
    
    邻接定义：若两个节点同属于某个三角形，则它们相邻。
    
    Parameters
    ----------
    node_num : int
    triangles : np.ndarray, shape (M, 3)
    triangle_neighbor : np.ndarray, shape (M, 3), optional
        每个三角形三边的相邻三角形索引（-1表示边界）
    
    Returns
    -------
    adj_num : int
        邻接关系总数（含自邻接）
    adj_col : np.ndarray, shape (node_num + 1,)
        列指针数组，用于CSR格式的邻接矩阵
    """
    triangle_num = triangles.shape[0]
    
    # 初始化计数
    adj_col = np.ones(node_num, dtype=int)  # 每个节点至少与自己相邻
    
    for t in range(triangle_num):
        n1, n2, n3 = triangles[t]
        
        # 边(1,2)
        if triangle_neighbor is None:
            adj_col[n1] += 1
            adj_col[n2] += 1
        else:
            t2 = triangle_neighbor[t, 0]
            if t2 < 0 or t < t2:
                adj_col[n1] += 1
                adj_col[n2] += 1
        
        # 边(2,3)
        if triangle_neighbor is None:
            adj_col[n2] += 1
            adj_col[n3] += 1
        else:
            t2 = triangle_neighbor[t, 1]
            if t2 < 0 or t < t2:
                adj_col[n2] += 1
                adj_col[n3] += 1
        
        # 边(3,1)
        if triangle_neighbor is None:
            adj_col[n3] += 1
            adj_col[n1] += 1
        else:
            t2 = triangle_neighbor[t, 2]
            if t2 < 0 or t < t2:
                adj_col[n3] += 1
                adj_col[n1] += 1
    
    # 转换为指针
    adj_col_ptr = np.zeros(node_num + 1, dtype=int)
    adj_col_ptr[0] = 1
    for i in range(node_num):
        adj_col_ptr[i + 1] = adj_col_ptr[i] + adj_col[i]
    
    adj_num = adj_col_ptr[node_num] - 1
    
    return adj_num, adj_col_ptr


def build_triangle_neighbors(triangles: np.ndarray) -> np.ndarray:
    """
    构建三角形邻接关系
    
    对每个三角形的每条边，找到共享该边的相邻三角形。
    若边在边界上（无相邻三角形），则标记为-1。
    
    Parameters
    ----------
    triangles : np.ndarray, shape (M, 3)
    
    Returns
    -------
    neighbors : np.ndarray, shape (M, 3)
        neighbors[t, i]表示三角形t的第i条边的相邻三角形索引
    """
    M = triangles.shape[0]
    neighbors = np.full((M, 3), -1, dtype=int)
    
    # 建立边到三角形的映射
    edge_to_tri = {}
    for t in range(M):
        for e in range(3):
            v1 = triangles[t, e]
            v2 = triangles[t, (e + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_to_tri:
                edge_to_tri[edge] = []
            edge_to_tri[edge].append((t, e))
    
    # 填充邻接关系
    for edge, tri_list in edge_to_tri.items():
        if len(tri_list) == 2:
            (t1, e1), (t2, e2) = tri_list
            neighbors[t1, e1] = t2
            neighbors[t2, e2] = t1
    
    return neighbors


def fermi_surface_2d_slice(hamiltonian_func: callable,
                            kx_range: Tuple[float, float],
                            ky_range: Tuple[float, float],
                            kz_fixed: float,
                            e_fermi: float,
                            grid_size: int = 50,
                            band_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取二维截面中的Fermi面并三角剖分
    
    策略：
    1. 在k空间网格上计算能量E(k)
    2. 使用marching squares算法提取E(k) = E_F的等值线
    3. 简化：直接取E(k)最接近E_F的点，然后进行三角剖分
    
    Parameters
    ----------
    hamiltonian_func : callable
        输入k点(N,3)，输出能量(N,)
    kx_range, ky_range : tuple
    kz_fixed : float
    e_fermi : float
        Fermi能级
    grid_size : int
    band_index : int
    
    Returns
    -------
    fs_points : np.ndarray, shape (M, 2)
        Fermi面上的点（kx, ky）
    triangles : np.ndarray, shape (K, 3)
        三角剖分
    """
    kx = np.linspace(kx_range[0], kx_range[1], grid_size)
    ky = np.linspace(ky_range[0], ky_range[1], grid_size)
    
    # 计算能量
    energies = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            k = np.array([[kx[i], ky[j], kz_fixed]])
            e = hamiltonian_func(k)
            energies[i, j] = e[0] if hasattr(e, '__len__') else e
    
    # 寻找Fermi面附近的点（能量差小于阈值）
    threshold = 0.05 * (np.max(energies) - np.min(energies))
    mask = np.abs(energies - e_fermi) < threshold
    
    # 收集候选点
    fs_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if mask[i, j]:
                fs_points.append([kx[i], ky[j]])
    
    fs_points = np.array(fs_points)
    
    if len(fs_points) < 3:
        return fs_points, np.zeros((0, 3), dtype=int)
    
    # 三角剖分
    triangles = delaunay_triangulate_2d(fs_points)
    
    return fs_points, triangles


def node_values_to_element_average(node_values: np.ndarray,
                                    triangles: np.ndarray) -> np.ndarray:
    """
    将节点值平均到三角形元素上
    
    基于种子项目1340_triangulation_node_to_element的核心思想。
    
    公式：
        V_element = (1/3) * sum_{i=1}^3 V_node_i
    
    Parameters
    ----------
    node_values : np.ndarray, shape (node_num,) 或 (node_num, D)
    triangles : np.ndarray, shape (M, 3)
    
    Returns
    -------
    element_values : np.ndarray, shape (M,) 或 (M, D)
    """
    M = triangles.shape[0]
    node_values = np.asarray(node_values)
    
    if node_values.ndim == 1:
        element_values = np.zeros(M)
        for i in range(M):
            element_values[i] = np.mean(node_values[triangles[i]])
    else:
        element_values = np.zeros((M, node_values.shape[1]))
        for i in range(M):
            element_values[i] = np.mean(node_values[triangles[i]], axis=0)
    
    return element_values


def triangulation_total_area(points: np.ndarray, triangles: np.ndarray) -> float:
    """
    计算三角剖分的总面积
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
    triangles : np.ndarray, shape (M, 3)
    
    Returns
    -------
    total_area : float
    """
    total = 0.0
    for tri in triangles:
        t = points[tri].T  # shape (2, 3)
        total += triangle_area_2d(t)
    
    return total
