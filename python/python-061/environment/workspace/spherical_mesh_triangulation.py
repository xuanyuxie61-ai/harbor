"""
球面三角网格生成与拓扑分析模块
================================
基于种子项目 874_ply_to_tri_surface 的三角网格转换思想，
结合 1194_t_puzzle_gui 的几何拼图/多边形分解思想。

核心科学问题：
    球面离散化是大气模式的基础。本模块实现基于正二十面体的
    球面递归细分（geodesic grid）和球面 Voronoi 图生成，
    用于构建非结构化的球面大气网格。

正二十面体顶点（黄金比例 φ = (1+√5)/2）：
    V = {(0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)} / ||·||

递归细分规则：
    对于每条边的中点，投影到单位球面：
        V_new = V_mid / ||V_mid||
    
    一个三角形细分为4个小三角形。

球面三角形面积（L'Huilier 公式）：
    令 s = (a+b+c)/2，其中 a,b,c 为球面边长（大圆弧对应的中心角）
    tan(E/4) = √(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))
    Area = E * R²
    
    其中 E 为球面过剩（spherical excess）。
"""

import numpy as np


PHI = (1.0 + np.sqrt(5.0)) / 2.0


def normalize_to_sphere(v):
    """
    将向量归一化到单位球面。
    
    参数:
        v: (n, 3) 或 (3,) 向量数组
    
    返回:
        归一化后的向量
    """
    v = np.array(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return v / norms[:, np.newaxis]


def icosahedron_vertices():
    """
    生成单位球面上正二十面体的12个顶点。
    """
    verts = np.array([
        [-1.0,  PHI,  0.0],
        [ 1.0,  PHI,  0.0],
        [-1.0, -PHI,  0.0],
        [ 1.0, -PHI,  0.0],
        [ 0.0, -1.0,  PHI],
        [ 0.0,  1.0,  PHI],
        [ 0.0, -1.0, -PHI],
        [ 0.0,  1.0, -PHI],
        [ PHI,  0.0, -1.0],
        [ PHI,  0.0,  1.0],
        [-PHI,  0.0, -1.0],
        [-PHI,  0.0,  1.0]
    ], dtype=float)
    return normalize_to_sphere(verts)


def icosahedron_faces():
    """
    正二十面体的20个面（三角形），每个面由3个顶点索引组成。
    """
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)
    return faces


def subdivide_sphere_mesh(vertices, faces):
    """
    对球面三角网格进行一次递归细分。
    
    每条边的中点投影到球面，每个三角形分为4个。
    基于 874_ply_to_tri_surface 的三角化思想。
    
    参数:
        vertices: (n_v, 3) 顶点坐标
        faces: (n_f, 3) 面片索引
    
    返回:
        new_vertices, new_faces: 细分后的网格
    """
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)
    
    new_vertices = vertices.tolist()
    new_faces = []
    
    # 边到中点索引的映射
    edge_map = {}
    
    def get_midpoint_index(v1, v2):
        """获取或创建边的中点顶点。"""
        key = tuple(sorted([v1, v2]))
        if key in edge_map:
            return edge_map[key]
        
        mid = (vertices[v1] + vertices[v2]) / 2.0
        mid = normalize_to_sphere(mid)
        idx = len(new_vertices)
        new_vertices.append(mid)
        edge_map[key] = idx
        return idx
    
    for face in faces:
        v0, v1, v2 = face
        
        # 三条边的中点
        a = get_midpoint_index(v0, v1)
        b = get_midpoint_index(v1, v2)
        c = get_midpoint_index(v2, v0)
        
        # 分为4个三角形
        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])
    
    return np.array(new_vertices, dtype=float), np.array(new_faces, dtype=int)


def generate_geodesic_grid(refinement_levels=3):
    """
    生成测地线网格（递归细分正二十面体）。
    
    参数:
        refinement_levels: 递归细分次数
    
    返回:
        vertices: (n_v, 3) 顶点
        faces: (n_f, 3) 面片
    """
    vertices = icosahedron_vertices()
    faces = icosahedron_faces()
    
    for _ in range(refinement_levels):
        vertices, faces = subdivide_sphere_mesh(vertices, faces)
    
    return vertices, faces


def spherical_triangle_area(v1, v2, v3, radius=6.371e6):
    """
    计算球面三角形的面积（L'Huilier 定理）。
    
    公式：
        令 a, b, c 为单位向量之间的中心角：
            cos(a) = v2·v3, cos(b) = v1·v3, cos(c) = v1·v2
        
        半周角 s = (a+b+c)/2
        
        tan(E/4) = √(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))
        
        Area = E * R²
    
    参数:
        v1, v2, v3: 三角形顶点（单位向量）
        radius: 球半径
    
    返回:
        area: 球面面积
    """
    v1 = normalize_to_sphere(v1)
    v2 = normalize_to_sphere(v2)
    v3 = normalize_to_sphere(v3)
    
    # 中心角（使用 atan2 保证数值稳定）
    a = np.arctan2(np.linalg.norm(np.cross(v2, v3)), np.dot(v2, v3))
    b = np.arctan2(np.linalg.norm(np.cross(v1, v3)), np.dot(v1, v3))
    c = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    
    s = 0.5 * (a + b + c)
    
    # 防止数值误差导致 tan 的参数为负
    tan_s2 = np.tan(s / 2.0)
    tan_sa2 = np.tan(max(s - a, 0.0) / 2.0)
    tan_sb2 = np.tan(max(s - b, 0.0) / 2.0)
    tan_sc2 = np.tan(max(s - c, 0.0) / 2.0)
    
    product = tan_s2 * tan_sa2 * tan_sb2 * tan_sc2
    product = max(product, 0.0)
    
    E = 4.0 * np.arctan(np.sqrt(product))
    area = E * radius**2
    
    return area


def compute_mesh_statistics(vertices, faces, radius=6.371e6):
    """
    计算球面网格的统计信息。
    
    返回:
        stats: 字典，包含顶点数、面数、平均面积、面积离散度等
    """
    n_v = len(vertices)
    n_f = len(faces)
    
    areas = []
    for face in faces:
        area = spherical_triangle_area(vertices[face[0]], vertices[face[1]],
                                       vertices[face[2]], radius)
        areas.append(area)
    
    areas = np.array(areas)
    
    stats = {
        'n_vertices': n_v,
        'n_faces': n_f,
        'total_area': np.sum(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'area_uniformity': np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
    }
    
    return stats


def spherical_voronoi_centroids(vertices, faces):
    """
    计算球面 Voronoi 单元的质心（基于 1194_t_puzzle_gui 的拼图思想）。
    
    每个三角形的质心投影到球面：
        C = (v1 + v2 + v3) / 3
        C_sphere = C / ||C||
    
    参数:
        vertices: (n_v, 3) 顶点
        faces: (n_f, 3) 面片
    
    返回:
        centroids: (n_f, 3) 面片质心
    """
    centroids = []
    for face in faces:
        c = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        c = normalize_to_sphere(c)
        centroids.append(c)
    return np.array(centroids)
