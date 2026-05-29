"""
climate_mesh.py
球面气候网格生成模块

基于正二十面体递归细分的球面三角网格生成，用于古气候模拟的空间离散化。
融合种子项目 1305_triangle_grid（三角形格点生成）与 748_medit_to_fem（网格 I/O 与格式转换思想）。
"""

import numpy as np


def _normalize(v):
    """将向量归一化为单位长度。"""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def generate_icosahedron():
    """
    生成单位球内接正二十面体。
    黄金比例 phi = (1 + sqrt(5)) / 2。
    返回: vertices (12, 3), faces (20, 3)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0,  phi, 0.0], [ 1.0,  phi, 0.0],
        [-1.0, -phi, 0.0], [ 1.0, -phi, 0.0],
        [0.0, -1.0,  phi], [0.0,  1.0,  phi],
        [0.0, -1.0, -phi], [0.0,  1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0,  1.0],
        [-phi, 0.0, -1.0], [-phi, 0.0,  1.0]
    ], dtype=np.float64)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)
    return vertices, faces


def subdivide_spherical_mesh(vertices, faces, n_subdiv=2):
    """
    对球面三角网格进行递归细分（Loop-like 细分）。
    每条边取中点后投影回球面，每个三角形细分为 4 个。
    """
    verts = {i: tuple(v) for i, v in enumerate(vertices)}
    face_list = [list(f) for f in faces]
    next_idx = len(verts)

    for _ in range(n_subdiv):
        edge_dict = {}
        new_faces = []

        def get_midpoint_index(i, j):
            nonlocal next_idx
            key = tuple(sorted((i, j)))
            if key not in edge_dict:
                vi = np.array(verts[i])
                vj = np.array(verts[j])
                mid = _normalize((vi + vj) * 0.5)
                edge_dict[key] = next_idx
                verts[next_idx] = tuple(mid)
                next_idx += 1
            return edge_dict[key]

        for tri in face_list:
            a, b, c = tri
            ab = get_midpoint_index(a, b)
            bc = get_midpoint_index(b, c)
            ca = get_midpoint_index(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        face_list = new_faces

    max_idx = max(verts.keys())
    verts_array = np.zeros((max_idx + 1, 3), dtype=np.float64)
    for idx, v in verts.items():
        verts_array[idx] = v
    return verts_array, np.array(face_list, dtype=int)


def compute_spherical_triangle_area(v1, v2, v3):
    """
    计算单位球面上三角形的面积。
    使用 L'Huilier 定理:
        tan(E/4) = sqrt(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2))
    其中 a,b,c 为球面角距离，E 为球面角盈，面积 = E。
    """
    def spherical_distance(a, b):
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.arccos(dot)

    a = spherical_distance(v2, v3)
    b = spherical_distance(v1, v3)
    c = spherical_distance(v1, v2)
    s = (a + b + c) * 0.5

    if s <= 0 or a <= 0 or b <= 0 or c <= 0:
        return 0.0

    tan_s2 = np.tan(s * 0.5)
    tan_sa2 = np.tan(max((s - a) * 0.5, 1e-15))
    tan_sb2 = np.tan(max((s - b) * 0.5, 1e-15))
    tan_sc2 = np.tan(max((s - c) * 0.5, 1e-15))

    tan_E4 = np.sqrt(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2)
    E = 4.0 * np.arctan(tan_E4)
    return E


def compute_mesh_areas(vertices, faces):
    """计算每个三角形单元的球面面积。"""
    areas = np.zeros(len(faces), dtype=np.float64)
    for i, tri in enumerate(faces):
        areas[i] = compute_spherical_triangle_area(
            vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        )
    return areas


def compute_dual_voronoi_areas(vertices, faces):
    """
    计算每个节点对应的 Voronoi 对偶面积，作为积分权重。
    采用重心分配法：每个三角形的面积三等分给三个顶点。
    """
    n_nodes = len(vertices)
    areas = np.zeros(n_nodes, dtype=np.float64)
    for tri in faces:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        area = compute_spherical_triangle_area(v1, v2, v3)
        areas[tri[0]] += area / 3.0
        areas[tri[1]] += area / 3.0
        areas[tri[2]] += area / 3.0
    return areas


def mesh_info(vertices, faces):
    """输出网格统计信息。"""
    areas = compute_mesh_areas(vertices, faces)
    return {
        'n_nodes': len(vertices),
        'n_faces': len(faces),
        'total_area': float(np.sum(areas)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas)),
        'mean_area': float(np.mean(areas))
    }
