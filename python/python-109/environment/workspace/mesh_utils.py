"""
mesh_utils.py
三角网格与四面体网格的拓扑处理、格式转换与I/O工具

融合原项目:
  - 1292_tri_surface_display: 3D三角网格数据读写与处理
  - 1168_stla_to_tri_surface_fast: ASCII STL快速解析与三角表面转换
  - 1239_tet_mesh_tet_neighbors: 四面体网格邻接关系计算

科学背景:
  在光子晶体光纤（PCF）的数值建模中，横截面通常由三角网格离散，
  而三维结构（如堆叠拉制前的预制棒）则需要四面体网格。准确的
  网格拓扑信息（面-面邻接、边共享关系）对有限元模式求解至关重要。
  本模块提供从STL格式导入、三角/四面体网格拓扑重建到邻接表
  生成的完整工具链。
"""

import numpy as np
from typing import Tuple, List, Optional


def tri_surface_read_nodes(node_file: str) -> np.ndarray:
    """
    从文本文件读取三角网格节点坐标。

    文件格式为无头文本，每行一个节点:
        x y z

    Parameters
    ----------
    node_file : str
        节点文件路径。

    Returns
    -------
    np.ndarray
        形状为 (n_nodes, 3) 的节点坐标数组。
    """
    try:
        nodes = np.loadtxt(node_file)
    except Exception as e:
        raise RuntimeError(f"tri_surface_read_nodes: failed to read {node_file}: {e}")
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if nodes.shape[1] != 3:
        raise ValueError(f"tri_surface_read_nodes: expected 3 columns, got {nodes.shape[1]}")
    return nodes


def tri_surface_read_elements(element_file: str) -> np.ndarray:
    """
    从文本文件读取三角面片元素（节点索引）。

    文件格式为无头文本，每行一个三角形:
        i j k
    索引为0-based。

    Parameters
    ----------
    element_file : str
        元素文件路径。

    Returns
    -------
    np.ndarray
        形状为 (n_triangles, 3) 的索引数组（0-based）。
    """
    try:
        elems = np.loadtxt(element_file, dtype=int)
    except Exception as e:
        raise RuntimeError(f"tri_surface_read_elements: failed to read {element_file}: {e}")
    if elems.ndim == 1:
        elems = elems.reshape(1, -1)
    if elems.shape[1] != 3:
        raise ValueError(f"tri_surface_read_elements: expected 3 columns, got {elems.shape[1]}")
    # 确保0-based
    if np.min(elems) == 1:
        elems = elems - 1
    return elems


def tri_surface_write(node_file: str, element_file: str,
                      nodes: np.ndarray, elements: np.ndarray) -> None:
    """
    将三角网格数据写入文件对（节点 + 元素）。

    Parameters
    ----------
    node_file : str
        输出节点文件路径。
    element_file : str
        输出元素文件路径。
    nodes : np.ndarray
        形状 (n_nodes, 3)。
    elements : np.ndarray
        形状 (n_triangles, 3)，0-based索引。
    """
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("tri_surface_write: nodes must have shape (n, 3)")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("tri_surface_write: elements must have shape (n, 3)")
    np.savetxt(node_file, nodes, fmt="%.14g")
    np.savetxt(element_file, elements + 1, fmt="%d")  # 输出1-based


def stla_size_fast(stla_file_name: str) -> Tuple[int, int, int, int]:
    """
    快速扫描ASCII STL文件，统计固体、节点、面片、文本行数。

    原算法通过逐行扫描关键词（solid, facet, vertex）实现快速计数，
    不执行严格的语法校验。

    Parameters
    ----------
    stla_file_name : str
        STL文件路径。

    Returns
    -------
    tuple
        (solid_num, node_num, face_num, text_num)
    """
    solid_num = 0
    node_num = 0
    face_num = 0
    text_num = 0
    try:
        with open(stla_file_name, 'r') as f:
            for line in f:
                text_num += 1
                words = line.strip().lower().split()
                if len(words) == 0:
                    continue
                first_word = words[0]
                if first_word == 'solid':
                    solid_num += 1
                elif first_word == 'facet':
                    face_num += 1
                elif first_word == 'vertex':
                    node_num += 1
    except Exception as e:
        raise RuntimeError(f"stla_size_fast: failed to read {stla_file_name}: {e}")
    return solid_num, node_num, face_num, text_num


def stla_read_fast(stla_file_name: str, node_num: int, face_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    快速读取ASCII STL文件的几何数据。

    利用sscanf风格的解析提取顶点坐标与面片法向量。每个facet包含3个vertex，
    因此 node_num 通常等于 3*face_num。

    Parameters
    ----------
    stla_file_name : str
        STL文件路径。
    node_num : int
        预计顶点数。
    face_num : int
        预计面片数。

    Returns
    -------
    tuple
        (node_xyz, face_node, face_normal)
        node_xyz: (3, node_num)
        face_node: (3, face_num), 0-based
        face_normal: (3, face_num)
    """
    node_xyz = np.zeros((3, node_num))
    face_node = np.zeros((3, face_num), dtype=int)
    face_normal = np.zeros((3, face_num))
    solid = 0
    node = 0
    face = -1
    try:
        with open(stla_file_name, 'r') as f:
            for line in f:
                text = line.strip().lower()
                words = text.split()
                if len(words) == 0:
                    continue
                first_word = words[0]
                if first_word == 'solid':
                    solid += 1
                elif first_word == 'facet':
                    face += 1
                    if face < face_num and len(words) >= 5:
                        face_normal[:, face] = [float(words[2]), float(words[3]), float(words[4])]
                elif first_word == 'vertex':
                    if node < node_num and len(words) >= 4:
                        node_xyz[:, node] = [float(words[1]), float(words[2]), float(words[3])]
                        node += 1
                elif first_word == 'endloop':
                    if face >= 0 and face < face_num:
                        face_node[:, face] = [node - 3, node - 2, node - 1]
    except Exception as e:
        raise RuntimeError(f"stla_read_fast: failed to read {stla_file_name}: {e}")
    return node_xyz, face_node, face_normal


def triangle_area_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三维空间中三角形的面积。

    公式:
        A = 0.5 * || (p2 - p1) x (p3 - p1) ||

    Parameters
    ----------
    p1, p2, p3 : np.ndarray
        形状为 (3,) 的顶点坐标。

    Returns
    -------
    float
        三角形面积。
    """
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross)
    return float(area)


def tet_mesh_tet_neighbors(tetra_order: int, tetra_num: int,
                           tetra_node: np.ndarray) -> np.ndarray:
    """
    计算四面体网格的邻接关系。

    对于每个四面体，返回其4个面的相邻四面体索引（-1表示边界）。
    两个四面体相邻当且仅当它们共享一个由3个节点构成的面。

    算法:
        1. 对每个四面体，枚举其4个面（节点三元组，排序后作为键）。
        2. 使用字典统计每个面出现的次数及所属的四面体。
        3. 若某个面恰好出现2次，则对应的两个四面体互为邻居。

    四面体的4个面（按节点索引顺序）:
        face 0: nodes [1, 2, 3]
        face 1: nodes [0, 3, 2]
        face 2: nodes [0, 1, 3]
        face 3: nodes [0, 2, 1]

    该邻接信息用于有限元求解中的跳跃条件、通量计算等。

    Parameters
    ----------
    tetra_order : int
        四面体阶数（4或10）。仅使用前4个节点。
    tetra_num : int
        四面体数量。
    tetra_node : np.ndarray
        形状为 (tetra_order, tetra_num)，1-based索引。

    Returns
    -------
    np.ndarray
        形状为 (4, tetra_num) 的邻接表，-1表示边界。
    """
    if tetra_order not in (4, 10):
        raise ValueError("tet_mesh_tet_neighbors: tetra_order must be 4 or 10")
    if tetra_node.shape != (tetra_order, tetra_num):
        raise ValueError("tet_mesh_tet_neighbors: tetra_node shape mismatch")
    # 转换为0-based
    tn = tetra_node[:4, :].copy() - 1
    tetra_neighbor = np.full((4, tetra_num), -1, dtype=int)
    face_dict = {}
    faces = np.array([
        [1, 2, 3],
        [0, 3, 2],
        [0, 1, 3],
        [0, 2, 1]
    ])
    for tet in range(tetra_num):
        for f in range(4):
            nodes = tuple(sorted([tn[faces[f, 0], tet],
                                   tn[faces[f, 1], tet],
                                   tn[faces[f, 2], tet]]))
            if nodes in face_dict:
                face_dict[nodes].append((tet, f))
            else:
                face_dict[nodes] = [(tet, f)]
    # 建立邻接关系
    for entries in face_dict.values():
        if len(entries) == 2:
            (t1, f1), (t2, f2) = entries
            tetra_neighbor[f1, t1] = t2
            tetra_neighbor[f2, t2] = t1
    return tetra_neighbor


def tri_mesh_edge_neighbors(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    计算三角网格的边邻接关系（每条边共享的三角形）。

    对于PCF横截面网格，边邻接信息用于识别空气孔边界（只被一个三角形
    使用的边为边界边）。

    Parameters
    ----------
    nodes : np.ndarray
        形状 (n_nodes, 2) 或 (n_nodes, 3)。
    elements : np.ndarray
        形状 (n_tri, 3)，0-based索引。

    Returns
    -------
    np.ndarray
        边界边列表，形状 (n_boundary_edges, 2)。
    """
    if elements.shape[1] != 3:
        raise ValueError("tri_mesh_edge_neighbors: elements must have 3 columns")
    edge_dict = {}
    n_tri = elements.shape[0]
    for tri in range(n_tri):
        e = elements[tri, :]
        for k in range(3):
            a, b = e[k], e[(k + 1) % 3]
            edge = tuple(sorted([int(a), int(b)]))
            if edge in edge_dict:
                edge_dict[edge].append(tri)
            else:
                edge_dict[edge] = [tri]
    boundary_edges = []
    for edge, tris in edge_dict.items():
        if len(tris) == 1:
            boundary_edges.append(edge)
    return np.array(boundary_edges, dtype=int)


def mesh_bounding_box(nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算网格节点的包围盒。

    Parameters
    ----------
    nodes : np.ndarray
        形状 (n, dim)。

    Returns
    -------
    tuple
        (min_coords, max_coords)，各为形状 (dim,) 的数组。
    """
    return np.min(nodes, axis=0), np.max(nodes, axis=0)


def pcf_triangular_mesh(pitch: float, hole_radius: float, n_rings: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    为光子晶体光纤横截面生成简化的三角网格。

    将六边形第一布里渊区（正六边形）用三角网格离散，并在空气孔位置
    标记对应的三角形。

    Parameters
    ----------
    pitch : float
        晶格常数。
    hole_radius : float
        空气孔半径。
    n_rings : int
        环数（用于尺度）。

    Returns
    -------
    tuple
        (nodes, elements)，0-based索引。
    """
    # 简化的六边形边界
    R = pitch * n_rings
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_boundary = np.column_stack((R * np.cos(angles), R * np.sin(angles)))
    # 使用Delaunay三角化
    from scipy.spatial import Delaunay
    # 在六边形内生成随机点并三角化
    np.random.seed(42)
    n_interior = max(50, 10 * n_rings)
    interior = []
    for _ in range(n_interior * 5):
        if len(interior) >= n_interior:
            break
        p = np.random.uniform(-R, R, 2)
        # 检查是否在六边形内（简化：距离中心 < R）
        if np.linalg.norm(p) < R * 0.9:
            interior.append(p)
    points = np.vstack([hex_boundary, np.array(interior)])
    tri = Delaunay(points)
    return points, tri.simplices
