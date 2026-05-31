"""
四面体网格边界提取与分子表面分析模块
基于 tet_mesh_boundary 核心算法：面提取、字典排序匹配、边界判定、节点重编号。

在蛋白质折叠中的应用：
- 从构象空间四面体网格中提取自由能盆地边界
- 识别分子表面 (Molecular Surface) 和溶剂可及表面 (SAS)
- 计算几何表面积和体积
- 验证网格水密性

数学基础:
    每个四面体有4个三角形面。对每面取3顶点编号升序排序。
    若相邻四面体共享一面，排序后该面出现连续两次 → 内部面。
    仅出现一次的面 → 边界面。
"""

import numpy as np
from typing import Tuple, List, Set


def sort_triple(a: int, b: int, c: int) -> Tuple[int, int, int]:
    """
    三个整数升序排序。
    
    Parameters
    ----------
    a, b, c : int
        待排序整数。
    
    Returns
    -------
    tuple
        升序排列的三元组。
    """
    arr = sorted([a, b, c])
    return tuple(arr)


def tet_mesh_boundary_count(tet_elements: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    统计四面体网格的边界节点数和边界面数，并标记边界节点。
    
    算法:
        1. 遍历每个四面体，生成其4个面（顶点索引升序）
        2. 对所有面排序后统计频次
        3. 出现1次的为边界面，出现2次的为内部面
        4. 标记边界面上的顶点为边界节点
    
    Parameters
    ----------
    tet_elements : np.ndarray, shape (ntet, 4)
        四面体单元连接表（0-based索引）。
    
    Returns
    -------
    n_boundary_nodes : int
        边界节点数。
    n_boundary_faces : int
        边界面数。
    boundary_node_mask : np.ndarray
        边界节点标记（1表示边界，0表示内部）。
    """
    ntet = tet_elements.shape[0]
    if tet_elements.shape[1] != 4:
        raise ValueError("Tetrahedral elements must have 4 vertices")
    
    # 收集所有面
    face_list = []
    for t in range(ntet):
        nodes = tet_elements[t]
        faces = [
            sort_triple(nodes[0], nodes[1], nodes[2]),
            sort_triple(nodes[0], nodes[1], nodes[3]),
            sort_triple(nodes[0], nodes[2], nodes[3]),
            sort_triple(nodes[1], nodes[2], nodes[3]),
        ]
        for f in faces:
            face_list.append(f)
    
    # 统计频次
    from collections import Counter
    face_counts = Counter(face_list)
    
    boundary_faces = [f for f, cnt in face_counts.items() if cnt == 1]
    n_boundary_faces = len(boundary_faces)
    
    # 标记边界节点
    n_nodes = int(tet_elements.max()) + 1
    boundary_node_mask = np.zeros(n_nodes, dtype=int)
    for f in boundary_faces:
        for vid in f:
            boundary_node_mask[vid] = 1
    
    n_boundary_nodes = int(np.sum(boundary_node_mask))
    return n_boundary_nodes, n_boundary_faces, boundary_node_mask


def tet_mesh_boundary_set(tet_elements: np.ndarray) -> np.ndarray:
    """
    生成边界面的节点连接表。
    
    Parameters
    ----------
    tet_elements : np.ndarray, shape (ntet, 4)
        四面体单元（0-based）。
    
    Returns
    -------
    boundary_faces : np.ndarray, shape (nbf, 3)
        边界三角形面连接表。
    """
    face_list = []
    for t in range(tet_elements.shape[0]):
        nodes = tet_elements[t]
        faces = [
            sort_triple(nodes[0], nodes[1], nodes[2]),
            sort_triple(nodes[0], nodes[1], nodes[3]),
            sort_triple(nodes[0], nodes[2], nodes[3]),
            sort_triple(nodes[1], nodes[2], nodes[3]),
        ]
        face_list.extend(faces)
    
    from collections import Counter
    face_counts = Counter(face_list)
    boundary_faces = [list(f) for f, cnt in face_counts.items() if cnt == 1]
    return np.array(boundary_faces, dtype=int)


def compute_surface_area_and_volume(nodes: np.ndarray,
                                    boundary_faces: np.ndarray) -> Tuple[float, float]:
    """
    计算边界表面的面积和包围体积。
    
    面积: 所有边界三角形面积之和。
    体积: 利用散度定理，对每个三角形计算有符号体积贡献:
        V = (1/6) * Σ ( (v1 × v2) · v3 )
    
    其中 v1, v2, v3 为三角形顶点坐标。
    
    Parameters
    ----------
    nodes : np.ndarray, shape (n, 3)
        节点坐标。
    boundary_faces : np.ndarray, shape (nf, 3)
        边界三角形面。
    
    Returns
    -------
    area : float
        表面积。
    volume : float
        包围体积。
    """
    area = 0.0
    volume = 0.0
    for face in boundary_faces:
        v1, v2, v3 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
        cross = np.cross(v2 - v1, v3 - v1)
        tri_area = 0.5 * np.linalg.norm(cross)
        area += tri_area
        volume += np.dot(cross, v1) / 6.0
    
    volume = abs(volume)
    return float(area), float(volume)


def extract_free_energy_basin_boundary(nodes: np.ndarray, tet_elements: np.ndarray,
                                       energy_threshold: float,
                                       node_energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从构象空间四面体网格中提取自由能盆地边界。
    
    策略:
        只保留那些至少有一个顶点能量低于阈值的四面体，
        然后提取这些保留四面体的边界。
        这近似于自由能盆地 {x: F(x) < F_threshold} 的边界。
    
    Parameters
    ----------
    nodes : np.ndarray
        节点坐标。
    tet_elements : np.ndarray
        四面体连接表。
    energy_threshold : float
        能量阈值。
    node_energies : np.ndarray
        每个节点上的能量值。
    
    Returns
    -------
    basin_nodes : np.ndarray
        盆地内节点。
    boundary_faces : np.ndarray
        盆地边界三角形面。
    """
    # 标记能量低于阈值的节点
    inside_mask = node_energies < energy_threshold
    
    # 只保留至少有一个顶点在盆地内的四面体
    valid_tets = []
    for t in range(tet_elements.shape[0]):
        if np.any(inside_mask[tet_elements[t]]):
            valid_tets.append(tet_elements[t])
    
    if len(valid_tets) == 0:
        return nodes, np.zeros((0, 3), dtype=int)
    
    valid_tets = np.array(valid_tets)
    boundary_faces = tet_mesh_boundary_set(valid_tets)
    return nodes, boundary_faces


def mesh_base_one_check(elements: np.ndarray) -> np.ndarray:
    """
    自动检测并转换 1-based 索引为 0-based 索引。
    
    Parameters
    ----------
    elements : np.ndarray
        单元连接表。
    
    Returns
    -------
    elements_fixed : np.ndarray
        确保为 0-based 的连接表。
    """
    if elements.min() == 1:
        return elements - 1
    return elements
