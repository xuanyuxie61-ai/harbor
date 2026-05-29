"""
网格方向一致性检查与修正模块
==============================
基于种子项目:
  - 1344_triangulation_orient: 三角剖分方向修正

科学背景:
  在有限元计算中，单元雅可比行列式的符号决定局部坐标系的方向。
  对于四面体单元，若节点排列导致雅可比为负，则单元"翻转"，会
  破坏数值积分的正确性。本模块确保：
  1. 所有四面体单元的雅可比行列式为正
  2. 二维表面三角形的顶点按逆时针排列(正方向)

关键公式:
  - 四面体有向体积: V = det([x1-x0, x2-x0, x3-x0]) / 6
  - 三角形有向面积: A = 0.5 * ((x1-x0)×(x2-x0)) · n
  - 若 V < 0，交换节点1和节点2可翻转方向
"""

import numpy as np
from typing import Tuple


def tetrahedron_jacobian_determinant(nodes: np.ndarray, element: np.ndarray) -> float:
    """
    计算4节点四面体单元的雅可比行列式(6倍有向体积)。
    J = det([x1-x0, x2-x0, x3-x0])
    """
    x0 = nodes[element[0]]
    x1 = nodes[element[1]]
    x2 = nodes[element[2]]
    x3 = nodes[element[3]]
    mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
    return float(np.linalg.det(mat))


def orient_tetrahedra(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    修正四面体单元的节点顺序，使得所有单元的雅可比行列式为正。
    若某单元雅可比为负，则交换节点1和节点2以翻转方向。

    参数:
        nodes: (N, 3) 节点坐标
        elements: (E, 4) 单元连接表

    返回:
        oriented_elements: 修正后的单元连接表
    """
    oriented = elements.copy()
    fixed_count = 0
    for i in range(oriented.shape[0]):
        detJ = tetrahedron_jacobian_determinant(nodes, oriented[i])
        if detJ < 0:
            # 交换节点1和节点2以翻转方向
            oriented[i, 1], oriented[i, 2] = oriented[i, 2], oriented[i, 1]
            fixed_count += 1
    # 边界处理: 对零体积单元发出警告
    zero_vol = 0
    for i in range(oriented.shape[0]):
        detJ = tetrahedron_jacobian_determinant(nodes, oriented[i])
        if abs(detJ) < 1e-14:
            zero_vol += 1
    if zero_vol > 0:
        print(f"[mesh_orientation] 警告: 检测到 {zero_vol} 个零体积或近似零体积单元")
    if fixed_count > 0:
        print(f"[mesh_orientation] 已修正 {fixed_count} 个方向错误的四面体单元")
    return oriented


def triangle_oriented_area(nodes: np.ndarray, tri: np.ndarray) -> float:
    """
    计算3D空间中三角形的有向面积(投影到最佳平面)。
    返回有向面积的两倍(叉积模)。
    """
    p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    return float(np.linalg.norm(cross) * 0.5)


def triangle_normal(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    计算三角形单位法向量。
    n = (v1 × v2) / |v1 × v2|
    """
    p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return cross / norm


def orient_surface_triangles(nodes: np.ndarray, triangles: np.ndarray,
                             reference_normal: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    调整表面三角形的顶点顺序，使其法向量与参考方向一致(外法向)。
    若某三角形法向量与参考法向量的点积为负，则交换最后两个顶点。

    参数:
        nodes: (N, 3) 节点坐标
        triangles: (M, 3) 三角形连接表
        reference_normal: 期望的法向量方向

    返回:
        oriented_triangles: 修正后的三角形连接表
    """
    oriented = triangles.copy()
    fixed = 0
    ref = reference_normal / (np.linalg.norm(reference_normal) + 1e-14)
    for i in range(oriented.shape[0]):
        n = triangle_normal(nodes, oriented[i])
        if np.dot(n, ref) < 0:
            oriented[i, 1], oriented[i, 2] = oriented[i, 2], oriented[i, 1]
            fixed += 1
    if fixed > 0:
        print(f"[mesh_orientation] 已修正 {fixed} 个表面三角形的法向方向")
    return oriented
