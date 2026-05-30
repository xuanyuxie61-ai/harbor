
import numpy as np
from typing import Tuple


def tetrahedron_jacobian_determinant(nodes: np.ndarray, element: np.ndarray) -> float:
    x0 = nodes[element[0]]
    x1 = nodes[element[1]]
    x2 = nodes[element[2]]
    x3 = nodes[element[3]]
    mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
    return float(np.linalg.det(mat))


def orient_tetrahedra(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    oriented = elements.copy()
    fixed_count = 0
    for i in range(oriented.shape[0]):
        detJ = tetrahedron_jacobian_determinant(nodes, oriented[i])
        if detJ < 0:

            oriented[i, 1], oriented[i, 2] = oriented[i, 2], oriented[i, 1]
            fixed_count += 1

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
    p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    return float(np.linalg.norm(cross) * 0.5)


def triangle_normal(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
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
