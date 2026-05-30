
import numpy as np
from typing import Tuple, List, Optional


def tri_surface_read_nodes(node_file: str) -> np.ndarray:
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
    try:
        elems = np.loadtxt(element_file, dtype=int)
    except Exception as e:
        raise RuntimeError(f"tri_surface_read_elements: failed to read {element_file}: {e}")
    if elems.ndim == 1:
        elems = elems.reshape(1, -1)
    if elems.shape[1] != 3:
        raise ValueError(f"tri_surface_read_elements: expected 3 columns, got {elems.shape[1]}")

    if np.min(elems) == 1:
        elems = elems - 1
    return elems


def tri_surface_write(node_file: str, element_file: str,
                      nodes: np.ndarray, elements: np.ndarray) -> None:
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("tri_surface_write: nodes must have shape (n, 3)")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("tri_surface_write: elements must have shape (n, 3)")
    np.savetxt(node_file, nodes, fmt="%.14g")
    np.savetxt(element_file, elements + 1, fmt="%d")


def stla_size_fast(stla_file_name: str) -> Tuple[int, int, int, int]:
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
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross)
    return float(area)


def tet_mesh_tet_neighbors(tetra_order: int, tetra_num: int,
                           tetra_node: np.ndarray) -> np.ndarray:
    if tetra_order not in (4, 10):
        raise ValueError("tet_mesh_tet_neighbors: tetra_order must be 4 or 10")
    if tetra_node.shape != (tetra_order, tetra_num):
        raise ValueError("tet_mesh_tet_neighbors: tetra_node shape mismatch")

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

    for entries in face_dict.values():
        if len(entries) == 2:
            (t1, f1), (t2, f2) = entries
            tetra_neighbor[f1, t1] = t2
            tetra_neighbor[f2, t2] = t1
    return tetra_neighbor


def tri_mesh_edge_neighbors(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
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
    return np.min(nodes, axis=0), np.max(nodes, axis=0)


def pcf_triangular_mesh(pitch: float, hole_radius: float, n_rings: int) -> Tuple[np.ndarray, np.ndarray]:

    R = pitch * n_rings
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_boundary = np.column_stack((R * np.cos(angles), R * np.sin(angles)))

    from scipy.spatial import Delaunay

    np.random.seed(42)
    n_interior = max(50, 10 * n_rings)
    interior = []
    for _ in range(n_interior * 5):
        if len(interior) >= n_interior:
            break
        p = np.random.uniform(-R, R, 2)

        if np.linalg.norm(p) < R * 0.9:
            interior.append(p)
    points = np.vstack([hex_boundary, np.array(interior)])
    tri = Delaunay(points)
    return points, tri.simplices
