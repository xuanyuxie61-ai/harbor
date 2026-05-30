
import numpy as np
from typing import Tuple, List, Set


def sort_triple(a: int, b: int, c: int) -> Tuple[int, int, int]:
    arr = sorted([a, b, c])
    return tuple(arr)


def tet_mesh_boundary_count(tet_elements: np.ndarray) -> Tuple[int, int, np.ndarray]:
    ntet = tet_elements.shape[0]
    if tet_elements.shape[1] != 4:
        raise ValueError("Tetrahedral elements must have 4 vertices")
    

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
    

    from collections import Counter
    face_counts = Counter(face_list)
    
    boundary_faces = [f for f, cnt in face_counts.items() if cnt == 1]
    n_boundary_faces = len(boundary_faces)
    

    n_nodes = int(tet_elements.max()) + 1
    boundary_node_mask = np.zeros(n_nodes, dtype=int)
    for f in boundary_faces:
        for vid in f:
            boundary_node_mask[vid] = 1
    
    n_boundary_nodes = int(np.sum(boundary_node_mask))
    return n_boundary_nodes, n_boundary_faces, boundary_node_mask


def tet_mesh_boundary_set(tet_elements: np.ndarray) -> np.ndarray:
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

    inside_mask = node_energies < energy_threshold
    

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
    if elements.min() == 1:
        return elements - 1
    return elements
