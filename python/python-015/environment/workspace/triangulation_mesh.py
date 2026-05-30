
import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, List, Optional


def delaunay_triangulate_2d(points: np.ndarray) -> np.ndarray:
    if points.shape[1] != 2:
        raise ValueError("点必须是二维的")
    
    if len(points) < 3:
        return np.zeros((0, 3), dtype=int)
    
    tri = Delaunay(points)
    return tri.simplices.astype(int)


def triangle_area_2d(t: np.ndarray) -> float:
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
    triangle_num = triangles.shape[0]
    

    edges = []
    for tri in triangles:
        edges.append(tuple(sorted([tri[0], tri[1]])))
        edges.append(tuple(sorted([tri[1], tri[2]])))
        edges.append(tuple(sorted([tri[2], tri[0]])))
    

    edge_count = {}
    for e in edges:
        edge_count[e] = edge_count.get(e, 0) + 1
    

    is_boundary = np.zeros(node_num, dtype=bool)
    for e, count in edge_count.items():
        if count == 1:
            is_boundary[e[0]] = True
            is_boundary[e[1]] = True
    
    return is_boundary


def triangulation_adjacency_count(node_num: int, triangles: np.ndarray,
                                   triangle_neighbor: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
    triangle_num = triangles.shape[0]
    

    adj_col = np.ones(node_num, dtype=int)
    
    for t in range(triangle_num):
        n1, n2, n3 = triangles[t]
        

        if triangle_neighbor is None:
            adj_col[n1] += 1
            adj_col[n2] += 1
        else:
            t2 = triangle_neighbor[t, 0]
            if t2 < 0 or t < t2:
                adj_col[n1] += 1
                adj_col[n2] += 1
        

        if triangle_neighbor is None:
            adj_col[n2] += 1
            adj_col[n3] += 1
        else:
            t2 = triangle_neighbor[t, 1]
            if t2 < 0 or t < t2:
                adj_col[n2] += 1
                adj_col[n3] += 1
        

        if triangle_neighbor is None:
            adj_col[n3] += 1
            adj_col[n1] += 1
        else:
            t2 = triangle_neighbor[t, 2]
            if t2 < 0 or t < t2:
                adj_col[n3] += 1
                adj_col[n1] += 1
    

    adj_col_ptr = np.zeros(node_num + 1, dtype=int)
    adj_col_ptr[0] = 1
    for i in range(node_num):
        adj_col_ptr[i + 1] = adj_col_ptr[i] + adj_col[i]
    
    adj_num = adj_col_ptr[node_num] - 1
    
    return adj_num, adj_col_ptr


def build_triangle_neighbors(triangles: np.ndarray) -> np.ndarray:
    M = triangles.shape[0]
    neighbors = np.full((M, 3), -1, dtype=int)
    

    edge_to_tri = {}
    for t in range(M):
        for e in range(3):
            v1 = triangles[t, e]
            v2 = triangles[t, (e + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_to_tri:
                edge_to_tri[edge] = []
            edge_to_tri[edge].append((t, e))
    

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
    kx = np.linspace(kx_range[0], kx_range[1], grid_size)
    ky = np.linspace(ky_range[0], ky_range[1], grid_size)
    

    energies = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            k = np.array([[kx[i], ky[j], kz_fixed]])
            e = hamiltonian_func(k)
            energies[i, j] = e[0] if hasattr(e, '__len__') else e
    

    threshold = 0.05 * (np.max(energies) - np.min(energies))
    mask = np.abs(energies - e_fermi) < threshold
    

    fs_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if mask[i, j]:
                fs_points.append([kx[i], ky[j]])
    
    fs_points = np.array(fs_points)
    
    if len(fs_points) < 3:
        return fs_points, np.zeros((0, 3), dtype=int)
    

    triangles = delaunay_triangulate_2d(fs_points)
    
    return fs_points, triangles


def node_values_to_element_average(node_values: np.ndarray,
                                    triangles: np.ndarray) -> np.ndarray:
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
    total = 0.0
    for tri in triangles:
        t = points[tri].T
        total += triangle_area_2d(t)
    
    return total
