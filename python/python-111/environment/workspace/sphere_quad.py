
import numpy as np
from typing import Tuple, List, Callable


def icosahedron_shape() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
    ], dtype=float)
    

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    vertices = verts / norms
    
    faces = np.array([
        [0, 2, 8], [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2],
        [3, 1, 9], [3, 9, 5], [3, 5, 7], [3, 7, 11], [3, 11, 1],
        [1, 4, 9], [1, 6, 4], [1, 11, 6], [1, 3, 11], [3, 9, 1],
        [2, 5, 8], [2, 7, 5], [2, 10, 7], [2, 0, 10], [0, 8, 2],
        [4, 8, 9], [4, 1, 6], [6, 11, 10], [10, 7, 2], [5, 9, 8],
        [7, 11, 5], [11, 3, 7], [3, 5, 9], [6, 1, 11], [8, 4, 9],
    ], dtype=int)
    return vertices, faces


def sphere01_triangle_project(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                               n: int) -> np.ndarray:
    points = []
    for i in range(n):
        for j in range(n - i):
            k = n - 1 - i - j
            s = i / (n - 1) if n > 1 else 1.0
            t = j / (n - 1) if n > 1 else 0.0
            u = k / (n - 1) if n > 1 else 0.0
            p = s * v1 + t * v2 + u * v3
            norm = np.linalg.norm(p)
            if norm > 1e-12:
                p = p / norm
            points.append(p)
    return np.array(points)


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:

    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    s = 0.5 * (a + b + c)

    tan_s2 = np.tan(max(s / 2.0, 1e-12))
    tan_sa2 = np.tan(max((s - a) / 2.0, 1e-12))
    tan_sb2 = np.tan(max((s - b) / 2.0, 1e-12))
    tan_sc2 = np.tan(max((s - c) / 2.0, 1e-12))
    
    area = 4.0 * np.arctan(np.sqrt(max(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2, 0.0)))
    return float(area)


def sphere01_quad_icos1c(n_subdivide: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if n_subdivide < 1:
        raise ValueError("n_subdivide must be at least 1")
    
    vertices, faces = icosahedron_shape()
    all_points = []
    all_weights = []
    
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]


        edge_points = []
        for i in range(n_subdivide + 1):
            for j in range(n_subdivide + 1 - i):
                k = n_subdivide - i - j
                s = i / n_subdivide
                t = j / n_subdivide
                u = k / n_subdivide
                p = s * v1 + t * v2 + u * v3
                norm = np.linalg.norm(p)
                if norm > 0:
                    p = p / norm
                edge_points.append(p)
        


        centroid = (v1 + v2 + v3) / 3.0
        centroid = centroid / np.linalg.norm(centroid)
        area = sphere01_triangle_vertices_to_area(v1, v2, v3)
        all_points.append(centroid)
        all_weights.append(area)
    
    points = np.array(all_points)
    weights = np.array(all_weights)

    total = np.sum(weights)
    if total > 0:
        weights = weights * (4.0 * np.pi / total)
    return points, weights


def sphere01_quad_mc(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)



    z = rng.uniform(-1.0, 1.0, size=n_samples)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    r = np.sqrt(1.0 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, z))
    weights = np.full(n_samples, 4.0 * np.pi / n_samples)
    return points, weights


def sphere01_monomial_integral(e1: int, e2: int, e3: int) -> float:
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative")
    if (e1 % 2 == 1) or (e2 % 2 == 1) or (e3 % 2 == 1):
        return 0.0
    
    from scipy.special import gamma
    num = 2.0 * gamma((e1 + 1.0) / 2.0) * gamma((e2 + 1.0) / 2.0) * gamma((e3 + 1.0) / 2.0)
    den = gamma((e1 + e2 + e3 + 3.0) / 2.0)
    return float(num / den)


def integrate_orientational_distribution(odf: Callable[[np.ndarray], np.ndarray],
                                         n_subdivide: int = 3) -> float:
    points, weights = sphere01_quad_icos1c(n_subdivide)
    values = odf(points)
    return float(np.sum(values * weights))


def compute_nmr_order_parameter(protein_orientation: np.ndarray,
                                 n_subdivide: int = 3) -> float:
    vec = np.array(protein_orientation, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return 0.0
    vec = vec / norm
    
    points, weights = sphere01_quad_icos1c(n_subdivide)
    cos_theta = np.dot(points, vec)
    p2 = 0.5 * (3.0 * cos_theta ** 2 - 1.0)
    s2 = float(np.sum(p2 * weights) / np.sum(weights))

    return abs(s2)
