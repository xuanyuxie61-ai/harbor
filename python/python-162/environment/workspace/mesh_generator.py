
import numpy as np
from typing import Tuple, List
from geometry_engine import BatteryCellGeometry


def in_rectangle(x: float, y: float, x1: float, x2: float, y1: float, y2: float) -> bool:
    return x1 - 1e-10 <= x <= x2 + 1e-10 and y1 - 1e-10 <= y <= y2 + 1e-10


def generate_structured_triangle_mesh(nx: int, ny: int, geometry: BatteryCellGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tw = geometry.total_width
    th = geometry.total_height
    dx = tw / nx
    dy = th / ny


    nodes = []
    node_index = {}
    idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy

            nodes.append([x, y])
            node_index[(i, j)] = idx
            idx += 1


    tab_extra = []

    nodes = np.array(nodes, dtype=float)


    elements = []
    region_tags = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_index[(i, j)]
            n2 = node_index[(i + 1, j)]
            n3 = node_index[(i + 1, j + 1)]
            n4 = node_index[(i, j + 1)]

            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])

            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dy
            tag = geometry.classify_point(cx, cy)
            region_tags.append(tag)
            region_tags.append(tag)

    elements = np.array(elements, dtype=int)
    region_tags = np.array(region_tags, dtype=object)
    return nodes, elements, region_tags


def laplacian_smooth_mesh(nodes: np.ndarray, elements: np.ndarray,
                          boundary_mask: np.ndarray, n_iter: int = 10) -> np.ndarray:
    new_nodes = nodes.copy()
    n = len(nodes)

    adj = [set() for _ in range(n)]
    for tri in elements:
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            adj[a].add(b)
            adj[b].add(a)

    for _ in range(n_iter):
        for i in range(n):
            if boundary_mask[i]:
                continue
            if len(adj[i]) == 0:
                continue
            neighbor_sum = np.zeros(2, dtype=float)
            for j in adj[i]:
                neighbor_sum += new_nodes[j]
            new_nodes[i] = neighbor_sum / len(adj[i])
    return new_nodes


def compute_element_quality(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n_elem = len(elements)
    quality = np.zeros(n_elem, dtype=float)
    for e in range(n_elem):
        tri = elements[e]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)

        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        denom = a * a + b * b + c * c
        if denom > 1e-14:
            quality[e] = 4.0 * np.sqrt(3.0) * area / denom
        else:
            quality[e] = 0.0
    return quality


def build_boundary_mask(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n = len(nodes)
    edge_count = {}
    for tri in elements:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
    mask = np.zeros(n, dtype=bool)
    for (a, b), count in edge_count.items():
        if count == 1:
            mask[a] = True
            mask[b] = True
    return mask


def fibonacci_seeding_2d(n_points: int, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    x_min, x_max, y_min, y_max = bbox
    points = np.zeros((n_points, 2), dtype=float)
    for i in range(n_points):
        r = np.sqrt((i + 0.5) / n_points)
        theta = 2.0 * np.pi * i / phi
        points[i, 0] = x_min + (x_max - x_min) * (0.5 + r * np.cos(theta) * 0.5)
        points[i, 1] = y_min + (y_max - y_min) * (0.5 + r * np.sin(theta) * 0.5)
    return points
