
import numpy as np
from typing import List, Tuple, Optional


def generate_triangular_mesh(
    nx: int, ny: int, domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    quadratic: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = domain
    n_vertex = nx * ny
    vertices = np.zeros((n_vertex, 2))

    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            vertices[idx, 0] = xmin + i * dx
            vertices[idx, 1] = ymin + j * dy


    n_elements = 2 * (nx - 1) * (ny - 1)
    lin_elements = np.zeros((n_elements, 3), dtype=int)

    e = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + (i + 1)
            n4 = (j + 1) * nx + i

            lin_elements[e, :] = [n1, n2, n3]
            e += 1
            lin_elements[e, :] = [n1, n3, n4]
            e += 1

    if not quadratic:

        boundary_nodes = []
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_nodes.append(j * nx + i)
        return vertices, lin_elements, np.array(boundary_nodes, dtype=int)



    edge_to_midnode = {}
    midnodes = []

    def get_edge_key(na, nb):
        return (min(na, nb), max(na, nb))

    for elem in lin_elements:

        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for na, nb in edges:
            ekey = get_edge_key(na, nb)
            if ekey not in edge_to_midnode:
                mid_coord = 0.5 * (vertices[na] + vertices[nb])
                mid_idx = n_vertex + len(midnodes)
                edge_to_midnode[ekey] = mid_idx
                midnodes.append(mid_coord)


    nodes = np.vstack([vertices, np.array(midnodes)])


    elements = np.zeros((n_elements, 6), dtype=int)
    for e_idx, elem in enumerate(lin_elements):

        elements[e_idx, 0] = elem[0]
        elements[e_idx, 1] = elem[1]
        elements[e_idx, 2] = elem[2]

        elements[e_idx, 3] = edge_to_midnode[get_edge_key(elem[0], elem[1])]
        elements[e_idx, 4] = edge_to_midnode[get_edge_key(elem[1], elem[2])]
        elements[e_idx, 5] = edge_to_midnode[get_edge_key(elem[2], elem[0])]


    boundary_nodes_set = set()
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes_set.add(j * nx + i)


    for e_idx, elem in enumerate(lin_elements):
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for na, nb in edges:

            xa, ya = vertices[na]
            xb, yb = vertices[nb]
            on_boundary = False
            if abs(xa - xmin) < 1e-10 and abs(xb - xmin) < 1e-10:
                on_boundary = True
            elif abs(xa - xmax) < 1e-10 and abs(xb - xmax) < 1e-10:
                on_boundary = True
            elif abs(ya - ymin) < 1e-10 and abs(yb - ymin) < 1e-10:
                on_boundary = True
            elif abs(ya - ymax) < 1e-10 and abs(yb - ymax) < 1e-10:
                on_boundary = True
            if on_boundary:
                boundary_nodes_set.add(edge_to_midnode[get_edge_key(na, nb)])

    boundary_nodes = np.array(sorted(boundary_nodes_set), dtype=int)

    return nodes, elements, boundary_nodes


def generate_equilateral_triangular_mesh(
    nx: int, ny: int, side_length: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = side_length
    sqrt3 = np.sqrt(3.0)


    n_nodes = nx * ny
    nodes = np.zeros((n_nodes, 2))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            x = i * L
            y = j * L * sqrt3 / 2.0
            if j % 2 == 1:
                x += L / 2.0
            nodes[idx, 0] = x
            nodes[idx, 1] = y


    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + (i + 1)
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + (i + 1)

            if j % 2 == 0:

                elements.append([n00, n10, n01])
                elements.append([n10, n11, n01])
            else:
                elements.append([n00, n11, n01])
                elements.append([n00, n10, n11])

    elements = np.array(elements, dtype=int)


    boundary_nodes = []
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes.append(j * nx + i)
    boundary_nodes = np.array(boundary_nodes, dtype=int)

    return nodes, elements, boundary_nodes


def triangle_area(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1])
        + p2[0] * (p3[1] - p1[1])
        + p3[0] * (p1[1] - p2[1])
    )
    return area


def barycentric_coordinates(
    p: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    A_total = triangle_area(p1, p2, p3)
    if A_total < 1e-15:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    lambda1 = triangle_area(p, p2, p3) / A_total
    lambda2 = triangle_area(p1, p, p3) / A_total
    lambda3 = triangle_area(p1, p2, p) / A_total


    s = lambda1 + lambda2 + lambda3
    if s > 0:
        return np.array([lambda1, lambda2, lambda3]) / s
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def is_point_in_triangle(
    p: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float = 1e-10
) -> bool:
    lam = barycentric_coordinates(p, p1, p2, p3)
    return np.all(lam >= -tol)


def compute_element_centroids(
    nodes: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    centroids = np.zeros((len(elements), 2))
    for e, elem in enumerate(elements):
        centroids[e] = (nodes[elem[0]] + nodes[elem[1]] + nodes[elem[2]]) / 3.0
    return centroids


def reverse_cuthill_mckee(
    adjacency: np.ndarray
) -> np.ndarray:
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    ordering = []


    degrees = np.sum(adjacency > 0, axis=1)
    start = int(np.argmin(degrees))

    queue = [start]
    visited[start] = True

    while queue:
        current = queue.pop(0)
        ordering.append(current)


        neighbors = np.where(adjacency[current] > 0)[0]
        unvisited = [v for v in neighbors if not visited[v]]
        unvisited.sort(key=lambda v: degrees[v])

        for v in unvisited:
            visited[v] = True
            queue.append(v)


    ordering = ordering[::-1]
    return np.array(ordering, dtype=int)


def build_adjacency_from_elements(
    n_nodes: int, elements: np.ndarray
) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for elem in elements:
        for i in range(3):
            for j in range(i + 1, 3):
                n1, n2 = elem[i], elem[j]
                adj[n1, n2] = 1
                adj[n2, n1] = 1
    return adj


def mesh_quality_metrics(
    nodes: np.ndarray, elements: np.ndarray
) -> dict:
    areas = []
    min_angles = []

    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        area = triangle_area(p1, p2, p3)
        areas.append(area)


        if a > 1e-14 and b > 1e-14 and c > 1e-14:
            cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
            angles = [
                np.arccos(np.clip(cos_alpha, -1, 1)),
                np.arccos(np.clip(cos_beta, -1, 1)),
                np.arccos(np.clip(cos_gamma, -1, 1)),
            ]
            min_angles.append(min(angles))

    areas = np.array(areas)
    min_angles = np.array(min_angles)

    return {
        "min_area": float(np.min(areas)) if len(areas) > 0 else 0.0,
        "max_area": float(np.max(areas)) if len(areas) > 0 else 0.0,
        "mean_area": float(np.mean(areas)) if len(areas) > 0 else 0.0,
        "min_angle_deg": float(np.degrees(np.min(min_angles))) if len(min_angles) > 0 else 0.0,
        "num_elements": len(elements),
    }
