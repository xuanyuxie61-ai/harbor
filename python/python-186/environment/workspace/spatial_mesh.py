
import numpy as np
from typing import Tuple, List, Optional


def generate_2d_triangular_mesh(nx: int, ny: int,
                                x_min: float = 0.0, x_max: float = 1.0,
                                y_min: float = 0.0, y_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y

    nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            idx = j * n_nodes_x + i
            nodes[idx, 0] = x_min + i * hx
            nodes[idx, 1] = y_min + j * hy

    n_elements = 2 * nx * ny
    elements = np.zeros((n_elements, 3), dtype=np.int32)

    e = 0
    for j in range(ny):
        for i in range(nx):
            n1 = j * n_nodes_x + i
            n2 = j * n_nodes_x + (i + 1)
            n3 = (j + 1) * n_nodes_x + i
            n4 = (j + 1) * n_nodes_x + (i + 1)

            elements[e, :] = [n1, n2, n3]
            elements[e + 1, :] = [n2, n4, n3]
            e += 2

    return nodes, elements


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def barycentric_coordinates(p: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    A = triangle_area(p1, p2, p3)
    if A < 1e-14:
        return -1.0, -1.0, -1.0

    lambda1 = triangle_area(p, p2, p3) / A
    lambda2 = triangle_area(p1, p, p3) / A
    lambda3 = 1.0 - lambda1 - lambda2

    return lambda1, lambda2, lambda3


def locate_point_in_mesh(nodes: np.ndarray, elements: np.ndarray,
                         point: np.ndarray) -> int:
    n_elements = elements.shape[0]
    best_elem = -1
    best_min_lambda = -1.0

    for e in range(n_elements):
        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        l1, l2, l3 = barycentric_coordinates(point, p1, p2, p3)
        min_lambda = min(l1, l2, l3)

        if min_lambda >= -1e-10:
            return e


        if min_lambda > best_min_lambda:
            best_min_lambda = min_lambda
            best_elem = e

    return best_elem


def refine_mesh_midpoint(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]


    edges = []
    edge_map = {}

    for e in range(n_elements):
        tri = elements[e, :]
        edge_pairs = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edge_pairs:
            key = tuple(sorted((int(a), int(b))))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)

    n_edges = len(edges)
    new_n_nodes = n_nodes + n_edges
    new_nodes = np.zeros((new_n_nodes, 2), dtype=np.float64)
    new_nodes[:n_nodes, :] = nodes


    mid_node_idx = {}
    for idx, (a, b) in enumerate(edges):
        new_nodes[n_nodes + idx, :] = 0.5 * (nodes[a, :] + nodes[b, :])
        mid_node_idx[(a, b)] = n_nodes + idx


    new_elements = np.zeros((4 * n_elements, 3), dtype=np.int32)

    for e in range(n_elements):
        tri = elements[e, :]
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])

        m01 = mid_node_idx[tuple(sorted((v0, v1)))]
        m12 = mid_node_idx[tuple(sorted((v1, v2)))]
        m20 = mid_node_idx[tuple(sorted((v2, v0)))]

        new_elements[4 * e + 0, :] = [v0, m01, m20]
        new_elements[4 * e + 1, :] = [m01, v1, m12]
        new_elements[4 * e + 2, :] = [m20, m12, v2]
        new_elements[4 * e + 3, :] = [m01, m12, m20]

    return new_nodes, new_elements


def fem_basis_t3(xi: float, eta: float) -> np.ndarray:
    N = np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)
    return N


def fem_sample_on_mesh(nodes: np.ndarray, elements: np.ndarray,
                       node_values: np.ndarray,
                       sample_points: np.ndarray) -> np.ndarray:
    n_samples = sample_points.shape[0]
    sample_values = np.zeros(n_samples, dtype=np.float64)

    for s in range(n_samples):
        p = sample_points[s, :]
        e = locate_point_in_mesh(nodes, elements, p)

        if e < 0:

            dists = np.sum((nodes - p)**2, axis=1)
            nearest = int(np.argmin(dists))
            sample_values[s] = node_values[nearest]
            continue

        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        l1, l2, l3 = barycentric_coordinates(p, p1, p2, p3)
        sample_values[s] = l1 * node_values[n1] + l2 * node_values[n2] + l3 * node_values[n3]

    return sample_values


def spatial_diffusion_operator(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for e in range(elements.shape[0]):
        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        A = triangle_area(p1, p2, p3)
        if A < 1e-14:
            continue


        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3





        dNdx = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * A)
        dNdy = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * A)


        Ke = A * (np.outer(dNdx, dNdx) + np.outer(dNdy, dNdy))


        idx = [n1, n2, n3]
        for i_loc in range(3):
            for j_loc in range(3):
                K[idx[i_loc], idx[j_loc]] += Ke[i_loc, j_loc]

    return K
