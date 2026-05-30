
import numpy as np
from typing import Tuple


def triangulation_t3_to_t4(nodes: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必须是形状为 (n_tri, 3) 的数组")

    n_nodes_t3 = len(nodes)
    n_tri = len(triangles)


    nodes_t4 = np.zeros((n_nodes_t3 + n_tri, 2), dtype=float)
    nodes_t4[:n_nodes_t3] = nodes


    triangles_t4 = np.zeros((n_tri, 4), dtype=int)
    triangles_t4[:, :3] = triangles


    node_count = n_nodes_t3
    for t in range(n_tri):
        idx = triangles[t]

        if np.any(idx < 0) or np.any(idx >= n_nodes_t3):
            raise ValueError(f"三角形 {t} 包含越界节点索引")


        centroid = np.mean(nodes[idx], axis=0)
        nodes_t4[node_count] = centroid
        triangles_t4[t, 3] = node_count
        node_count += 1

    return nodes_t4, triangles_t4


def t4_shape_functions(xi: float, eta: float) -> np.ndarray:
    if xi < -1e-10 or eta < -1e-10 or xi + eta > 1.0 + 1e-10:
        raise ValueError(f"参考坐标 ({xi}, {eta}) 超出参考三角形")

    N = np.zeros(4, dtype=float)
    N[0] = 1.0 - xi - eta
    N[1] = xi
    N[2] = eta
    N[3] = 27.0 * xi * eta * (1.0 - xi - eta)
    return N


def t4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    if xi < -1e-10 or eta < -1e-10 or xi + eta > 1.0 + 1e-10:
        raise ValueError(f"参考坐标 ({xi}, {eta}) 超出参考三角形")

    dN_dxi = np.zeros(4, dtype=float)
    dN_deta = np.zeros(4, dtype=float)

    dN_dxi[0] = -1.0
    dN_dxi[1] = 1.0
    dN_dxi[2] = 0.0
    dN_dxi[3] = 27.0 * eta * (1.0 - 2.0 * xi - eta)

    dN_deta[0] = -1.0
    dN_deta[1] = 0.0
    dN_deta[2] = 1.0
    dN_deta[3] = 27.0 * xi * (1.0 - xi - 2.0 * eta)

    return dN_dxi, dN_deta


def interpolate_on_t4_mesh(nodes_t4: np.ndarray, triangles_t4: np.ndarray,
                           node_values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    node_values = np.asarray(node_values, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    results = np.zeros(len(query_points), dtype=float)

    for q in range(len(query_points)):
        pt = query_points[q]
        found = False

        for t in range(len(triangles_t4)):
            tri = triangles_t4[t]
            p = nodes_t4[tri[:3]]


            A = triangle_area(p[0], p[1], p[2])
            if abs(A) < 1e-14:
                continue

            w0 = triangle_area(pt, p[1], p[2]) / A
            w1 = triangle_area(pt, p[2], p[0]) / A
            w2 = 1.0 - w0 - w1

            if w0 >= -1e-10 and w1 >= -1e-10 and w2 >= -1e-10:


                xi = w1
                eta = w2
                N = t4_shape_functions(xi, eta)
                vals = node_values[tri]
                results[q] = np.dot(N, vals)
                found = True
                break

        if not found:

            dists = np.linalg.norm(nodes_t4 - pt, axis=1)
            results[q] = node_values[np.argmin(dists)]

    return results


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
