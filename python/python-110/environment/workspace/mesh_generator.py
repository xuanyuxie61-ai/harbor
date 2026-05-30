
import numpy as np
from typing import Tuple, List
from utils import validate_array_1d, validate_array_2d


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 0.5 * abs(cross)


def quadrilateral_area(quad_xy: np.ndarray) -> float:
    quad_xy = validate_array_2d(quad_xy, "quad_xy")
    if quad_xy.shape != (2, 4):
        raise ValueError("quad_xy must be 2x4 array")
    a1 = triangle_area(quad_xy[:, 0], quad_xy[:, 1], quad_xy[:, 2])
    a2 = triangle_area(quad_xy[:, 2], quad_xy[:, 3], quad_xy[:, 0])
    if a1 < 0 or a2 < 0:

        a1 = triangle_area(quad_xy[:, 1], quad_xy[:, 2], quad_xy[:, 3])
        a2 = triangle_area(quad_xy[:, 3], quad_xy[:, 0], quad_xy[:, 1])
    if a1 < 0 or a2 < 0:
        raise ValueError("Quadrilateral vertices seem degenerate or wrongly ordered")
    return a1 + a2


def reference_to_physical_q4(
    q4: np.ndarray, rs: np.ndarray
) -> np.ndarray:
    q4 = validate_array_2d(q4, "q4")
    rs = validate_array_2d(rs, "rs")
    if q4.shape[0] != 2 or q4.shape[1] != 4:
        raise ValueError("q4 must be 2x4")
    if rs.shape[0] != 2:
        raise ValueError("rs must have 2 rows")
    n = rs.shape[1]
    r = rs[0, :]
    s = rs[1, :]
    psi = np.zeros((4, n), dtype=float)
    psi[0, :] = (1.0 - r) * (1.0 - s)
    psi[1, :] = r * (1.0 - s)
    psi[2, :] = r * s
    psi[3, :] = (1.0 - r) * s
    xy = q4 @ psi
    return xy


def generate_circular_domain_nodes(
    R: float, n_r: int, n_theta: int
) -> Tuple[np.ndarray, np.ndarray]:
    if R <= 0:
        raise ValueError("Radius must be positive")
    if n_r < 2 or n_theta < 3:
        raise ValueError("Need at least 2 radial and 3 angular divisions")

    rho = np.linspace(0.0, 1.0, n_r) ** 0.8
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    nodes = []
    node_map = {}
    idx = 0
    for i, r_val in enumerate(rho):
        for j, th in enumerate(theta):
            x = r_val * R * np.cos(th)
            y = r_val * R * np.sin(th)
            nodes.append([x, y])
            node_map[(i, j)] = idx
            idx += 1
    nodes = np.array(nodes, dtype=float).T
    elements = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            e = [
                node_map[(i, j)],
                node_map[(i + 1, j)],
                node_map[(i + 1, j_next)],
                node_map[(i, j_next)],
            ]
            elements.append(e)
    elements = np.array(elements, dtype=int).T
    return nodes, elements


def compute_mesh_areas(
    nodes: np.ndarray, elements: np.ndarray
) -> Tuple[np.ndarray, float]:
    nodes = validate_array_2d(nodes, "nodes")
    elements = validate_array_2d(elements, "elements")
    if nodes.shape[0] != 2:
        raise ValueError("nodes must be 2 x N")
    if elements.shape[0] != 4:
        raise ValueError("elements must be 4 x N")
    n_elem = elements.shape[1]
    elem_areas = np.zeros(n_elem, dtype=float)
    for e in range(n_elem):
        q4 = np.zeros((2, 4), dtype=float)
        for k in range(4):
            node_idx = elements[k, e]
            if not (0 <= node_idx < nodes.shape[1]):
                raise ValueError(f"Invalid node index {node_idx} in element {e}")
            q4[:, k] = nodes[:, node_idx]
        elem_areas[e] = quadrilateral_area(q4)
    mesh_area = float(np.sum(elem_areas))
    return elem_areas, mesh_area


def node_to_element_average(
    node_values: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    node_values = validate_array_1d(node_values, "node_values")
    elements = validate_array_2d(elements, "elements")
    if elements.shape[0] != 4:
        raise ValueError("Only Q4 elements supported")
    n_elem = elements.shape[1]
    elem_values = np.zeros(n_elem, dtype=float)
    for e in range(n_elem):
        s = 0.0
        for k in range(4):
            idx = elements[k, e]
            if not (0 <= idx < node_values.size):
                raise ValueError(f"Node index {idx} out of bounds")
            s += node_values[idx]
        elem_values[e] = s / 4.0
    return elem_values


def sample_quadrilateral(
    quad_xy: np.ndarray, n_sample: int = 1
) -> np.ndarray:
    quad_xy = validate_array_2d(quad_xy, "quad_xy")
    if quad_xy.shape != (2, 4):
        raise ValueError("quad_xy must be 2x4")
    pts = np.zeros((2, n_sample), dtype=float)
    for i in range(n_sample):

        a1 = triangle_area(quad_xy[:, 0], quad_xy[:, 1], quad_xy[:, 2])
        a2 = triangle_area(quad_xy[:, 0], quad_xy[:, 2], quad_xy[:, 3])
        if a1 + a2 < 1e-20:
            pts[:, i] = quad_xy[:, 0]
            continue
        r = np.random.rand()
        if r < a1 / (a1 + a2):

            tri = quad_xy[:, [0, 1, 2]]
        else:

            tri = quad_xy[:, [0, 2, 3]]

        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        pts[:, i] = tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])
    return pts


def sample_q4_mesh(
    nodes: np.ndarray, elements: np.ndarray, sample_num: int
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = validate_array_2d(nodes, "nodes")
    elements = validate_array_2d(elements, "elements")
    elem_areas, _ = compute_mesh_areas(nodes, elements)
    n_elem = elements.shape[1]
    cum_area = np.zeros(n_elem + 1, dtype=float)
    cum_area[1:] = np.cumsum(elem_areas)
    total_area = cum_area[-1]
    if total_area < 1e-20:
        raise ValueError("Mesh has zero total area")
    cum_area /= total_area
    sample_xy = np.zeros((2, sample_num), dtype=float)
    sample_elem = np.zeros(sample_num, dtype=int)
    for s in range(sample_num):
        r = np.random.rand()

        left, right = 0, n_elem
        while right - left > 1:
            mid = (left + right) // 2
            if cum_area[mid] < r:
                left = mid
            else:
                right = mid
        elem_idx = left
        q4 = np.zeros((2, 4), dtype=float)
        for k in range(4):
            q4[:, k] = nodes[:, elements[k, elem_idx]]
        pt = sample_quadrilateral(q4, 1)
        sample_xy[:, s] = pt[:, 0]
        sample_elem[s] = elem_idx
    return sample_xy, sample_elem


def area_estimate_grid_in_polygon(
    polygon: np.ndarray, n_grid: int = 256
) -> float:
    polygon = validate_array_2d(polygon, "polygon")
    if polygon.shape[0] != 2:
        raise ValueError("polygon must be 2 x N")
    x_min, x_max = np.min(polygon[0, :]), np.max(polygon[0, :])
    y_min, y_max = np.min(polygon[1, :]), np.max(polygon[1, :])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Degenerate polygon bounding box")
    dx = (x_max - x_min) / (n_grid + 1)
    dy = (y_max - y_min) / (n_grid + 1)
    gx = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, n_grid)
    gy = np.linspace(y_min + 0.5 * dy, y_max - 0.5 * dy, n_grid)
    X, Y = np.meshgrid(gx, gy)

    inside = points_in_polygon(X.ravel(), Y.ravel(), polygon)
    estimate = np.sum(inside) / (n_grid ** 2)
    return float(estimate)


def points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    poly_x = polygon[0, :]
    poly_y = polygon[1, :]
    n = poly_x.size
    inside = np.zeros(x.size, dtype=bool)
    for i in range(x.size):
        xi, yi = x[i], y[i]
        c = False
        for j in range(n):
            xj, yj = poly_x[j], poly_y[j]
            xk, yk = poly_x[(j + 1) % n], poly_y[(j + 1) % n]
            if ((yj > yi) != (yk > yi)) and (xi < (xk - xj) * (yi - yj) / (yk - yj + 1e-18) + xj):
                c = not c
        inside[i] = c
    return inside
