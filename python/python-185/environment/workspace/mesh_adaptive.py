
import numpy as np
from typing import Tuple, List


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def arc_cosine_safe(c: float) -> float:
    c2 = max(-1.0, min(1.0, float(c)))
    return np.arccos(c2)


def alpha_measure_single(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    a_len = np.linalg.norm(p2 - p3)
    b_len = np.linalg.norm(p1 - p3)
    c_len = np.linalg.norm(p1 - p2)

    if a_len < 1e-14 and b_len < 1e-14 and c_len < 1e-14:
        return 1.0

    if c_len < 1e-14 or b_len < 1e-14:
        angle_a = np.pi
    else:
        cos_a = (b_len ** 2 + c_len ** 2 - a_len ** 2) / (2.0 * b_len * c_len)
        angle_a = arc_cosine_safe(cos_a)

    if a_len < 1e-14 or c_len < 1e-14:
        angle_b = np.pi
    else:
        cos_b = (a_len ** 2 + c_len ** 2 - b_len ** 2) / (2.0 * a_len * c_len)
        angle_b = arc_cosine_safe(cos_b)

    if a_len < 1e-14 or b_len < 1e-14:
        angle_c = np.pi
    else:
        cos_c = (a_len ** 2 + b_len ** 2 - c_len ** 2) / (2.0 * a_len * b_len)
        angle_c = arc_cosine_safe(cos_c)

    min_angle = min(angle_a, angle_b, angle_c)
    alpha = min_angle * 3.0 / np.pi
    return max(0.0, min(1.0, alpha))


def q_measure_single(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    a_len = np.linalg.norm(p2 - p3)
    b_len = np.linalg.norm(p1 - p3)
    c_len = np.linalg.norm(p1 - p2)

    area = abs(triangle_area(p1, p2, p3))
    if area < 1e-14:
        return 0.0

    r_in = 2.0 * area / (a_len + b_len + c_len)
    r_out = a_len * b_len * c_len / (4.0 * area)

    if r_out < 1e-14:
        return 0.0

    q = 2.0 * r_in / r_out
    return max(0.0, min(1.0, q))


def evaluate_triangulation_quality(nodes: np.ndarray, triangles: np.ndarray) -> dict:
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be of shape (n_tri, 3)")

    n_tri = triangles.shape[0]
    alpha_values = np.zeros(n_tri, dtype=float)
    q_values = np.zeros(n_tri, dtype=float)
    areas = np.zeros(n_tri, dtype=float)

    for t in range(n_tri):
        idx = triangles[t]
        if np.any(idx < 0) or np.any(idx >= len(nodes)):
            raise ValueError(f"Triangle {t} contains out-of-bounds node index")

        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        alpha_values[t] = alpha_measure_single(p1, p2, p3)
        q_values[t] = q_measure_single(p1, p2, p3)
        areas[t] = abs(triangle_area(p1, p2, p3))

    area_total = np.sum(areas)
    if area_total < 1e-14:
        area_weights = np.ones(n_tri) / n_tri
    else:
        area_weights = areas / area_total

    result = {
        'alpha_min': float(np.min(alpha_values)),
        'alpha_ave': float(np.mean(alpha_values)),
        'alpha_area': float(np.sum(alpha_values * area_weights)),
        'q_min': float(np.min(q_values)),
        'q_ave': float(np.mean(q_values)),
        'q_area': float(np.sum(q_values * area_weights)),
        'area_min': float(np.min(areas)) if n_tri > 0 else 0.0,
        'area_max': float(np.max(areas)) if n_tri > 0 else 0.0,
        'area_ratio': float(np.min(areas) / np.max(areas)) if np.max(areas) > 1e-14 else 0.0,
        'area_ave': float(np.mean(areas)),
    }
    return result


def generate_uniform_triangulation(width: float, height: float,
                                   nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")

    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    X, Y = np.meshgrid(x, y)

    nodes = np.column_stack([X.ravel(), Y.ravel()])

    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + (i + 1)
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + (i + 1)
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    return nodes, np.array(triangles, dtype=int)


def adaptive_refinement_by_gradient(image: np.ndarray, nodes: np.ndarray,
                                    triangles: np.ndarray,
                                    quality_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=float)
    H, W = image.shape

    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    refine_mask = np.zeros(len(triangles), dtype=bool)
    for t in range(len(triangles)):
        idx = triangles[t]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        q = q_measure_single(p1, p2, p3)

        cx = (p1[0] + p2[0] + p3[0]) / 3.0
        cy = (p1[1] + p2[1] + p3[1]) / 3.0

        ix = int(np.clip(cx / W * (W - 1), 0, W - 1))
        iy = int(np.clip(cy / H * (H - 1), 0, H - 1))
        g = grad_mag[iy, ix]

        if q < quality_threshold and g > np.mean(grad_mag):
            refine_mask[t] = True

    if not np.any(refine_mask):
        return nodes, triangles

    new_nodes = [nodes.copy()]
    node_offset = len(nodes)
    new_triangles = []

    for t in range(len(triangles)):
        if refine_mask[t]:
            idx = triangles[t]
            p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
            centroid = (p1 + p2 + p3) / 3.0
            new_nodes.append(centroid.reshape(1, 2))
            new_idx = node_offset
            node_offset += 1
            new_triangles.append([idx[0], idx[1], new_idx])
            new_triangles.append([idx[1], idx[2], new_idx])
            new_triangles.append([idx[2], idx[0], new_idx])
        else:
            new_triangles.append(triangles[t].tolist())

    all_nodes = np.vstack(new_nodes)
    all_triangles = np.array(new_triangles, dtype=int)
    return all_nodes, all_triangles
