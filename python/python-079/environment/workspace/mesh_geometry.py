
import numpy as np
from typing import Tuple, List, Optional
from utils import arc_cosine_safe, gamma_func






def line_cvt_lloyd_step(
    n: int, a: float, b: float, x: np.ndarray, constrained: bool = True
) -> np.ndarray:
    if n < 2:
        raise ValueError("节点数 n 至少为 2")
    x = np.asarray(x, dtype=float).copy()
    x = np.sort(x)
    if constrained:
        x[0] = a
        x[-1] = b
    x_new = x.copy()
    for j in range(1, n - 1):
        x_new[j] = 0.25 * (x[j - 1] + 2.0 * x[j] + x[j + 1])
    if constrained:
        x_new[0] = a
        x_new[-1] = b
    return x_new


def line_cvt_energy(n: int, a: float, b: float, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    if x[0] < a - 1e-12 or x[-1] > b + 1e-12:
        raise ValueError("节点超出区间 [a, b]")
    energy = 0.0
    for j in range(n):
        if j == 0:
            ml = a
        else:
            ml = 0.5 * (x[j - 1] + x[j])
        if j == n - 1:
            mr = b
        else:
            mr = 0.5 * (x[j] + x[j + 1])
        energy += ((x[j] - ml) ** 3.0 + (mr - x[j]) ** 3.0) / 3.0
    return energy


def generate_cvt_nodes_1d(
    n: int,
    a: float,
    b: float,
    n_iter: int = 200,
    constrained: bool = True,
) -> np.ndarray:
    if n < 2:
        raise ValueError("节点数 n 至少为 2")
    if a >= b:
        raise ValueError("区间左端点 a 必须小于右端点 b")
    x = np.linspace(a, b, n)
    for _ in range(n_iter):
        x_new = line_cvt_lloyd_step(n, a, b, x, constrained=constrained)
        if np.max(np.abs(x_new - x)) < 1e-12:
            break
        x = x_new
    return x






def polygon_area_2d(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("v 必须是 n×2 的顶点坐标数组")
    n = v.shape[0]
    if n < 3:
        return 0.0
    x = v[:, 0]
    y = v[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return float(area)


def polygon_centroid_2d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    area = polygon_area_2d(v)
    if abs(area) < 1e-15:
        return np.zeros(2)
    n = v.shape[0]
    x = v[:, 0]
    y = v[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def polygon_moment_integral(v: np.ndarray, p: int, q: int) -> float:
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    if n < 3:
        return 0.0
    integral = 0.0
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1



        edge_int = 0.0
        for k in range(p + 1):
            for l in range(q + 1):
                coeff = (
                    _binomial(p, k)
                    * _binomial(q, l)
                    * (x1 ** (p - k))
                    * (dx ** k)
                    * (y1 ** (q - l))
                    * (dy ** l)
                    * (x1 * dy - y1 * dx)
                    / (k + l + 1)
                )
                edge_int += coeff
        integral += edge_int
    return abs(integral)


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res


def polygon_second_moments(v: np.ndarray) -> Tuple[float, float, float]:
    I_xx = polygon_moment_integral(v, 0, 2)
    I_yy = polygon_moment_integral(v, 2, 0)
    I_xy = polygon_moment_integral(v, 1, 1)
    return I_xx, I_yy, I_xy


def polygon_solid_angle_3d(
    poly: np.ndarray, p: np.ndarray
) -> float:
    poly = np.asarray(poly, dtype=float)
    p = np.asarray(p, dtype=float)
    n = poly.shape[0]
    if n < 3:
        return 0.0

    u = poly - p
    norms = np.linalg.norm(u, axis=1)
    if np.any(norms < 1e-15):
        return 0.0
    u = u / norms[:, np.newaxis]

    angle_sum = 0.0
    for i in range(n):
        v1 = u[(i - 1) % n]
        v2 = u[i]
        v3 = u[(i + 1) % n]

        n1 = np.cross(v2, v1)
        n2 = np.cross(v2, v3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-15 or n2_norm < 1e-15:
            continue
        cos_alpha = np.dot(n1, n2) / (n1_norm * n2_norm)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        angle_sum += np.arccos(cos_alpha)
    solid_angle = angle_sum - (n - 2) * np.pi
    return solid_angle


def polygon_triangulate_earclip(
    x: np.ndarray, y: np.ndarray
) -> List[Tuple[int, int, int]]:
    n = len(x)
    if n < 3:
        return []
    indices = list(range(n))
    triangles = []
    while len(indices) > 3:
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            if _is_convex(x, y, i_prev, i_curr, i_next, indices):
                if _is_ear(x, y, i_prev, i_curr, i_next, indices):
                    triangles.append((i_prev, i_curr, i_next))
                    indices.pop(i)
                    ear_found = True
                    break
        if not ear_found:

            triangles.append((indices[0], indices[1], indices[2]))
            indices.pop(1)
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _is_convex(
    x: np.ndarray, y: np.ndarray, i: int, j: int, k: int, active: List[int]
) -> bool:
    cross = (x[j] - x[i]) * (y[k] - y[j]) - (y[j] - y[i]) * (x[k] - x[j])
    return cross > 1e-12


def _is_ear(
    x: np.ndarray, y: np.ndarray, i: int, j: int, k: int, active: List[int]
) -> bool:
    for idx in active:
        if idx in (i, j, k):
            continue
        if _point_in_triangle(x[idx], y[idx], x[i], y[i], x[j], y[j], x[k], y[k]):
            return False
    return True


def _point_in_triangle(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> bool:
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-15:
        return False
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    c = 1.0 - a - b
    return (a >= -1e-12) and (b >= -1e-12) and (c >= -1e-12)






def triangle_angles_2d(
    t: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    a = np.linalg.norm(t[1] - t[2])
    b = np.linalg.norm(t[0] - t[2])
    c = np.linalg.norm(t[0] - t[1])
    angles = np.zeros(3)

    eps = 1e-12
    if a < eps or b < eps or c < eps:
        return angles
    cos_a = max(-1.0, min(1.0, (b * b + c * c - a * a) / (2.0 * b * c)))
    cos_b = max(-1.0, min(1.0, (a * a + c * c - b * b) / (2.0 * a * c)))
    angles[0] = arc_cosine_safe(cos_a)
    angles[1] = arc_cosine_safe(cos_b)
    angles[2] = np.pi - angles[0] - angles[1]
    return angles


def triangulation_delaunay_discrepancy(
    nodes: np.ndarray, triangles: List[Tuple[int, int, int]]
) -> float:
    if len(triangles) < 2:
        return 0.0

    edge_to_tri = {}
    for ti, tri in enumerate(triangles):
        edges = [(min(tri[0], tri[1]), max(tri[0], tri[1])),
                 (min(tri[1], tri[2]), max(tri[1], tri[2])),
                 (min(tri[2], tri[0]), max(tri[2], tri[0]))]
        for e in edges:
            edge_to_tri.setdefault(e, []).append(ti)

    max_disc = 0.0
    checked = set()
    for e, tris in edge_to_tri.items():
        if len(tris) != 2:
            continue
        t1_idx, t2_idx = tris[0], tris[1]
        key = (min(t1_idx, t2_idx), max(t1_idx, t2_idx))
        if key in checked:
            continue
        checked.add(key)
        tri1 = triangles[t1_idx]
        tri2 = triangles[t2_idx]

        v1 = [v for v in tri1 if v not in e][0]
        v2 = [v for v in tri2 if v not in e][0]

        quad = [nodes[v1], nodes[e[0]], nodes[v2], nodes[e[1]]]


        cur_min = min(
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[e[0]], nodes[e[1]]]))),
            np.min(triangle_angles_2d(np.array([nodes[v2], nodes[e[0]], nodes[e[1]]]))),
        )
        alt_min = min(
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[v2], nodes[e[0]]]))),
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[v2], nodes[e[1]]]))),
        )
        max_disc = max(max_disc, alt_min - cur_min)
    return max_disc






def simplex01_volume(m: int) -> float:
    if m < 1:
        return 1.0
    vol = 1.0
    for k in range(1, m + 1):
        vol /= float(k)
    return vol


def simplex01_monomial_integral(m: int, e: np.ndarray) -> float:
    e = np.asarray(e, dtype=int)
    if len(e) != m:
        raise ValueError("指数数组长度必须与维度 m 一致")
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    numerator = 1.0
    for val in e:
        numerator *= gamma_func(float(val + 1))
    den_arg = float(m + np.sum(e) + 1)
    denominator = gamma_func(den_arg)
    return numerator / denominator


def triangle_exact_integral_fem(
    nodes: np.ndarray, exponents: List[Tuple[int, int]]
) -> List[float]:
    results = []
    for p, q in exponents:
        val = simplex01_monomial_integral(2, np.array([p, q]))
        results.append(val)
    return results






def compute_waterplane_properties(
    waterline_vertices: np.ndarray,
) -> dict:
    v = np.asarray(waterline_vertices, dtype=float)
    area = abs(polygon_area_2d(v))
    centroid = polygon_centroid_2d(v)
    I_xx, I_yy, I_xy = polygon_second_moments(v)


    draft = 20.0
    V_disp = area * draft
    BM_t = I_xx / V_disp if V_disp > 1e-12 else 0.0
    BM_l = I_yy / V_disp if V_disp > 1e-12 else 0.0
    return {
        "area": area,
        "centroid": centroid,
        "I_xx": I_xx,
        "I_yy": I_yy,
        "I_xy": I_xy,
        "BM_t": BM_t,
        "BM_l": BM_l,
    }


def generate_platform_waterline(
    platform_type: str = "semi-submersible",
) -> np.ndarray:
    if platform_type == "semi-submersible":

        vertices = np.array(
            [
                [-40.0, -30.0],
                [-40.0, 30.0],
                [-15.0, 30.0],
                [-15.0, 10.0],
                [15.0, 10.0],
                [15.0, 30.0],
                [40.0, 30.0],
                [40.0, -30.0],
                [15.0, -30.0],
                [15.0, -10.0],
                [-15.0, -10.0],
                [-15.0, -30.0],
            ],
            dtype=float,
        )
    elif platform_type == "spar":

        theta = np.linspace(0, 2 * np.pi, 33, endpoint=False)
        r = 20.0
        vertices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    elif platform_type == "tension-leg":

        vertices = np.array(
            [[-35, -35], [35, -35], [35, 35], [-35, 35]], dtype=float
        )
    else:
        raise ValueError(f"未知的平台类型: {platform_type}")
    return vertices
