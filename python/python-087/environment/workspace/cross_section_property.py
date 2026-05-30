
import numpy as np
from scipy.special import gamma as Gamma
from typing import Tuple, List


def regular_hexagon_vertices(R: float = 1.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts = np.column_stack((R * np.cos(angles), R * np.sin(angles)))
    return verts


def hexagon_area(R: float = 1.0) -> float:
    return 1.5 * np.sqrt(3.0) * R ** 2


def hexagon_uniform_sample(R: float = 1.0, n_samples: int = 1000) -> np.ndarray:
    verts = regular_hexagon_vertices(R)
    samples = np.zeros((n_samples, 2), dtype=np.float64)

    triangles = []
    for k in range(6):
        triangles.append(np.array([[0.0, 0.0], verts[k], verts[(k + 1) % 6]]))
    areas = []
    for tri in triangles:

        a = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        areas.append(a)
    areas = np.array(areas)
    probs = areas / areas.sum()
    for i in range(n_samples):

        t_idx = np.searchsorted(np.cumsum(probs), np.random.rand())
        tri = triangles[t_idx]

        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        alpha = 1.0 - sqrt_r1
        beta = sqrt_r1 * (1.0 - r2)
        gamma = sqrt_r1 * r2
        samples[i] = alpha * tri[0] + beta * tri[1] + gamma * tri[2]
    return samples


def honeycomb_cell_geometry(cell_size: float, wall_thickness: float,
                            n_rings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if cell_size <= 0 or wall_thickness <= 0:
        raise ValueError("cell_size 与 wall_thickness 必须为正")
    centers = []

    dx = 1.5 * cell_size
    dy = np.sqrt(3.0) * cell_size
    centers.append(np.array([0.0, 0.0]))
    for ring in range(1, n_rings + 1):
        for k in range(6):
            angle = k * np.pi / 3.0
            for step in range(ring):

                cx = ring * dx * np.cos(angle) + step * dx * np.cos(angle + 2 * np.pi / 3.0)
                cy = ring * dy * np.sin(angle) + step * dy * np.sin(angle + 2 * np.pi / 3.0)
                centers.append(np.array([cx, cy]))
    centers = np.array(centers)

    uniq = []
    for c in centers:
        if not any(np.linalg.norm(c - u) < 1e-9 for u in uniq):
            uniq.append(c)
    centers = np.array(uniq)
    verts_list = []
    for c in centers:
        verts = regular_hexagon_vertices(cell_size) + c
        verts_list.append(verts)
    verts_list = np.array(verts_list)
    return centers, verts_list


def hyperball_monomial_integral(dim: int, exponents: Tuple[int, ...],
                                radius: float = 1.0) -> float:
    exponents = tuple(int(e) for e in exponents)
    if len(exponents) != dim:
        raise ValueError("指数元组长度必须与维度一致")
    if any(e < 0 for e in exponents):
        raise ValueError("指数必须非负")
    if any(e % 2 == 1 for e in exponents):
        return 0.0
    S = sum(e + 1 for e in exponents)

    numerator = 1.0
    for e in exponents:
        numerator *= Gamma(0.5 * (e + 1))
    denominator = Gamma(0.5 * S)
    val = 2.0 * (radius ** S) / S * numerator / denominator
    return float(val)


def section_property_monte_carlo(polygon_vertices: np.ndarray,
                                  n_samples: int = 50000) -> dict:
    poly = np.asarray(polygon_vertices, dtype=np.float64)
    if poly.shape[1] != 2:
        raise ValueError("顶点必须为二维")

    ymin, ymax = poly[:, 0].min(), poly[:, 0].max()
    zmin, zmax = poly[:, 1].min(), poly[:, 1].max()
    area_bbox = (ymax - ymin) * (zmax - zmin)
    if area_bbox <= 0:
        raise ValueError("包围盒面积为零")

    def point_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
        n = len(poly)
        inside = np.zeros(len(points), dtype=bool)
        x, y = points[:, 0], points[:, 1]
        for i in range(n):
            j = (i + 1) % n
            xi, yi = poly[i]
            xj, yj = poly[j]

            intersect = ((yi > y) != (yj > y)) & \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
            inside ^= intersect
        return inside


    batch = n_samples
    pts = np.random.uniform([ymin, zmin], [ymax, zmax], size=(batch, 2))
    mask = point_in_polygon(pts, poly)
    accepted = pts[mask]
    n_in = accepted.shape[0]
    if n_in == 0:
        raise RuntimeError("Monte Carlo 采样全部落在多边形外，请检查顶点顺序或范围")
    area_est = area_bbox * n_in / batch
    centroid = accepted.mean(axis=0)
    I_z = area_est * np.mean(accepted[:, 0] ** 2)
    I_y = area_est * np.mean(accepted[:, 1] ** 2)
    J = I_y + I_z
    return {
        "area": float(area_est),
        "centroid_y": float(centroid[0]),
        "centroid_z": float(centroid[1]),
        "I_y": float(I_y),
        "I_z": float(I_z),
        "J": float(J),
        "n_accepted": int(n_in)
    }


def equivalent_honeycomb_properties(cell_size: float, wall_thickness: float,
                                    E_s: float, rho_s: float) -> dict:
    l = cell_size
    t = wall_thickness
    if t >= l:
        raise ValueError("壁厚不能大于等于边长")
    ratio = t / l
    rho_star = rho_s * (2.0 / np.sqrt(3.0)) * ratio * (1.0 - 0.5 * ratio)
    E_star = E_s * (4.0 / np.sqrt(3.0)) * (ratio ** 3) / (1.0 + 3.0 * ratio ** 2)
    G_star = E_s * (1.0 / np.sqrt(3.0)) * ratio / (1.0 + 3.0 * ratio ** 2)
    return {
        "rho_star": float(rho_star),
        "E_star": float(E_star),
        "G_star": float(G_star),
        "relative_density": float(rho_star / rho_s)
    }
