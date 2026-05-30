
import numpy as np
from typing import List, Tuple


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    v0, v1, v2 = np.asarray(v0), np.asarray(v1), np.asarray(v2)
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross)


def triangle_centroid(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return (np.asarray(v0) + np.asarray(v1) + np.asarray(v2)) / 3.0


def barycentric_coordinates(p: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                            v2: np.ndarray) -> Tuple[float, float, float]:
    v0, v1, v2, p = np.asarray(v0), np.asarray(v1), np.asarray(v2), np.asarray(p)
    A_total = triangle_area(v0, v1, v2)
    if A_total < 1e-20:
        return 1.0, 0.0, 0.0
    A0 = triangle_area(p, v1, v2)
    A1 = triangle_area(v0, p, v2)
    A2 = triangle_area(v0, v1, p)
    return A0 / A_total, A1 / A_total, A2 / A_total


def point_in_triangle(p: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                      v2: np.ndarray, tol: float = 1e-10) -> bool:
    l0, l1, l2 = barycentric_coordinates(p, v0, v1, v2)
    return (l0 >= -tol) and (l1 >= -tol) and (l2 >= -tol) and abs(l0 + l1 + l2 - 1.0) < tol


def integrate_over_triangles(vertices_list: List[np.ndarray],
                              flux_func) -> float:
    total = 0.0
    for verts in vertices_list:
        v0, v1, v2 = verts[0], verts[1], verts[2]
        A = triangle_area(v0, v1, v2)
        if A < 1e-20:
            continue
        centroid = triangle_centroid(v0, v1, v2)
        try:
            flux_val = flux_func(centroid)
            if np.isfinite(flux_val):
                total += A * flux_val
        except Exception:
            continue
    return total


def regular_surface_mesh(nx: int, ny: int, xlim: Tuple[float, float],
                         ylim: Tuple[float, float]) -> List[np.ndarray]:
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = np.array([x[i], y[j], 0.0])
            p10 = np.array([x[i+1], y[j], 0.0])
            p01 = np.array([x[i], y[j+1], 0.0])
            p11 = np.array([x[i+1], y[j+1], 0.0])
            triangles.append(np.array([p00, p10, p11]))
            triangles.append(np.array([p00, p11, p01]))
    return triangles


def surface_sensible_heat_flux(t_sfc: float, t_air: float,
                                wind_speed: float,
                                drag_coeff: float = 1.2e-3,
                                rho: float = 1.225,
                                cp: float = 1004.0) -> float:
    return rho * cp * drag_coeff * wind_speed * (t_sfc - t_air)


def surface_latent_heat_flux(q_sfc: float, q_air: float,
                              wind_speed: float,
                              drag_coeff: float = 1.2e-3,
                              rho: float = 1.225,
                              Lv: float = 2.501e6) -> float:
    return rho * Lv * drag_coeff * wind_speed * (q_sfc - q_air)
