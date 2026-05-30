
import numpy as np
from typing import List, Tuple


def tetrahedron01_volume() -> float:
    return 1.0 / 6.0


def tetrahedron01_monomial_integral(e: Tuple[int, int, int]) -> float:
    e1, e2, e3 = e
    from math import factorial
    return factorial(e1) * factorial(e2) * factorial(e3) / factorial(e1 + e2 + e3 + 3)


def tetrahedron01_sample(n: int) -> np.ndarray:
    points = np.zeros((n, 3))
    for i in range(n):
        e1, e2, e3 = np.random.exponential(1.0, 3)
        s = e1 + e2 + e3
        if s < 1e-20:

            u = np.random.rand(3)
            u.sort()
            points[i] = [u[0], u[1] - u[0], u[2] - u[1]]
        else:
            points[i] = [e1 / s, e2 / s, e3 / s]
    return points


def map_to_physical_tetrahedron(points_ref: np.ndarray,
                                 vertices: np.ndarray) -> np.ndarray:
    v0 = vertices[0]
    J = np.column_stack([vertices[1] - v0,
                         vertices[2] - v0,
                         vertices[3] - v0])
    return v0 + points_ref @ J.T


def tetrahedron_physical_volume(vertices: np.ndarray) -> float:
    J = np.column_stack([vertices[1] - vertices[0],
                         vertices[2] - vertices[0],
                         vertices[3] - vertices[0]])
    return abs(np.linalg.det(J)) / 6.0


class PrecipitationVolumeEstimator:

    def __init__(self, n_samples_per_cell: int = 256):
        self.n_samples = n_samples_per_cell

    def estimate_cell_precipitation(self, vertices: np.ndarray,
                                     precip_rate_func) -> float:
        vol = tetrahedron_physical_volume(vertices)
        if vol < 1e-20:
            return 0.0

        ref_points = tetrahedron01_sample(self.n_samples)
        phys_points = map_to_physical_tetrahedron(ref_points, vertices)

        total = 0.0
        valid = 0
        for pt in phys_points:
            try:
                val = precip_rate_func(pt)
                if np.isfinite(val) and val >= 0.0:
                    total += val
                    valid += 1
            except Exception:
                continue

        if valid == 0:
            return 0.0
        return vol * total / valid

    def estimate_domain_precipitation(self, tetrahedra_vertices: List[np.ndarray],
                                       precip_rate_func) -> float:
        total = 0.0
        for verts in tetrahedra_vertices:
            total += self.estimate_cell_precipitation(verts, precip_rate_func)
        return total

    def estimate_from_gridded_field(self, precip_rate_2d: np.ndarray,
                                     dx: float, dy: float, dz_levels: np.ndarray) -> float:
        nz, ny, nx = precip_rate_2d.shape if precip_rate_2d.ndim == 3 else (1, *precip_rate_2d.shape)
        if precip_rate_2d.ndim == 2:
            precip_rate_2d = precip_rate_2d.reshape((1, ny, nx))

        total_precip = 0.0

        for k in range(nz):
            dz = dz_levels[k] if k < len(dz_levels) else dz_levels[-1] if len(dz_levels) > 0 else 1000.0
            for j in range(ny):
                for i in range(nx):
                    rate = precip_rate_2d[k, j, i]
                    if np.isfinite(rate) and rate > 0.0:
                        cell_vol = dx * dy * dz
                        total_precip += cell_vol * rate
        return total_precip
