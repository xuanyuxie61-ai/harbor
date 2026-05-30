
import numpy as np
from typing import Callable, Optional, Tuple


def cvt_lloyd_1d(n: int,
                 a: float,
                 b: float,
                 density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 it_num: int = 100,
                 tol: float = 1e-10) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2")
    if b <= a:
        raise ValueError("b must be > a")
    if it_num < 1:
        raise ValueError("it_num must be >= 1")

    if density is None:
        density = lambda x: np.ones_like(x, dtype=np.float64)


    x = np.linspace(a, b, n, dtype=np.float64)

    for it in range(it_num):

        boundaries = np.zeros(n + 1, dtype=np.float64)
        boundaries[0] = a
        boundaries[-1] = b
        boundaries[1:-1] = 0.5 * (x[:-1] + x[1:])

        x_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            left = boundaries[i]
            right = boundaries[i + 1]

            nquad = max(5, int(100 * (right - left) / (b - a)) + 1)
            t = np.linspace(left, right, nquad)
            w = np.ones(nquad, dtype=np.float64)
            w[0] = 0.5
            w[-1] = 0.5
            w[1:-1:2] = 2.0
            w[2:-1:2] = 4.0
            w *= (right - left) / (3.0 * (nquad - 1) // 2 * 2) if (nquad % 2 == 1) else (right - left) / (nquad - 1)
            if nquad % 2 == 0:

                w = np.ones(nquad, dtype=np.float64)
                w[0] = 0.5
                w[-1] = 0.5
                w *= (right - left) / (nquad - 1)

            rho_vals = density(t)

            rho_vals = np.clip(rho_vals, 1e-12, None)
            mass = np.sum(w * rho_vals)
            moment = np.sum(w * t * rho_vals)
            x_new[i] = moment / mass if mass > 0 else 0.5 * (left + right)


        x_new = np.clip(np.sort(x_new), a, b)

        diff = np.max(np.abs(x_new - x))
        x = x_new
        if diff < tol:
            break

    return x


def tetrahedron_grid_count(n: int) -> int:
    if n < 0:
        return 0
    return (n + 3) * (n + 2) * (n + 1) // 6


def tetrahedron_grid(n: int, vertices: np.ndarray) -> np.ndarray:
    if vertices.shape != (3, 4):
        raise ValueError("vertices must have shape (3, 4)")
    ng = tetrahedron_grid_count(n)
    tg = np.zeros((3, ng), dtype=np.float64)
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                coeff = np.array([i, j, k, l], dtype=np.float64) / n
                tg[:, p] = vertices @ coeff
                p += 1
    return tg


def fibonacci_spiral_disk(n: int, R: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi / (phi ** 2)

    k = np.arange(n, dtype=np.float64)
    r = R * np.sqrt(k / (n - 0.5))
    theta = k * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def adaptive_density_function(x: np.ndarray,
                              steepness: float = 10.0,
                              center: float = 0.5) -> np.ndarray:
    dx = np.abs(x - center)
    rho = np.exp(-steepness * dx) + 0.1
    return rho


def generate_composite_mesh_1d(n_base: int = 65,
                               a: float = 0.0,
                               b: float = 1.0,
                               steepness: float = 12.0,
                               center: float = 0.5) -> np.ndarray:
    density = lambda x: adaptive_density_function(x, steepness=steepness, center=center)
    cvt_pts = cvt_lloyd_1d(n_base, a, b, density=density, it_num=80, tol=1e-12)
    return cvt_pts
