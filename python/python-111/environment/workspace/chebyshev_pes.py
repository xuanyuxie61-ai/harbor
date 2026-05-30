
import numpy as np
from typing import Callable


def chebyshev_zeros(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    j = np.arange(1, n + 1)
    zeros = np.cos(np.pi * (2.0 * j - 1.0) / (2.0 * n))
    return zeros


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    if a >= b:
        raise ValueError("Require a < b")
    if n < 1:
        raise ValueError("n must be at least 1")
    
    z = chebyshev_zeros(n)

    x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * z
    f_vals = f(x_nodes)
    
    coeffs = np.zeros(n)
    for j in range(n):
        angle = np.pi * j * (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
        coeffs[j] = (2.0 / n) * np.sum(f_vals * np.cos(angle))
    return coeffs


def chebyshev_interpolant(a: float, b: float, n: int, coeffs: np.ndarray,
                          x_query: np.ndarray) -> np.ndarray:
    if a >= b:
        raise ValueError("Require a < b")
    if coeffs.size != n:
        raise ValueError("coeffs length must equal n")
    






    raise NotImplementedError("Hole 1: 请补全 Clenshaw 递推算法")


def chebyshev_derivative(a: float, b: float, n: int, coeffs: np.ndarray,
                         x_query: np.ndarray) -> np.ndarray:
    if n < 2:
        return np.zeros_like(x_query)
    
    dc = np.zeros(n)
    dc[n - 1] = 0.0
    dc[n - 2] = 2.0 * (n - 1) * coeffs[n - 1]
    for k in range(n - 3, -1, -1):
        dc[k] = dc[k + 2] + 2.0 * (k + 1) * coeffs[k + 1]
    

    scale = 2.0 / (b - a)
    return scale * chebyshev_interpolant(a, b, n, dc, x_query)


def fit_free_energy_profile(coordinate_values: np.ndarray, free_energy: np.ndarray,
                            order: int = 16) -> tuple:
    if len(coordinate_values) != len(free_energy):
        raise ValueError("Lengths of coordinate_values and free_energy must match")
    a = float(coordinate_values.min())
    b = float(coordinate_values.max())
    if a >= b:
        raise ValueError("Invalid coordinate range")
    

    def f_interp(x):
        return np.interp(x, coordinate_values, free_energy)
    
    coeffs = chebyshev_coefficients(a, b, order, f_interp)
    return a, b, coeffs


def approximate_potential_energy_surface_2d(x_vals: np.ndarray, y_vals: np.ndarray,
                                            energy_grid: np.ndarray,
                                            nx_cheb: int = 12, ny_cheb: int = 12) -> tuple:
    ax, bx = float(x_vals.min()), float(x_vals.max())
    ay, by = float(y_vals.min()), float(y_vals.max())
    

    zx = chebyshev_zeros(nx_cheb)
    zy = chebyshev_zeros(ny_cheb)
    x_nodes = 0.5 * (ax + bx) + 0.5 * (bx - ax) * zx
    y_nodes = 0.5 * (ay + by) + 0.5 * (by - ay) * zy
    
    from scipy.interpolate import RectBivariateSpline

    spline = RectBivariateSpline(x_vals, y_vals, energy_grid, kx=1, ky=1)
    sample_grid = spline.ev(x_nodes[:, None], y_nodes[None, :])
    
    coeffs_2d = np.zeros((nx_cheb, ny_cheb))
    for i in range(nx_cheb):
        angle_x = np.pi * i * (2.0 * np.arange(1, nx_cheb + 1) - 1.0) / (2.0 * nx_cheb)
        cx = (2.0 / nx_cheb) * np.cos(angle_x)
        for j in range(ny_cheb):
            angle_y = np.pi * j * (2.0 * np.arange(1, ny_cheb + 1) - 1.0) / (2.0 * ny_cheb)
            cy = (2.0 / ny_cheb) * np.cos(angle_y)
            coeffs_2d[i, j] = np.sum(sample_grid * cx[:, None] * cy[None, :])
    return ax, bx, ay, by, coeffs_2d
