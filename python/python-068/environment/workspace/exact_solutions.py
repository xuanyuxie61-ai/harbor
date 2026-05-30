
import numpy as np
from scipy.special import roots_hermite


def manufactured_population_field(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    field_id: int = 0
) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing='ij')

    if field_id == 0:

        cx = np.pi + 0.5 * np.sin(t)
        cy = np.pi + 0.3 * np.cos(t)
        return 50.0 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 2.0)
    elif field_id == 1:

        return 5.0 * (1.0 + np.sin(X - t) * np.cos(Y - 0.5 * t))
    elif field_id == 2:

        return 20.0 * np.exp(-((X - np.pi) ** 2 + (Y - np.pi) ** 2) / 4.0)
    elif field_id == 3:

        cx = np.pi - 0.5 * np.sin(t)
        cy = np.pi - 0.3 * np.cos(t)
        return 40.0 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 3.0)
    elif field_id == 4:

        return 4.0 * np.exp(-0.1 * t) * (1.0 + np.cos(X + t) * np.sin(Y + 0.3 * t))
    else:

        return 15.0 * np.exp(-((X - 1.5 * np.pi) ** 2 + (Y - 0.5 * np.pi) ** 2) / 5.0)


def manufactured_source_terms(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D: float = 0.01,
    vx: float = 0.1,
    vy: float = 0.05
) -> np.ndarray:


    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    sources = np.zeros((6, nx, ny))
    for fid in range(6):
        u = manufactured_population_field(x, y, t, fid)

        dt = 1e-6
        ut = (manufactured_population_field(x, y, t + dt, fid) - u) / dt


        d2u_dx2 = np.zeros_like(u)
        d2u_dy2 = np.zeros_like(u)
        d2u_dx2[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx ** 2
        d2u_dy2[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dy ** 2


        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)

        sources[fid] = ut - D * (d2u_dx2 + d2u_dy2) + vx * du_dx + vy * du_dy

    return sources


def burgers_exact_solution_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    nu: float = 0.01,
    amplitude: float = 1.0
) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing='ij')

    phi_x = np.exp(-(X - 4 * t) ** 2 / (4 * nu * (t + 1))) + \
            np.exp(-(X - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1)))
    phi_y = np.exp(-(Y - 4 * t) ** 2 / (4 * nu * (t + 1))) + \
            np.exp(-(Y - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1)))

    dphi_x = -(X - 4 * t) / (2 * nu * (t + 1)) * np.exp(-(X - 4 * t) ** 2 / (4 * nu * (t + 1))) \
             - (X - 4 * t - 2 * np.pi) / (2 * nu * (t + 1)) * np.exp(-(X - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1)))
    dphi_y = -(Y - 4 * t) / (2 * nu * (t + 1)) * np.exp(-(Y - 4 * t) ** 2 / (4 * nu * (t + 1))) \
             - (Y - 4 * t - 2 * np.pi) / (2 * nu * (t + 1)) * np.exp(-(Y - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1)))

    ux = 4.0 - 2.0 * nu * dphi_x / phi_x
    uy = 4.0 - 2.0 * nu * dphi_y / phi_y
    return amplitude * (ux + uy) * 0.5


def gauss_hermite_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_hermite(n)
    return x, w


def integrate_gauss_hermite_2d(
    func,
    n: int = 8,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0
) -> float:
    x, wx = gauss_hermite_quadrature(n)
    y, wy = gauss_hermite_quadrature(n)
    total = 0.0
    for i in range(n):
        for j in range(n):
            xi = np.sqrt(2.0) * sigma_x * x[i]
            yi = np.sqrt(2.0) * sigma_y * y[j]
            total += wx[i] * wy[j] * func(xi, yi)
    return float(total)
