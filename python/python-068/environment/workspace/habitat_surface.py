
import numpy as np


def bernstein_basis(u: float) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return np.array([
        (1.0 - u) ** 3,
        3.0 * u * (1.0 - u) ** 2,
        3.0 * u ** 2 * (1.0 - u),
        u ** 3
    ])


def bezier_patch_evaluate(control_points: np.ndarray, u: float, v: float) -> float:
    uvec = bernstein_basis(u)
    vvec = bernstein_basis(v)
    return float(uvec @ control_points @ vvec)


def bezier_surface_grid(control_points: np.ndarray, nu: int = 64, nv: int = 64) -> np.ndarray:
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    Z = np.zeros((nu, nv))
    for i in range(nu):
        uvec = bernstein_basis(u[i])
        for j in range(nv):
            vvec = bernstein_basis(v[j])
            Z[i, j] = uvec @ control_points @ vvec
    return Z


def create_habitat_carrying_capacity(
    nx: int = 64,
    ny: int = 64,
    K_base: float = 100.0,
    K_peak: float = 200.0
) -> np.ndarray:

    cp = np.array([
        [0.0, 0.1, 0.1, 0.0],
        [0.1, 0.5, 0.5, 0.1],
        [0.1, 0.5, 0.5, 0.1],
        [0.0, 0.1, 0.1, 0.0]
    ], dtype=float)
    B = bezier_surface_grid(cp, nx, ny)
    K = K_base + (K_peak - K_base) * B
    return K


def create_growth_rate_map(
    nx: int = 64,
    ny: int = 64,
    r_base: float = 0.5,
    r_peak: float = 1.5
) -> np.ndarray:
    cp = np.array([
        [0.0, 0.05, 0.05, 0.0],
        [0.05, 0.4, 0.4, 0.05],
        [0.05, 0.4, 0.4, 0.05],
        [0.0, 0.05, 0.05, 0.0]
    ], dtype=float)
    B = bezier_surface_grid(cp, nx, ny)
    r = r_base + (r_peak - r_base) * B
    return r


def bezier_surface_gradient(control_points: np.ndarray, u: float, v: float) -> tuple[float, float]:
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)


    dBu = np.array([
        -3.0 * (1.0 - u) ** 2,
        3.0 * (1.0 - u) ** 2 - 6.0 * u * (1.0 - u),
        6.0 * u * (1.0 - u) - 3.0 * u ** 2,
        3.0 * u ** 2
    ])
    dBv = np.array([
        -3.0 * (1.0 - v) ** 2,
        3.0 * (1.0 - v) ** 2 - 6.0 * v * (1.0 - v),
        6.0 * v * (1.0 - v) - 3.0 * v ** 2,
        3.0 * v ** 2
    ])

    du = float(dBu @ control_points @ bernstein_basis(v))
    dv = float(bernstein_basis(u) @ control_points @ dBv)
    return du, dv
