
import numpy as np
from typing import Tuple, Optional


class FEMGravityError(Exception):
    pass


def wathen_element_matrix() -> np.ndarray:
    em = np.array([
        [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
        [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
        [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
        [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
        [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
        [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
        [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
        [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
    ], dtype=float)
    return em


def assemble_fem_system_2d(
    nx: int,
    ny: int,
    density_grid: np.ndarray,
    g_const: float = 6.67430e-11,
    dx: float = 1.0,
    dy: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    if density_grid.shape != (ny, nx):
        raise FEMGravityError(f"密度网格形状 {density_grid.shape} 不匹配 (ny={ny}, nx={nx})")

    n = nx * ny

    ml = nx + 1


    K = np.zeros((n, n))
    F = np.zeros(n)


    factor = 1.0 / 6.0 * (dy / dx + dx / dy)
    ke = factor * np.array([
        [4.0, -1.0, -2.0, -1.0],
        [-1.0, 4.0, -1.0, -2.0],
        [-2.0, -1.0, 4.0, -1.0],
        [-1.0, -2.0, -1.0, 4.0]
    ])

    for j in range(ny - 1):
        for i in range(nx - 1):




            nodes = [
                j * nx + i,
                j * nx + i + 1,
                (j + 1) * nx + i,
                (j + 1) * nx + i + 1
            ]

            rho_avg = 0.25 * (
                density_grid[j, i] + density_grid[j, i + 1] +
                density_grid[j + 1, i] + density_grid[j + 1, i + 1]
            )

            rhs_per_node = -4.0 * np.pi * g_const * rho_avg * dx * dy / 4.0
            for idx, node in enumerate(nodes):
                F[node] += rhs_per_node
                for jdx, node_j in enumerate(nodes):
                    K[node, node_j] += ke[idx, jdx]


    for i in range(nx):
        K[i, :] = 0.0
        K[:, i] = 0.0
        K[i, i] = 1.0
        F[i] = 0.0
        bottom = (ny - 1) * nx + i
        K[bottom, :] = 0.0
        K[:, bottom] = 0.0
        K[bottom, bottom] = 1.0
        F[bottom] = 0.0
    for j in range(ny):
        left = j * nx
        right = j * nx + nx - 1
        K[left, :] = 0.0
        K[:, left] = 0.0
        K[left, left] = 1.0
        F[left] = 0.0
        K[right, :] = 0.0
        K[:, right] = 0.0
        K[right, right] = 1.0
        F[right] = 0.0



    A_blt = dense_to_r8blt(K, ml)

    return A_blt, F, n, ml


def dense_to_r8blt(K: np.ndarray, ml: int) -> np.ndarray:
    n = K.shape[0]
    A = np.zeros((ml + 1, n))
    for j in range(n):
        A[0, j] = K[j, j]
        ihi = min(j + ml, n - 1)
        for i in range(j + 1, ihi + 1):
            A[i - j, j] = K[i, j]
    return A


def r8blt_sl(n: int, ml: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = b.copy()
    for j in range(n):
        if abs(a[0, j]) < 1e-16:
            raise FEMGravityError(f"零对角元在索引 {j}，矩阵奇异")
        x[j] = x[j] / a[0, j]
        ihi = min(j + ml, n - 1)
        for i in range(j + 1, ihi + 1):
            x[i] = x[i] - a[i - j, j] * x[j]
    return x


def solve_internal_potential_2d(
    nx: int = 32,
    ny: int = 32,
    density_profile: Optional[np.ndarray] = None,
    r_max: float = 1e3,
    g_const: float = 6.67430e-11
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = r_max / (nx - 1)
    dy = r_max / (ny - 1)
    x_coords = np.linspace(-r_max / 2, r_max / 2, nx)
    y_coords = np.linspace(-r_max / 2, r_max / 2, ny)

    if density_profile is None:

        X, Y = np.meshgrid(x_coords, y_coords)
        r = np.sqrt(X ** 2 + Y ** 2)
        density_profile = 3000.0 * np.exp(-r / (0.3 * r_max))
        density_profile = np.clip(density_profile, 100.0, 5000.0)

    A_blt, F, n, ml = assemble_fem_system_2d(nx, ny, density_profile, g_const, dx, dy)


    phi_flat = r8blt_sl(n, ml, A_blt, F)
    phi = phi_flat.reshape((ny, nx))

    return phi, x_coords, y_coords


def internal_gravity_from_potential(
    phi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = phi.shape
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)

    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0


    if nx > 2:
        gx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
        gx[:, 0] = -(phi[:, 1] - phi[:, 0]) / dx
        gx[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
    if ny > 2:
        gy[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
        gy[0, :] = -(phi[1, :] - phi[0, :]) / dy
        gy[-1, :] = -(phi[-1, :] - phi[-2, :]) / dy

    return gx, gy
