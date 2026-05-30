
import numpy as np
from typing import Tuple


def gradient_2d_centered(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)

    if nx < 2 or ny < 2:
        return dfdx, dfdy


    if nx >= 3:
        dfdx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)
    dfdx[:, 0] = (field[:, 1] - field[:, 0]) / dx
    dfdx[:, -1] = (field[:, -1] - field[:, -2]) / dx


    if ny >= 3:
        dfdy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)
    dfdy[0, :] = (field[1, :] - field[0, :]) / dy
    dfdy[-1, :] = (field[-1, :] - field[-2, :]) / dy

    return dfdx, dfdy


def gradient_3d_centered(field: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if field.ndim != 3:
        raise ValueError("Field must be 3D")
    nz, ny, nx = field.shape
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)
    dfdz = np.zeros_like(field)

    if nx >= 3:
        dfdx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx)
    dfdx[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    dfdx[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx

    if ny >= 3:
        dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dy)
    dfdy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dy
    dfdy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dy

    if nz >= 3:
        dfdz[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2.0 * dz)
    dfdz[0, :, :] = (field[1, :, :] - field[0, :, :]) / dz
    dfdz[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dz

    return dfdx, dfdy, dfdz


def divergence_2d(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dudx, _ = gradient_2d_centered(u, dx, dy)
    _, dvdy = gradient_2d_centered(v, dx, dy)
    return dudx + dvdy


def moisture_flux_convergence(qv: np.ndarray, u: np.ndarray, v: np.ndarray,
                              w: np.ndarray, rho: np.ndarray,
                              dx: float, dy: float, dz: float) -> np.ndarray:
    if not (qv.shape == u.shape == v.shape == w.shape == rho.shape):
        raise ValueError("All input fields must have the same shape")
    if qv.ndim != 3:
        raise ValueError("Inputs must be 3D")

    flux_x = rho * qv * u
    flux_y = rho * qv * v
    flux_z = rho * qv * w

    dfx_dx, _, _ = gradient_3d_centered(flux_x, dx, dy, dz)
    _, dfy_dy, _ = gradient_3d_centered(flux_y, dx, dy, dz)
    _, _, dfz_dz = gradient_3d_centered(flux_z, dx, dy, dz)

    mfc = -(dfx_dx + dfy_dy + dfz_dz)

    mfc = np.where(np.isfinite(mfc), mfc, 0.0)
    return mfc


def moisture_flux_convergence_2d(qv: np.ndarray, u: np.ndarray, v: np.ndarray,
                                 rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if not (qv.shape == u.shape == v.shape == rho.shape):
        raise ValueError("All input fields must have the same shape")
    flux_x = rho * qv * u
    flux_y = rho * qv * v
    dfx_dx, _ = gradient_2d_centered(flux_x, dx, dy)
    _, dfy_dy = gradient_2d_centered(flux_y, dx, dy)
    mfc = -(dfx_dx + dfy_dy)
    return np.where(np.isfinite(mfc), mfc, 0.0)


def laplacian_9point_torus(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return np.zeros_like(field)

    lap = np.zeros_like(field)
    coeff = 1.0 / (6.0 * dx * dy)

    for j in range(ny):
        for i in range(nx):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            ip = (i + 1) % nx
            im = (i - 1) % nx

            lap[j, i] = coeff * (
                1.0 * field[jm, im] + 4.0 * field[jm, i] + 1.0 * field[jm, ip]
                + 4.0 * field[j, im] - 20.0 * field[j, i] + 4.0 * field[j, ip]
                + 1.0 * field[jp, im] + 4.0 * field[jp, i] + 1.0 * field[jp, ip]
            )
    return lap


def laplacian_5point(field: np.ndarray, dx: float, dy: float,
                     periodic_x: bool = False, periodic_y: bool = False) -> np.ndarray:
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return np.zeros_like(field)

    lap = np.zeros_like(field)
    dx2 = dx * dx
    dy2 = dy * dy


    lap[1:-1, 1:-1] = (
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dx2
        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dy2
    )


    if periodic_x:
        lap[1:-1, 0] = (field[1:-1, 1] - 2.0 * field[1:-1, 0] + field[1:-1, -1]) / dx2 + (field[2:, 0] - 2.0 * field[1:-1, 0] + field[:-2, 0]) / dy2
        lap[1:-1, -1] = (field[1:-1, 0] - 2.0 * field[1:-1, -1] + field[1:-1, -2]) / dx2 + (field[2:, -1] - 2.0 * field[1:-1, -1] + field[:-2, -1]) / dy2
    if periodic_y:
        lap[0, 1:-1] = (field[0, 2:] - 2.0 * field[0, 1:-1] + field[0, :-2]) / dx2 + (field[1, 1:-1] - 2.0 * field[0, 1:-1] + field[-1, 1:-1]) / dy2
        lap[-1, 1:-1] = (field[-1, 2:] - 2.0 * field[-1, 1:-1] + field[-1, :-2]) / dx2 + (field[0, 1:-1] - 2.0 * field[-1, 1:-1] + field[-2, 1:-1]) / dy2

    return lap
