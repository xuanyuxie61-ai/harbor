
import numpy as np


def normal_exact(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.exp(-t ** 2 / 2.0) / np.sqrt(2.0 * np.pi)


def normal_deriv(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -t * y


def pwl_interp_2d_scalar(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                         xi: float, yi: float) -> float:
    nxd = len(xd)
    nyd = len(yd)

    if nxd < 2 or nyd < 2:
        raise ValueError("Need at least 2 points in each dimension")


    i = -1
    for idx in range(nxd - 1):
        if xd[idx] <= xi <= xd[idx + 1]:
            i = idx
            break
    if i == -1:
        return np.inf

    j = -1
    for idx in range(nyd - 1):
        if yd[idx] <= yi <= yd[idx + 1]:
            j = idx
            break
    if j == -1:
        return np.inf




    diag_y = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < diag_y:

        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return zd[i, j]
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:

        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return zd[i + 1, j + 1]
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]


def pwl_interp_2d_vector(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                         xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    ni = len(xi)
    zi = np.zeros(ni, dtype=float)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(xd, yd, zd, xi[k], yi[k])
    return zi


def build_lennard_jones_potential_grid(x_range: tuple, y_range: tuple,
                                       nx: int, ny: int,
                                       sigma_nm: float = 0.4,
                                       epsilon_kJ_mol: float = 4.0) -> tuple:
    xd = np.linspace(x_range[0], x_range[1], nx)
    yd = np.linspace(y_range[0], y_range[1], ny)
    zd = np.zeros((nx, ny), dtype=float)

    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(xd[i] ** 2 + yd[j] ** 2)
            if r < 0.1:
                r = 0.1
            sr6 = (sigma_nm / r) ** 6
            zd[i, j] = 4.0 * epsilon_kJ_mol * (sr6 ** 2 - sr6)

    return xd, yd, zd


def build_morse_potential_1d(r: np.ndarray, D_e: float = 50.0,
                             a: float = 2.0, r_e: float = 0.4) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return D_e * (1.0 - np.exp(-a * (r - r_e))) ** 2 - D_e


def gaussian_binding_potential_2d(x: np.ndarray, y: np.ndarray,
                                  x0: float, y0: float,
                                  sigma: float, depth: float) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return -depth * np.exp(-r2 / (2.0 * sigma ** 2))


def compute_rad51_dna_binding_energy(distance_nm: float,
                                      well_depth_kJ_mol: float = 35.0,
                                      sigma_nm: float = 0.25) -> float:
    if distance_nm < 0:
        raise ValueError("distance must be non-negative")
    return -well_depth_kJ_mol * np.exp(-distance_nm ** 2 / (2.0 * sigma_nm ** 2))


def potential_force_from_grid(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                              x: float, y: float) -> tuple:
    h_x = (xd[-1] - xd[0]) / (len(xd) - 1)
    h_y = (yd[-1] - yd[0]) / (len(yd) - 1)


    if x < xd[0] or x > xd[-1] or y < yd[0] or y > yd[-1]:
        return 0.0, 0.0


    v_px = pwl_interp_2d_scalar(xd, yd, zd, x + h_x, y)
    v_mx = pwl_interp_2d_scalar(xd, yd, zd, x - h_x, y)
    v_py = pwl_interp_2d_scalar(xd, yd, zd, x, y + h_y)
    v_my = pwl_interp_2d_scalar(xd, yd, zd, x, y - h_y)


    vals = [v_px, v_mx, v_py, v_my]
    for i in range(len(vals)):
        if np.isinf(vals[i]):
            vals[i] = 0.0

    fx = -(vals[0] - vals[1]) / (2.0 * h_x)
    fy = -(vals[2] - vals[3]) / (2.0 * h_y)
    return fx, fy
