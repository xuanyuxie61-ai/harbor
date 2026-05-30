
import numpy as np
from typing import Tuple


def r8vec_bracket5(n: int, x: np.ndarray, xval: float) -> int:
    if n < 2 or xval < x[0] or xval > x[-1]:
        return -1

    lo, hi = 0, n - 1
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def pwl_interp_2d_scalar(
    nxd: int, nyd: int,
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    xi: float, yi: float
) -> float:
    i = r8vec_bracket5(nxd, xd, xi)
    if i == -1:
        return np.inf

    j = r8vec_bracket5(nyd, yd, yi)
    if j == -1:
        return np.inf





    y_diag = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < y_diag:

        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return np.inf
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
            return np.inf
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i, j + 1] + beta * zd[i + 1, j + 1] + gamma * zd[i + 1, j]


def pwl_interp_2d_vector(
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    xi: np.ndarray, yi: np.ndarray
) -> np.ndarray:
    nxd, nyd = len(xd), len(yd)
    ni = len(xi)
    zi = np.full(ni, np.inf)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi[k], yi[k])
    return zi


class PerovskiteMaterial:

    def __init__(self):

        self.T_grid = np.linspace(200.0, 400.0, 21)
        self.x_grid = np.linspace(0.0, 1.0, 11)


        self.Eg_grid = self._build_eg_table()

        self.mu_n_grid = self._build_mu_n_table()

        self.mu_p_grid = self._build_mu_p_table()

        self.alpha_grid = self._build_alpha_table()

    def _build_eg_table(self) -> np.ndarray:
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        Eg0 = 1.57 + 0.72 * Xg
        S = 8.0e-4
        theta_D = 150.0
        Eg = Eg0 - S * Tg ** 2 / (Tg + theta_D)
        return Eg

    def _build_mu_n_table(self) -> np.ndarray:
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        mu_n0 = 20.0
        mu_n = mu_n0 * (Tg / 300.0) ** (-1.5) * (1.0 + 0.2 * Xg) ** (-1)
        return np.clip(mu_n, 0.1, 500.0)

    def _build_mu_p_table(self) -> np.ndarray:
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        mu_p0 = 10.0
        mu_p = mu_p0 * (Tg / 300.0) ** (-1.5) * (1.0 + 0.3 * Xg) ** (-1)
        return np.clip(mu_p, 0.1, 500.0)

    def _build_alpha_table(self) -> np.ndarray:
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        Eg0 = 1.57 + 0.72 * Xg
        S = 8.0e-4
        theta_D = 150.0
        Eg = Eg0 - S * Tg ** 2 / (Tg + theta_D)
        E_photon = 2.07
        alpha0 = 5.0e4
        alpha = alpha0 * np.sqrt(np.maximum(E_photon - Eg, 0.0) + 0.01)
        return np.clip(alpha, 1e3, 1e6)

    def get_params(self, T: float, x: float) -> dict:

        T_clipped = np.clip(T, self.T_grid.min(), self.T_grid.max())
        x_clipped = np.clip(x, self.x_grid.min(), self.x_grid.max())

        Eg = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                   self.T_grid, self.x_grid, self.Eg_grid,
                                   T_clipped, x_clipped)
        mu_n = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                     self.T_grid, self.x_grid, self.mu_n_grid,
                                     T_clipped, x_clipped)
        mu_p = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                     self.T_grid, self.x_grid, self.mu_p_grid,
                                     T_clipped, x_clipped)
        alpha = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                      self.T_grid, self.x_grid, self.alpha_grid,
                                      T_clipped, x_clipped)


        if not np.isfinite(Eg):
            Eg = float(self.Eg_grid[len(self.T_grid)//2, len(self.x_grid)//2])
        if not np.isfinite(mu_n):
            mu_n = 10.0
        if not np.isfinite(mu_p):
            mu_p = 5.0
        if not np.isfinite(alpha):
            alpha = 5.0e4

        return {
            "bandgap_eV": float(Eg),
            "electron_mobility": float(mu_n),
            "hole_mobility": float(mu_p),
            "absorption_coeff_600nm": float(alpha),
        }


if __name__ == "__main__":
    mat = PerovskiteMaterial()
    params = mat.get_params(300.0, 0.3)
    print("Perovskite @ T=300K, x=0.3:", params)
