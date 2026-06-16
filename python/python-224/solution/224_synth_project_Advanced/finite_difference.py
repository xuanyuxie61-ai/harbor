"""
finite_difference.py
====================
High-order finite-difference (FD) operators for derivatives of the
profile likelihood with respect to signal strengths.

The Fornberg algorithm produces centered FD weights of arbitrary order
on uniform or non-uniform grids. For a uniform grid with spacing h,
the 2m-th order centered weights for the 2nd derivative satisfy:
    Sum_k c_k f(x + k h) / h^2 = f''(x) + O(h^{2m})

Specific formulas used:
  2nd-order (m=1):  c = [1, -2, 1]
  4th-order (m=2):  c = [-1/12, 4/3, -5/2, 4/3, -1/12]
  6th-order (m=3):  c = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
  8th-order (m=4):  c = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]

Von Neumann stability analysis for the diffusion equation u_t = alpha u_xx:
  amplification factor g(xi) = 1 + alpha dt/h^2 * Sum_k c_k exp(i k xi h)
  stability requires |g| <= 1 for all xi.

The maximum stable time step is:
  dt_max = h^2 / (2 alpha * Sum_k |c_k|)

For Higgs likelihood, we treat the signal-strength profile as a
"temperature" field and solve a pseudo-time diffusion equation to
smooth out numerical noise. This is the "likelihood diffusion" step
(inspired by seed 1258_VLOGroup_PoGMDM).

Grid tiling configurations for parameter scans (seed 906_pram_view)
are provided by tile_grid().
"""
from __future__ import annotations
import numpy as np


class HighOrderFD:
    """
    High-order centered finite-difference operators via Fornberg's algorithm.
    """

    def __init__(self, max_order: int = 8) -> None:
        self.max_order = max_order
        # Pre-compute coefficients for 2nd derivative on uniform grid
        self._coeffs_d2 = {
            2: np.array([1.0, -2.0, 1.0]),
            4: np.array([-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]),
            6: np.array(
                [1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0,
                 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]
            ),
            8: np.array(
                [-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
                 -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0]
            ),
        }
        # Coefficients for 1st derivative
        self._coeffs_d1 = {
            2: np.array([-1.0, 0.0, 1.0]) / 2.0,
            4: np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,
            6: np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0,
            8: np.array(
                [1.0, -32.0 / 3.0, 56.0, -224.0, 0.0, 224.0, -56.0, 32.0 / 3.0, -1.0]
            ) / 280.0,
        }

    # ------------------------------------------------------------------ #
    #                       Fornberg algorithm                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def fornberg_weights(x: np.ndarray, x0: float, deriv: int) -> np.ndarray:
        """
        Compute FD weights for derivative of order `deriv` at x0
        from function values at grid points x (arbitrary spacing).

        Reference: B. Fornberg, Math. Comp. 51 (1988) 699-706.
        """
        n = len(x)
        if deriv >= n:
            return np.zeros(n)
        c = np.zeros((n, deriv + 1))
        c[0, 0] = 1.0
        c1 = 1.0
        for i in range(1, n):
            mn = min(i, deriv)
            c2 = 1.0
            for j in range(i):
                c3 = x[i] - x[j]
                c2 *= c3
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - (x[i] - x0) * c[i - 1, k]) / c2
                c[i, 0] = -c1 * (x[i] - x0) * c[i - 1, 0] / c2
                for k in range(mn, 0, -1):
                    c[j, k] = ((x[i] - x0) * c[j, k] - k * c[j, k - 1]) / c3
                c[j, 0] = (x[i] - x0) * c[j, 0] / c3
            c1 = c2
        return c[:, deriv]

    # ------------------------------------------------------------------ #
    #                       Derivative evaluation                        #
    # ------------------------------------------------------------------ #
    def d1(self, f, x0: float, h: float = 1e-3, order: int = 4) -> float:
        """First derivative f'(x0) using centered FD of given order."""
        if order not in self._coeffs_d1:
            raise ValueError(f"Order {order} not available for d1")
        c = self._coeffs_d1[order]
        m = (len(c) - 1) // 2
        total = 0.0
        for k, ck in enumerate(c):
            xk = x0 + (k - m) * h
            total += ck * float(f(xk))
        return total / h

    def d2(self, f, x0: float, h: float = 1e-2, order: int = 4) -> float:
        """Second derivative f''(x0) using centered FD of given order."""
        if order not in self._coeffs_d2:
            raise ValueError(f"Order {order} not available for d2")
        c = self._coeffs_d2[order]
        m = (len(c) - 1) // 2
        total = 0.0
        for k, ck in enumerate(c):
            xk = x0 + (k - m) * h
            total += ck * float(f(xk))
        return total / (h * h)

    def gradient(self, f, x0: np.ndarray, h: float = 1e-3, order: int = 4) -> np.ndarray:
        """Gradient of scalar function f at vector x0."""
        x0 = np.asarray(x0, dtype=float)
        grad = np.zeros_like(x0)
        for i in range(len(x0)):
            def g(t):
                xx = x0.copy()
                xx[i] = t
                return float(f(xx))
            grad[i] = self.d1(g, x0[i], h=h, order=order)
        return grad

    def hessian(self, f, x0: np.ndarray, h: float = 5e-3, order: int = 4) -> np.ndarray:
        """
        Hessian matrix H_{ij} = d^2 f / dx_i dx_j at vector x0.
        Off-diagonals via mixed second-order FD:
            d^2 f / dx_i dx_j ~ [f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4 h^2)
        """
        x0 = np.asarray(x0, dtype=float)
        n = len(x0)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    def g(t):
                        xx = x0.copy()
                        xx[i] = t
                        return float(f(xx))
                    H[i, i] = self.d2(g, x0[i], h=h, order=order)
                else:
                    def f_pp():
                        xx = x0.copy()
                        xx[i] += h
                        xx[j] += h
                        return float(f(xx))

                    def f_pm():
                        xx = x0.copy()
                        xx[i] += h
                        xx[j] -= h
                        return float(f(xx))

                    def f_mp():
                        xx = x0.copy()
                        xx[i] -= h
                        xx[j] += h
                        return float(f(xx))

                    def f_mm():
                        xx = x0.copy()
                        xx[i] -= h
                        xx[j] -= h
                        return float(f(xx))

                    H[i, j] = (f_pp() - f_pm() - f_mp() + f_mm()) / (4.0 * h * h)
                    H[j, i] = H[i, j]
        return H

    # ------------------------------------------------------------------ #
    #                    Von Neumann stability limit                     #
    # ------------------------------------------------------------------ #
    def von_neumann_dtmax(self, alpha: float, h: float, order: int = 4) -> float:
        """
        Maximum stable time step for u_t = alpha u_xx using FD order `order`.
        Condition: dt <= h^2 / (2 alpha * Sum_k |c_k|)
        """
        c = self._coeffs_d2[order]
        return h ** 2 / (2.0 * alpha * np.sum(np.abs(c)))

    def dispersion_error(self, xi_h: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Dispersion error for 2nd-derivative FD operator:
            error(xi) = (Sum_k c_k exp(i k xi h) - (xi h)^2) / (xi h)^2
        """
        c = self._coeffs_d2[order]
        m = (len(c) - 1) // 2
        k_vals = np.arange(-m, m + 1)
        amplification = np.zeros_like(xi_h, dtype=complex)
        for k, ck in zip(k_vals, c):
            amplification += ck * np.exp(1j * k * xi_h)
        return (amplification.real + xi_h ** 2) / np.maximum(xi_h ** 2, 1e-12)

    # ------------------------------------------------------------------ #
    #                    Grid tiling (seed 906_pram_view)                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def tile_grid(bounds: list, tiles_per_dim: int) -> np.ndarray:
        """
        Generate a tiled parameter grid over `bounds = [(lo_i, hi_i), ...]`.
        Returns array of shape (N_tiles, N_dims) of tile centers.
        (seed 906_pram_view: parallel RAM view of tiled parameter grids)
        """
        axes = [np.linspace(lo, hi, tiles_per_dim) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)
