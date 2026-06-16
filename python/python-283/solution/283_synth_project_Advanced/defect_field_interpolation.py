"""
defect_field_interpolation.py
=============================
Interpolation utilities for defect-related scalar fields on the perovskite
device mesh. Combines:
    - 792_nearest_interp_1d : nearest-neighbor interpolation
    - 359_fd1d_display      : 1D finite-difference data representation

The nearest-neighbor interpolant is used to map defect densities from
a coarse mesh (where DFT calculations are performed) to a fine mesh (where
the device-scale Poisson/drift-diffusion simulation runs). This is the
simplest QoI transfer operator and serves as a baseline for higher-order
interpolants (linear, cubic).

The 1D FD representation renders a piecewise-linear function from a set of
node-value pairs (x_i, f_i). This is used to visualize the electrostatic
potential, carrier densities, and defect charge as a function of position
across the perovskite slab --- but since visualization is removed, we only
provide the numerical representation and convergence diagnostics.

Mathematical background
-----------------------
Nearest-neighbor interpolant:
    L(x) = y_k   where k = argmin_j |x - x_j|
This is a piecewise-constant function with discontinuities at the midpoints
of adjacent data points. The interpolation error is O(h) where h is the
maximum spacing between data points.

Higher-order extensions (not implemented but referenced):
- Linear: O(h^2) with continuous interpolant
- Cubic spline: O(h^4) with C^2 continuity
- PCHIP: shape-preserving, monotone
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Nearest-neighbor interpolation (from 792)
# ============================================================================
def nearest_interp_1d(xd: NDArray, yd: NDArray,
                      xi: NDArray) -> NDArray:
    """Evaluate the nearest-neighbor interpolant at points xi.
    xd, yd : data points and values (length ND)
    xi     : interpolation points (length NI)
    Returns yi of length NI where yi[j] = yd[k] with k = argmin |xi[j] - xd|.
    """
    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    xi = np.asarray(xi).ravel()
    # Use searchsorted for O(N log N) lookup
    idx = np.searchsorted(xd, xi, side='left')
    # Clip to valid range
    idx = np.clip(idx, 1, len(xd) - 1)
    # Choose the closer of idx-1 or idx
    left = idx - 1
    right = idx
    dist_left = np.abs(xi - xd[left])
    dist_right = np.abs(xi - xd[right])
    chosen = np.where(dist_left <= dist_right, left, right)
    return yd[chosen]


def piecewise_linear_interp(xd: NDArray, yd: NDArray,
                            xi: NDArray) -> NDArray:
    """Piecewise-linear interpolation (for comparison with nearest-neighbor)."""
    return np.interp(xi, xd, yd)


def interpolation_error(xd: NDArray, yd_true: NDArray,
                        interp_fn, n_test: int = 200) -> float:
    """Estimate the L2 interpolation error of interp_fn against a reference
    piecewise-linear interpolant on a fine grid."""
    xi = np.linspace(xd.min(), xd.max(), n_test)
    yd_fine = piecewise_linear_interp(xd, yd_true, xi)
    yd_interp = interp_fn(xi)
    return float(np.sqrt(np.mean((yd_interp - yd_fine) ** 2)))


# ============================================================================
# 1D FD data representation (from 359)
# ============================================================================
class FD1DField:
    """Represent a 1D finite-difference scalar field as a set of nodes and
    values. Provides:
        - piecewise-linear evaluation at arbitrary points
        - L2, Linf norms
        - convergence rate estimation from two grids
    """

    def __init__(self, x: NDArray, y: NDArray, label: str = "f"):
        self.x = np.asarray(x).ravel()
        self.y = np.asarray(y).ravel()
        self.label = label
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

    def __call__(self, xi: NDArray) -> NDArray:
        """Evaluate the piecewise-linear interpolant at xi."""
        return np.interp(xi, self.x, self.y)

    def nearest(self, xi: NDArray) -> NDArray:
        """Evaluate the nearest-neighbor interpolant at xi."""
        return nearest_interp_1d(self.x, self.y, xi)

    def L2_norm(self) -> float:
        """Compute the L2 norm ||f||_2 = sqrt(Integral f^2 dx)."""
        return float(np.sqrt(np.trapz(self.y ** 2, self.x)))

    def Linf_norm(self) -> float:
        """Compute the L-infinity norm ||f||_inf = max |f|."""
        return float(np.max(np.abs(self.y)))

    def derivative(self, p: int = 2) -> "FD1DField":
        """Compute the derivative using 2p-th order centered FD."""
        from high_order_fd import vandermonde_derivative_weights
        nx = len(self.x)
        dx = self.x[1] - self.x[0]
        pts = np.arange(-p, p + 1, dtype=float)
        w = vandermonde_derivative_weights(pts, 0.0, deriv_order=1)
        dy = np.zeros(nx)
        for i in range(p, nx - p):
            s = 0.0
            for k in range(-p, p + 1):
                s += w[k + p] * self.y[i + k]
            dy[i] = s / dx
        # Boundaries: one-sided
        dy[:p] = dy[p]
        dy[-p:] = dy[-p - 1]
        return FD1DField(self.x, dy, label=f"d{self.label}/dx")

    @staticmethod
    def convergence_rate(f_h: "FD1DField", f_h2: "FD1DField",
                         f_h4: "FD1DField", probe_x: float) -> float:
        """Estimate convergence order at probe_x from three grids with
        spacings h, h/2, h/4. Uses Richardson extrapolation formula."""
        v_h = float(np.interp(probe_x, f_h.x, f_h.y))
        v_h2 = float(np.interp(probe_x, f_h2.x, f_h2.y))
        v_h4 = float(np.interp(probe_x, f_h4.x, f_h4.y))
        d1 = v_h - v_h2
        d2 = v_h2 - v_h4
        if abs(d2) < 1e-30:
            return float('nan')
        return math.log2(abs(d1) / abs(d2))


# ============================================================================
# Defect-density upsampling: coarse (DFT) -> fine (device scale)
# ============================================================================
def upsample_defect_density(x_coarse: NDArray,
                            N_t_coarse: NDArray,
                            x_fine: NDArray,
                            method: str = "nearest") -> NDArray:
    """Upsample a coarse-grid defect density to a fine grid.
    method: 'nearest' | 'linear'
    Returns the upsampled N_t on x_fine.
    """
    if method == "nearest":
        return nearest_interp_1d(x_coarse, N_t_coarse, x_fine)
    elif method == "linear":
        return piecewise_linear_interp(x_coarse, N_t_coarse, x_fine)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Driver: test interpolation on a sample defect profile
# ============================================================================
def test_defect_interpolation() -> dict:
    """Test nearest vs linear interpolation on a Gaussian defect profile."""
    # Coarse grid (DFT resolution): 10 points
    x_coarse = np.linspace(0.0, 1.0, 10)
    N_t_coarse = 1e22 * np.exp(-((x_coarse - 0.5) ** 2) / (2.0 * 0.1 ** 2))
    # Fine grid (device resolution): 200 points
    x_fine = np.linspace(0.0, 1.0, 200)
    N_t_fine_true = 1e22 * np.exp(-((x_fine - 0.5) ** 2) / (2.0 * 0.1 ** 2))
    # Interpolation
    N_t_nearest = upsample_defect_density(x_coarse, N_t_coarse, x_fine,
                                          method="nearest")
    N_t_linear = upsample_defect_density(x_coarse, N_t_coarse, x_fine,
                                         method="linear")
    err_nearest = float(np.sqrt(np.mean((N_t_nearest - N_t_fine_true) ** 2)))
    err_linear = float(np.sqrt(np.mean((N_t_linear - N_t_fine_true) ** 2)))
    # Build FD1DField objects
    field_nearest = FD1DField(x_fine, N_t_nearest, label="N_t_nearest")
    field_linear = FD1DField(x_fine, N_t_linear, label="N_t_linear")
    return {
        "err_nearest": err_nearest,
        "err_linear": err_linear,
        "ratio": err_nearest / max(err_linear, 1e-30),
        "L2_nearest": field_nearest.L2_norm(),
        "L2_linear": field_linear.L2_norm(),
        "Linf_nearest": field_nearest.Linf_norm(),
        "Linf_linear": field_linear.Linf_norm(),
    }
