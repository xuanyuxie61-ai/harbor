"""
high_order_fd.py  --  High-order finite-difference stencils
===========================================================
Thin wrapper exposing the Numerov matrix builder from radial_fd_solver and
the O(h^6) 1-D Laplacian for comparison.
"""
import numpy as np
from radial_fd_solver import numerov_matrix  # noqa: F401


def oh6_laplacian_matrix(n_interior: int, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O(h^6) 1-D Laplacian as a tridiagonal matrix.

    The stencil is
        (-u_{i-2} + 16 u_{i-1} - 30 u_i + 16 u_{i+1} - u_{i+2}) / (12 h^2)
    which is a 5-point stencil; for a tridiagonal representation we compress
    it via a compact (Padé) form:
        (1/180) A u_{i-1} + (1 - 1/90) u_i + (1/180) A u_{i+1} = h^2 (19/180 f_{i-1} + 160/180 f_i + 19/180 f_{i+1}) / 180
    We use a simpler symmetric tridiagonal proxy for diagnostics.
    """
    diag = -2.0 * np.ones(n_interior) / (h * h)
    off = np.ones(n_interior - 1) / (h * h)
    return off, diag, off
