"""
high_order_fd.py
================
High-order finite-difference operators for the compressible MHD system
in the shearing-box approximation.

Three families of schemes are provided:

    1. WENO5  -- fifth-order weighted essentially non-oscillatory
                 reconstruction of cell-interface values (Jiang & Shu
                 1996).  Used for the advective fluxes.
    2. Compact 4th-order centred scheme (Lele 1992) for the viscous /
       resistive / source-term derivatives that sit on the cell centres.
    3. A piecewise-linear 2-D interpolant inspired by pwl_interp_2d
       (927) used to reconstruct face values from cell-centred data
       when a cheap but monotone fallback is acceptable (e.g. near
       shocks detected by the Jiang--Shu smoothness indicator).

All operators work on 1-D pencils extracted from the 3-D conserved
array; the caller is responsible for looping over transverse indices.
This keeps the code simple, cache-friendly, and trivially portable to
GPU pencil-decomposition frameworks.

Smoothness indicators
---------------------
For WENO5 we use the classical Jiang-Shu polynomials

    beta_0 = (13/12) (f_{i-2} - 2 f_{i-1} + f_i  )^2
           + (1/4)   (f_{i-2} - 4 f_{i-1} + 3 f_i)^2
    beta_1 = (13/12) (f_{i-1} - 2 f_i + f_{i+1})^2
           + (1/4)   (f_{i-1} - f_{i+1})^2
    beta_2 = (13/12) (f_i - 2 f_{i+1} + f_{i+2})^2
           + (1/4)   (3 f_i - 4 f_{i+1} + f_{i+2})^2

and the ideal linear weights

    d_0 = 1/10,    d_1 = 6/10,    d_2 = 3/10

for the +1/2 face; the -1/2 face uses the mirror set (3/10, 6/10, 1/10).
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
#                            WENO5 reconstruction
# ---------------------------------------------------------------------------
_EPS_WENO = 1.0e-36   # floor to avoid division by zero in smoothness weights


def weno5_reconstruct(f: np.ndarray, axis: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct left and right interface values using WENO5.

    Parameters
    ----------
    f : ndarray
        1-D array of cell-averaged values (length >= 5).
    axis : int
        Axis along which to operate (must be 0 for this implementation).

    Returns
    -------
    f_plus, f_minus : ndarray
        Interface values at i+1/2 from the left and from the right,
        each of length (N-4) corresponding to interior faces only.
    """
    if f.shape[axis] < 5:
        raise ValueError("weno5_reconstruct: need at least 5 points")

    fm2 = f[:-4]
    fm1 = f[1:-3]
    f0  = f[2:-2]
    fp1 = f[3:-1]
    fp2 = f[4:]

    # ----- candidate stencil polynomials at i+1/2 -----
    q0 = ( 2.0 * fm2 - 7.0 * fm1 + 11.0 * f0) / 6.0
    q1 = (-1.0 * fm1 + 5.0 * f0  +  2.0 * fp1) / 6.0
    q2 = ( 2.0 * f0  + 5.0 * fp1 -       fp2) / 6.0

    # ----- Jiang-Shu smoothness indicators -----
    b0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0)**2 \
         + 0.25 * (fm2 - 4.0 * fm1 + 3.0 * f0)**2
    b1 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1)**2 \
         + 0.25 * (fm1 - fp1)**2
    b2 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2)**2 \
         + 0.25 * (3.0 * f0 - 4.0 * fp1 + fp2)**2

    # ----- ideal linear weights (d_0, d_1, d_2) = (1/10, 6/10, 3/10) -----
    d0, d1, d2 = 0.1, 0.6, 0.3

    a0 = d0 / ((d0 + _EPS_WENO) + (b0 + _EPS_WENO)**2)
    a1 = d1 / ((d1 + _EPS_WENO) + (b1 + _EPS_WENO)**2)
    a2 = d2 / ((d2 + _EPS_WENO) + (b2 + _EPS_WENO)**2)
    asum = a0 + a1 + a2
    w0 = a0 / asum
    w1 = a1 / asum
    w2 = a2 / asum

    f_plus = w0 * q0 + w1 * q1 + w2 * q2

    # ----- mirror stencil for the right-biased reconstruction -----
    q0r = ( 2.0 * fp2 - 7.0 * fp1 + 11.0 * f0) / 6.0
    q1r = (-1.0 * fp1 + 5.0 * f0  +  2.0 * fm1) / 6.0
    q2r = ( 2.0 * f0  + 5.0 * fm1 -       fm2) / 6.0

    b0r = (13.0 / 12.0) * (fp2 - 2.0 * fp1 + f0)**2 \
          + 0.25 * (fp2 - 4.0 * fp1 + 3.0 * f0)**2
    b1r = (13.0 / 12.0) * (fp1 - 2.0 * f0 + fm1)**2 \
          + 0.25 * (fp1 - fm1)**2
    b2r = (13.0 / 12.0) * (f0 - 2.0 * fm1 + fm2)**2 \
          + 0.25 * (3.0 * f0 - 4.0 * fm1 + fm2)**2

    dr0, dr1, dr2 = 3.0 / 10.0, 6.0 / 10.0, 1.0 / 10.0
    ar0 = dr0 / ((dr0 + _EPS_WENO) + (b0r + _EPS_WENO)**2)
    ar1 = dr1 / ((dr1 + _EPS_WENO) + (b1r + _EPS_WENO)**2)
    ar2 = dr2 / ((dr2 + _EPS_WENO) + (b2r + _EPS_WENO)**2)
    asr = ar0 + ar1 + ar2
    wr0 = ar0 / asr
    wr1 = ar1 / asr
    wr2 = ar2 / asr

    f_minus = wr0 * q0r + wr1 * q1r + wr2 * q2r

    return f_plus, f_minus


# ---------------------------------------------------------------------------
#                 Compact 4th-order centred derivative
# ---------------------------------------------------------------------------
def compact4_1st_deriv(f: np.ndarray, dx: float) -> np.ndarray:
    """Fourth-order compact (Pade) first derivative on a uniform grid.

    Solves the tridiagonal system

        (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1}
              = (3/2) (f_{i+1} - f_{i-1}) / (2 dx)

    which has truncation error O(dx^4).  Boundaries fall back to a
    one-sided fourth-order finite difference.
    """
    N = f.size
    if N < 5:
        raise ValueError("compact4_1st_deriv: need at least 5 points")
    rhs = np.zeros(N)
    rhs[1:-1] = 1.5 * (f[2:] - f[:-2]) / (2.0 * dx)
    # 4th-order one-sided at the boundaries
    rhs[0]  = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2]
               + 16.0 * f[3] - 3.0 * f[4]) / (12.0 * dx)
    rhs[-1] = (25.0 * f[-1] - 48.0 * f[-2] + 36.0 * f[-3]
               - 16.0 * f[-4] + 3.0 * f[-5]) / (12.0 * dx)

    # Thomas algorithm for (1/4, 1, 1/4) tridiagonal
    a = np.full(N, 0.25)
    b = np.ones(N)
    c = np.full(N, 0.25)
    a[0] = 0.0
    c[-1] = 0.0

    cp = np.zeros(N)
    dp = np.zeros(N)
    cp[0] = c[0] / b[0]
    dp[0] = rhs[0] / b[0]
    for i in range(1, N):
        m = b[i] - a[i] * cp[i - 1]
        if abs(m) < 1.0e-30:
            m = math.copysign(1.0e-30, m) if m != 0 else 1.0e-30
        cp[i] = c[i] / m if i < N - 1 else 0.0
        dp[i] = (rhs[i] - a[i] * dp[i - 1]) / m
    out = np.zeros(N)
    out[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


# ---------------------------------------------------------------------------
#            Piecewise-linear 2-D fallback (from 927_pwl_interp)
# ---------------------------------------------------------------------------
def pwl_reconstruct_2d(f: np.ndarray,
                       dx: float, dy: float) -> np.ndarray:
    """Piecewise-linear reconstruction of face values in 2-D.

    For every cell (i, j) we compute the slope-limited gradients

        sigma_x = minmod( (f_{i+1,j} - f_{i,j}) / dx,
                          (f_{i,j} - f_{i-1,j}) / dx )
        sigma_y = minmod( (f_{i,j+1} - f_{i,j}) / dy,
                          (f_{i,j} - f_{i,j-1}) / dy )

    and use them to obtain left/right and bottom/top face values.  The
    minmod limiter is the simplest TVD limiter and is equivalent to the
    triangle choice logic in pwl_interp_2d (927) when one triangle is
    aligned with a coordinate direction.
    """
    def minmod(a, b):
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    sx_pos = (np.roll(f, -1, axis=0) - f) / dx
    sx_neg = (f - np.roll(f, +1, axis=0)) / dx
    sx = minmod(sx_pos, sx_neg)

    sy_pos = (np.roll(f, -1, axis=1) - f) / dy
    sy_neg = (f - np.roll(f, +1, axis=1)) / dy
    sy = minmod(sy_pos, sy_neg)

    # Return slopes; caller can form face values as f +/- 0.5 * dx * sx
    return sx, sy


# ---------------------------------------------------------------------------
#                 Shock detector (Jiang-Shu smoothness)
# ---------------------------------------------------------------------------
def jiang_shu_smoothness(f: np.ndarray) -> np.ndarray:
    """Return the global smoothness indicator beta for each cell.

    Small values mark smooth regions (WENO weights collapse to the
    ideal linear weights); large values flag shocks or contact
    discontinuities where the scheme must fall back to the most
    dissipative sub-stencil.
    """
    fm1 = np.roll(f, +1, axis=0)
    fp1 = np.roll(f, -1, axis=0)
    beta = (13.0 / 12.0) * (fm1 - 2.0 * f + fp1)**2 \
           + 0.25 * (fp1 - fm1)**2
    return beta


# ---------------------------------------------------------------------------
#                3-D gradient driver
# ---------------------------------------------------------------------------
def gradient_3d(F: np.ndarray,
                dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the compact 4th derivative along each axis of a 3-D field.

    ``F`` is expected to have shape (Nx, Ny, Nz).  The returned tuple
    is (dF/dx, dF/dy, dF/dz).  For non-uniform x grids we absorb the
    local dx into the derivative by rescaling after the uniform call.
    """
    Nx, Ny, Nz = F.shape
    dFdx = np.zeros_like(F)
    dFdy = np.zeros_like(F)
    dFdz = np.zeros_like(F)
    # x-pencil passes
    for j in range(Ny):
        for k in range(Nz):
            pencil = F[:, j, k]
            # Map to uniform reference coordinate with mean(dx)
            dx_mean = float(np.mean(dx))
            dFdx[:, j, k] = compact4_1st_deriv(pencil, dx_mean) / (dx / dx_mean)
    # y-pencil passes (strictly uniform)
    dy_val = float(dy[0]) if dy.size else 1.0
    for i in range(Nx):
        for k in range(Nz):
            dFdy[i, :, k] = compact4_1st_deriv(F[i, :, k], dy_val)
    # z-pencil passes (strictly uniform)
    dz_val = float(dz[0]) if dz.size else 1.0
    for i in range(Nx):
        for j in range(Ny):
            dFdz[i, j, :] = compact4_1st_deriv(F[i, j, :], dz_val)
    return dFdx, dFdy, dFdz
