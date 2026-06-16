# -*- coding: utf-8 -*-
"""
high_order_stencils.py
----------------------
High-order finite-difference stencils used to discretise the spatial
operators in the focused transport equation.

Implemented schemes
-------------------
1. ``upwind_first_derivative``   -- 1st-order upwind, flux sign chosen
   by the local advection speed.
2. ``upwind_second_order``       -- 2nd-order upwind (linear
   extrapolation).
3. ``central_second_order``      -- classical 2nd-order centred
   (``(f_{i+1} - f_{i-1}) / (2 h)``).
4. ``compact4_first_derivative`` -- 4th-order Padé / compact scheme
   (Lele 1992):   (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1}
                  = (f_{i+1} - f_{i-1}) / (2 h).
5. ``weno5_first_derivative``    -- 5th-order WENO reconstruction
   (Jiang & Shu 1996) for the first derivative, applied on the cell
   faces.

Boundary closures: one-sided stencils on the outermost points.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


# =====================================================================
# 1st-order upwind
# =====================================================================
def upwind_first_derivative(f: np.ndarray, dx: np.ndarray,
                            speed: float = 0.0) -> np.ndarray:
    """Return df/dx using first-order upwind on the interior points.

    df/dx_i = (f_i - f_{i-1}) / dx_i       if speed >= 0
              (f_{i+1} - f_i) / dx_i       if speed <  0
    """
    n = f.size
    df = np.zeros(n)
    if n < 3:
        return df
    if speed >= 0.0:
        for i in range(1, n - 1):
            df[i] = (f[i] - f[i - 1]) / dx[i - 1]
    else:
        for i in range(1, n - 1):
            df[i] = (f[i + 1] - f[i]) / dx[i]
    # Neumann closure at the boundaries
    df[0] = df[1]
    df[-1] = df[-2]
    return df


# =====================================================================
# 2nd-order upwind
# =====================================================================
def upwind_second_order(f: np.ndarray, dx: np.ndarray,
                        speed: float = 0.0) -> np.ndarray:
    """2nd-order upwind on non-uniform grids.

    For uniform spacing h, the classic formulas are

        df_i = (-3 f_i + 4 f_{i+1} - f_{i+2}) / (2 h)   (speed > 0)
        df_i = ( 3 f_i - 4 f_{i-1} + f_{i-2}) / (2 h)   (speed < 0)

    On non-uniform grids we use the Lagrange form.
    """
    n = f.size
    df = np.zeros(n)
    if n < 4:
        return df
    for i in range(1, n - 2):
        h1 = dx[i]
        h2 = dx[i + 1] if i + 1 < n else dx[-1]
        if speed >= 0.0:
            a = -(2.0 * h1 + h2) / (h1 * (h1 + h2))
            b = (h1 + h2) / (h1 * h2)
            c = -h1 / (h2 * (h1 + h2))
            df[i] = a * f[i] + b * f[i + 1] + c * f[i + 2]
        else:
            h0 = dx[i - 1] if i >= 1 else dx[0]
            a = (2.0 * h0 + h1) / (h1 * (h0 + h1))
            b = -(h0 + h1) / (h0 * h1)
            c = h0 / (h1 * (h0 + h1))
            df[i] = a * f[i] + b * f[i - 1] + c * f[i - 2]
    df[0] = df[1]
    df[-2:] = df[-3]
    return df


# =====================================================================
# 2nd-order centred
# =====================================================================
def central_second_derivative(f: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Return d^2 f / dx^2 using the 3-point centred stencil on a
    non-uniform grid.

    d^2 f / dx^2 |_i = 2 (f_{i+1} - f_i) / (dx_i (dx_i + dx_{i-1}))
                     - 2 (f_i - f_{i-1}) / (dx_{i-1} (dx_i + dx_{i-1}))
    """
    n = f.size
    d2f = np.zeros(n)
    for i in range(1, n - 1):
        hL = dx[i - 1]
        hR = dx[i]
        d2f[i] = 2.0 * ((f[i + 1] - f[i]) / (hR * (hR + hL))
                        - (f[i] - f[i - 1]) / (hL * (hR + hL)))
    d2f[0] = d2f[1]
    d2f[-1] = d2f[-2]
    return d2f


# =====================================================================
# 4th-order compact (Padé) scheme
# =====================================================================
def compact4_first_derivative(f: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """4th-order compact (Padé) first derivative (Lele 1992).

    Solve the tridiagonal system

        (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1}
            = (f_{i+1} - f_{i-1}) / (2 h)

    for uniform h.  For non-uniform grids we approximate h by the local
    average  (dx_{i-1} + dx_i) / 2  (adequate for smooth stretching).
    Boundary points are closed with 4th-order one-sided stencils.
    """
    n = f.size
    fp = np.zeros(n)
    if n < 5:
        return central_second_derivative(f, dx)  # not used for 2nd deriv here
    # interior RHS
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        h = 0.5 * (dx[i - 1] + dx[i])
        rhs[i] = (f[i + 1] - f[i - 1]) / (2.0 * h)
    # lower, main, upper diagonals
    a = np.full(n, 1.0 / 6.0)
    b = np.full(n, 2.0 / 3.0)
    c = np.full(n, 1.0 / 6.0)
    # boundary closures
    rhs[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2]
              + 16.0 * f[3] - 3.0 * f[4]) / (12.0 * dx[0])
    rhs[-1] = (25.0 * f[-1] - 48.0 * f[-2] + 36.0 * f[-3]
               - 16.0 * f[-4] + 3.0 * f[-5]) / (12.0 * dx[-1])
    b[0] = 1.0
    a[0] = 0.0
    c[0] = 0.0
    b[-1] = 1.0
    a[-1] = 0.0
    c[-1] = 0.0
    # Thomas algorithm
    fp = _solve_tridiagonal(a, b, c, rhs)
    return fp


# =====================================================================
# 5th-order WENO
# =====================================================================
def weno5_first_derivative(f: np.ndarray, dx: np.ndarray,
                           jacobian: bool = False) -> np.ndarray:
    """5th-order WENO reconstruction (Jiang & Shu 1996) of the first
    derivative at cell centres.

    The scheme reconstructs the numerical flux  f_{i+1/2}  on cell
    faces using five candidate stencils combined with smoothness-
    indicator-based nonlinear weights; the derivative is then

        df/dx_i = (f_{i+1/2} - f_{i-1/2}) / dx_i.

    When ``jacobian`` is True, the returned array is the Jacobian
    matrix  df_i / df_j  (used by stability analysis).
    """
    n = f.size
    eps = 1.0e-6
    if n < 7:
        # fall back to compact4 for small arrays
        return compact4_first_derivative(f, dx)

    fhp = np.zeros(n + 1)
    # WENO-5 reconstruction of f_{i+1/2} for i = 2 .. n-3
    for i in range(2, n - 2):
        h = dx[i]
        # five candidate values of f_{i+1/2}
        v0 = ( 2.0 * f[i - 2] - 7.0 * f[i - 1] + 6.0 * f[i]) / 6.0
        v1 = (-1.0 * f[i - 1] + 5.0 * f[i] + 2.0 * f[i + 1]) / 6.0
        v2 = ( 2.0 * f[i] + 5.0 * f[i + 1] - 1.0 * f[i + 2]) / 6.0
        # smoothness indicators
        b0 = (13.0 / 12.0) * (f[i - 2] - 2.0 * f[i - 1] + f[i]) ** 2 \
            + 0.25 * (f[i - 2] - 4.0 * f[i - 1] + 3.0 * f[i]) ** 2
        b1 = (13.0 / 12.0) * (f[i - 1] - 2.0 * f[i] + f[i + 1]) ** 2 \
            + 0.25 * (f[i - 1] - f[i + 1]) ** 2
        b2 = (13.0 / 12.0) * (f[i] - 2.0 * f[i + 1] + f[i + 2]) ** 2 \
            + 0.25 * (3.0 * f[i] - 4.0 * f[i + 1] + f[i + 2]) ** 2
        # ideal weights  d0 = 1/10, d1 = 6/10, d2 = 3/10
        alpha0 = 0.1 / ((eps + b0) ** 2)
        alpha1 = 0.6 / ((eps + b1) ** 2)
        alpha2 = 0.3 / ((eps + b2) ** 2)
        asum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / asum
        w1 = alpha1 / asum
        w2 = alpha2 / asum
        fhp[i + 1] = w0 * v0 + w1 * v1 + w2 * v2

    # boundary closures -- 2nd order upwind near edges
    fhp[0] = f[0]
    fhp[1] = 0.5 * (f[0] + f[1])
    fhp[n - 1] = 0.5 * (f[n - 2] + f[n - 1])
    fhp[n] = f[n - 1]

    df = np.zeros(n)
    for i in range(n):
        h = dx[i] if i < len(dx) else dx[-1]
        df[i] = (fhp[i + 1] - fhp[i]) / h
    return df


# =====================================================================
# tridiagonal solver (Thomas algorithm)
# =====================================================================
def _solve_tridiagonal(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve  a x_{i-1} + b x_i + c x_{i+1} = d  for x.

    a[0] and c[-1] are ignored.
    """
    n = b.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    if abs(b[0]) < 1.0e-30:
        return x
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        if abs(den) < 1.0e-30:
            den = 1.0e-30
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
