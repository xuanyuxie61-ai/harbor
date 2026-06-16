# -*- coding: utf-8 -*-
"""
finite_difference.py
====================
High-order compact finite-difference operators for the 1D Lagrangian
stellar structure equations.

Background
----------
In a 1D Lagrangian stellar model the independent variable is the
enclosed mass m (grams) and the dependent variables are
(r, L, T, P, rho, composition X_i).  The structure equations are

    dr/dm       = 1 / (4 pi r^2 rho)
    dP/dm       = - G m / (16 pi^2 r^4)  -  (1/c) dF_rad/dm
    dL/dm       = epsilon_nuc - epsilon_nu -  T ds/dt
    dT/dm       = nabla * (T/P) * dP/dm

where nabla is the temperature gradient (radiative, convective, or
interpolated across the boundary).

We discretise these on a non-uniform mass grid m_i (i = 0, ..., N)
using **compact (Pade) finite differences**.  Compact schemes
(Lele 1992) achieve formal O(h^p) accuracy on a wider stencil than
explicit schemes of the same order and have superior spectral
resolution, which is essential for capturing the sharp composition
gradients at shell-burning boundaries.

The schemes we implement
------------------------
1. Explicit central O(h^2), O(h^4), O(h^6)
2. Implicit (compact) tridiagonal O(h^4) :  Pad4 scheme
     alpha f'_{i-1} + f'_i + alpha f'_{i+1}
        = a (f_{i+1} - f_{i-1})/(2h) + b (f_{i+2} - f_{i-2})/(4h)
   with alpha = 1/4, a = 3/2, b = 0 for Pad4 (fourth-order).
3. Boundary closures: O(h^3) one-sided for i=0,1 and i=N-1,N.
4. Artificial dissipation: 4th-order hyperviscosity
     D f_i = -(h^4 / 16) (f_{i+2} - 4 f_{i+1} + 6 f_i
                           - 4 f_{i-1} + f_{i-2})
   added to the RHS near convectively unstable regions to suppress
   odd-even decoupling.

References
----------
  Lele, S.K., Compact finite difference schemes with spectral-like
  resolution, J. Comput. Phys. 103, 16-42, 1992.
  Kippenhahn, Weigert & Weiss, Stellar Structure and Evolution, 2nd ed.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import math


# ---------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------

def mass_grid(M_total: float, N: int,
              eta: float = 1.5) -> List[float]:
    """Construct a non-uniform Lagrangian mass grid m_i for i = 0..N.

    Grid points are concentrated toward the centre (small m) using a
    power-law mapping
        m_i = M_total * (i / N)^eta
    where eta > 1 gives central concentration.

    The parameter eta is analogous to the weight function in the CVT
    non-uniform sampling (seed 253_cvt_circle_nonuniform): it
    redistributes points to resolve the steep central gradients.
    """
    if N < 2:
        raise ValueError("mass_grid: N must be >= 2")
    if eta <= 0.0:
        eta = 1.0
    grid = [0.0] * (N + 1)
    for i in range(N + 1):
        grid[i] = M_total * (i / N) ** eta
    grid[0] = 0.0
    grid[N] = M_total
    return grid


def spacings(m: List[float]) -> List[float]:
    """Return the grid spacings h_i = m_{i+1} - m_i, i = 0..N-1."""
    N = len(m) - 1
    return [m[i+1] - m[i] for i in range(N)]


# ---------------------------------------------------------------------
# Explicit finite-difference operators
# ---------------------------------------------------------------------

def fd_central_2(f: List[float], h: List[float]) -> List[float]:
    """Second-order central first derivative on a non-uniform grid.

    For interior point i:
        f'_i = (h_{i-1}^2 (f_{i+1} - f_i) + h_i^2 (f_i - f_{i-1}))
               / (h_i h_{i-1} (h_i + h_{i-1}))

    which reduces to the standard central difference when h is uniform.
    """
    N = len(f) - 1
    df = [0.0] * (N + 1)
    # Interior
    for i in range(1, N):
        hm = h[i-1]
        hp = h[i]
        if hm + hp <= 0.0:
            df[i] = 0.0
            continue
        df[i] = (hm*hm * (f[i+1] - f[i]) + hp*hp * (f[i] - f[i-1])) / (
                  hp * hm * (hm + hp))
    # Boundaries: one-sided O(h)
    if h[0] > 0.0:
        df[0] = (f[1] - f[0]) / h[0]
    if h[-1] > 0.0:
        df[N] = (f[N] - f[N-1]) / h[-1]
    return df


def fd_central_4(f: List[float], h: List[float]) -> List[float]:
    """Fourth-order explicit first derivative on a uniform grid."""
    N = len(f) - 1
    df = [0.0] * (N + 1)
    # Interior: f'_i = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
    for i in range(2, N - 1):
        havg = 0.25 * (h[i-2] + h[i-1] + h[i] + h[i-1])
        if havg <= 0.0:
            df[i] = 0.0
            continue
        df[i] = (-f[i+2] + 8.0 * f[i+1] - 8.0 * f[i-1] + f[i-2]) / (12.0 * havg)
    # Boundaries: drop to O(h^2)
    for i in [0, 1, N-1, N]:
        if i == 0 and h[0] > 0.0 and h[1] > 0.0:
            df[i] = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0*h[0])
        elif i == N and h[-1] > 0.0 and h[-2] > 0.0:
            df[i] = (3.0*f[N] - 4.0*f[N-1] + f[N-2]) / (2.0*h[-1])
        elif h[0] > 0.0:
            df[i] = (f[min(i+1,N)] - f[max(i-1,0)]) / (2.0*h[min(i, N-1)])
    return df


# ---------------------------------------------------------------------
# Compact (Pade) fourth-order tridiagonal scheme
# ---------------------------------------------------------------------

def compact_fd4(f: List[float], h: float,
                alpha: float = 0.25) -> List[float]:
    """Compact fourth-order (Pade) first derivative on a uniform grid.

    Implicit scheme:
        alpha f'_{i-1} + f'_i + alpha f'_{i+1}
            = (3/2) * (f_{i+1} - f_{i-1}) / (2 h)
    with alpha = 1/4 gives O(h^4) with a 3-point stencil.

    The tridiagonal system is solved by the standard Thomas algorithm.
    Boundary closures use one-sided O(h^3) schemes.

    Reference: Lele 1992, eq. (2.1.7).
    """
    N = len(f) - 1
    if N < 4:
        return fd_central_2(f, [h] * N)

    # Right-hand side
    a = 1.5
    rhs = [0.0] * (N + 1)
    for i in range(1, N):
        rhs[i] = a * (f[i+1] - f[i-1]) / (2.0 * h)

    # Solve alpha * x_{i-1} + x_i + alpha * x_{i+1} = rhs_i
    # Thomas algorithm
    n = N + 1
    a_sub = [alpha] * n    # sub-diagonal
    b_diag = [1.0] * n     # diagonal
    c_sup = [alpha] * n    # super-diagonal

    # Boundary closures: modified first/last equations of the system
    # use one-sided O(h^2) formulas:
    #   2 f'_0 + f'_1 = (-5 f_0 + 4 f_1 + f_2) / (2 h)
    #   f'_{N-1} + 2 f'_N = (5 f_N - 4 f_{N-1} - f_{N-2}) / (2 h)
    if N >= 2:
        a_sub[0] = 0.0
        b_diag[0] = 2.0
        c_sup[0] = 1.0
        rhs[0] = (-5.0*f[0] + 4.0*f[1] + f[2]) / (2.0*h)
        a_sub[N] = 1.0
        b_diag[N] = 2.0
        c_sup[N] = 0.0
        rhs[N] = (5.0*f[N] - 4.0*f[N-1] - f[N-2]) / (2.0*h)

    d = rhs[:]

    # Forward elimination
    for i in range(1, n):
        if abs(b_diag[i-1]) < 1.0e-300:
            break
        m = a_sub[i] / b_diag[i-1]
        b_diag[i] -= m * c_sup[i-1]
        d[i] -= m * d[i-1]

    # Back substitution
    x = [0.0] * n
    if abs(b_diag[n-1]) < 1.0e-300:
        return x
    x[n-1] = d[n-1] / b_diag[n-1]
    for i in range(n-2, -1, -1):
        if abs(b_diag[i]) < 1.0e-300:
            x[i] = 0.0
        else:
            x[i] = (d[i] - c_sup[i] * x[i+1]) / b_diag[i]
    return x


# ---------------------------------------------------------------------
# Artificial dissipation (hyperviscosity)
# ---------------------------------------------------------------------

def hyperviscosity(f: List[float], h: float,
                   coeff: float = 1.0/16.0) -> List[float]:
    """Apply fourth-order hyperviscosity  D f_i = -coeff * h^4 *
    (f_{i+2} - 4 f_{i+1} + 6 f_i - 4 f_{i-1} + f_{i-2}).

    Near the boundaries where the stencil is incomplete we drop the
    contribution.

    The hyperviscosity coefficient 1/16 is chosen to damp the highest
    resolvable wavenumber (pi / h) with time-scale  tau ~ h^4 / (coeff)
    while leaving large scales essentially unaffected.
    """
    N = len(f) - 1
    Df = [0.0] * (N + 1)
    h4 = h**4
    for i in range(2, N - 1):
        Df[i] = -coeff * h4 * (
            f[i+2] - 4.0*f[i+1] + 6.0*f[i] - 4.0*f[i-1] + f[i-2])
    return Df


# ---------------------------------------------------------------------
# Second derivative (explicit O(h^2))
# ---------------------------------------------------------------------

def fd_second_2(f: List[float], h: List[float]) -> List[float]:
    """Second derivative with O(h^2) central on non-uniform grid.

    For interior i:
        f''_i = 2 * ((f_{i+1} - f_i)/h_i - (f_i - f_{i-1})/h_{i-1})
                / (h_i + h_{i-1})
    """
    N = len(f) - 1
    d2f = [0.0] * (N + 1)
    for i in range(1, N):
        hm = h[i-1]; hp = h[i]
        if hm + hp <= 0.0:
            continue
        d2f[i] = 2.0 * ((f[i+1] - f[i])/hp - (f[i] - f[i-1])/hm) / (hm + hp)
    if N >= 1 and h[0] > 0.0 and len(h) >= 2 and h[1] > 0.0:
        d2f[0] = (f[2] - 2.0*f[1] + f[0]) / (h[0] * h[0]) if h[0] == h[1] else 0.0
    if N >= 1 and h[-1] > 0.0 and len(h) >= 2 and h[-2] > 0.0:
        d2f[N] = (f[N] - 2.0*f[N-1] + f[N-2]) / (h[-1] * h[-1]) if h[-1] == h[-2] else 0.0
    return d2f


# ---------------------------------------------------------------------
# Von Neumann stability check for explicit advection-diffusion
# ---------------------------------------------------------------------

def von_neumann_max_dt(velocity: float, diffusivity: float,
                       h: float, scheme: str = "central4") -> float:
    """Return the maximum stable time-step for an explicit
    advection-diffusion equation
        u_t + v u_x = D u_{xx}
    discretised on a uniform grid of spacing h.

    For scheme "central2" (FTCS, O(h^2)):
        |v| dt / h <= 1,   2 D dt / h^2 <= 1
    =>   dt_max = min(h/|v|, h^2/(2D))  if D > 0 else h/|v|

    For scheme "central4":
        the stability limit tightens by ~ factor 1.5 due to the
        wider stencil; we use an empirical factor 2/3.

    This is the von Neumann necessary condition (sufficient for
    linear problems with constant coefficients).
    """
    if h <= 0.0:
        return 0.0
    dt_adv = h / max(abs(velocity), 1.0e-30)
    if diffusivity > 0.0:
        dt_diff = h * h / (2.0 * diffusivity)
    else:
        dt_diff = float("inf")
    factor = 2.0/3.0 if scheme == "central4" else 1.0
    return factor * min(dt_adv, dt_diff)


# ---------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------

def convergence_order(f_func, df_exact, h_values, scheme="compact4"):
    """Empirical convergence order test on a smooth function."""
    import math
    errors = []
    for h in h_values:
        N = max(8, int(1.0 / h))
        m = [i * h for i in range(N + 1)]
        f = [f_func(mi) for mi in m]
        if scheme == "compact4":
            dfn = compact_fd4(f, h)
        elif scheme == "central4":
            dfn = fd_central_4(f, [h]*N)
        else:
            dfn = fd_central_2(f, [h]*N)
        err = max(abs(dfn[i] - df_exact(m[i])) for i in range(1, N))
        errors.append(err)
    orders = []
    for i in range(1, len(h_values)):
        if errors[i-1] > 0 and errors[i] > 0 and h_values[i-1] != h_values[i]:
            orders.append(math.log(errors[i-1]/errors[i]) /
                          math.log(h_values[i-1]/h_values[i]))
    return orders
