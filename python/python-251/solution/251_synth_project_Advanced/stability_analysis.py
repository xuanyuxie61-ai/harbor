"""
stability_analysis.py
=====================
von Neumann stability analysis and CFL timestep selection for the
high-order MHD scheme, plus an implicit pressure solve based on the
Cholesky factorisation of the symmetric positive-definite operator
that arises from the low-Mach-number projection.

The module combines three distinct seeds:

    * The classical Cholesky factorisation of AS algorithm 7
      (026_asa007) is used as the core linear solver for the implicit
      pressure Poisson equation in the low-Mach projection step.
    * The integer bisection (095) is used to find the coarsest CFL
      number that still keeps the von Neumann amplification factor
      below unity for a given grid.
    * The Monte-Carlo sampling (533) is used to estimate the
      probability that a given dt remains stable when the maximum
      wave speed fluctuates due to turbulence.

Linear stability analysis
-------------------------
For the linearised advection equation u_t + a u_x = 0 discretised
with compact 4th-order space and SSP-RK3 time integration, the
amplification factor is

    G(theta) = 1 + z + z^2 / 2 + z^3 / 6

where z = -i a dt / dx * phi_4(theta) and phi_4 is the modified wave
number of the compact scheme

    phi_4(theta) = (3/2) sin(theta) / (1 + (1/2) cos(theta))

The scheme is stable when |G| <= 1 for all theta in [0, pi], which
gives a critical CFL number CFL_crit ~ 1.7 for RK3 with this spatial
operator (compared to sqrt(3) ~ 1.73 for RK3 with a 2nd-order scheme).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple

import physical_constants as pc
import high_order_fd as hfd
import mhd_equations as mhd


# ---------------------------------------------------------------------------
#              von Neumann amplification factor (compact4 + RK3)
# ---------------------------------------------------------------------------
def amplification_factor(theta: np.ndarray, cfl: float) -> np.ndarray:
    """Return |G(theta)| for compact-4 space + SSP-RK3 time.

    Parameters
    ----------
    theta : ndarray
        Non-dimensional wave numbers in [0, pi].
    cfl : float
        Courant number a dt / dx.
    """
    # Modified wave number of the compact 4th scheme
    phi4 = (3.0 / 2.0) * np.sin(theta) / (1.0 + 0.5 * np.cos(theta) + 1.0e-30)
    # z = -i * CFL * phi4
    z = -1j * cfl * phi4
    # SSP-RK3 amplification polynomial: G = 1 + z + z^2/2 + z^3/6
    G = 1.0 + z + 0.5 * z**2 + (1.0 / 6.0) * z**3
    return np.abs(G)


def max_amplification(cfl: float, n_theta: int = 2048) -> float:
    """Return max_theta |G(theta, cfl)|."""
    theta = np.linspace(0.0, math.pi, n_theta)
    return float(np.max(amplification_factor(theta, cfl)))


def critical_cfl(n_theta: int = 2048, tol: float = 1.0e-4) -> float:
    """Find the critical CFL for which max |G| = 1 (stability limit).

    Uses the integer bisection algorithm (095) on a rescaled CFL axis:
    we bisect on n_cfl in [1, 300] representing CFL = n_cfl / 100.
    """
    def residual(n_cfl: int) -> int:
        cfl = n_cfl / 100.0
        Gmax = max_amplification(cfl, n_theta)
        # +1 if stable, -1 if unstable (sign-change convention)
        return 1 if Gmax <= 1.0 + tol else -1

    a, b = 1, 300
    fa, fb = residual(a), residual(b)
    if fa == fb:
        return 3.0  # fallback
    while abs(b - a) > 1:
        c = (a + b) // 2
        fc = residual(c)
        if fc == fa:
            a, fa = c, fc
        else:
            b, fb = c, fc
    return a / 100.0


# ---------------------------------------------------------------------------
#                       Global CFL timestep
# ---------------------------------------------------------------------------
def cfl_timestep(U: np.ndarray, g,
                 safety: float = 0.4) -> float:
    """Return dt = safety * min(dx, dy, dz) / (|v| + c_f)_max."""
    cf_max = mhd.max_signal_speed(U)
    dx_min = float(np.min(g.dx))
    dy_min = float(np.min(g.dy))
    dz_min = float(np.min(g.dz))
    d_min = min(dx_min, dy_min, dz_min)
    if cf_max <= 0:
        cf_max = 1.0
    return safety * d_min / cf_max


# ---------------------------------------------------------------------------
#         Monte-Carlo estimate of stability probability (from 533)
# ---------------------------------------------------------------------------
def stability_probability(U: np.ndarray, g,
                          dt: float,
                          trials: int = 64,
                          sigma_v: float = 0.05) -> float:
    """Estimate P(stable | fluctuating wave speed) by Monte-Carlo.

    The turbulent velocity fluctuations add a random component to the
    maximum signal speed; we model this as Gaussian with standard
    deviation ``sigma_v`` relative to the mean and count how often
    the resulting dt * (|v| + c_f) / dx remains below the critical
    CFL.
    """
    scales = pc.derived_scales()
    cf_mean = mhd.max_signal_speed(U)
    dx_min = min(float(np.min(g.dx)), float(np.min(g.dy)), float(np.min(g.dz)))
    cfl_crit = critical_cfl()

    rng = np.random.default_rng(pc.get("seed") + int(dt * 1000) % 10_000)
    fluctuations = rng.normal(1.0, sigma_v, size=trials)
    stable_count = 0
    for f in fluctuations:
        cf_sample = cf_mean * max(f, 0.1)
        cfl_sample = dt * cf_sample / dx_min
        if cfl_sample <= cfl_crit:
            stable_count += 1
    return stable_count / trials


# ---------------------------------------------------------------------------
#      Cholesky-based pressure Poisson solver (from 026_asa007)
# ---------------------------------------------------------------------------
def cholesky_factor(A: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Compute the Cholesky factor U of a symmetric positive-definite
    matrix A, returning (U, nullty, ifault).

    This is a direct Python translation of cholesky.m (026_asa007).
    The matrix is passed as a dense 2-D array and the factor U is
    upper triangular such that A = U^T U.
    """
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("cholesky_factor: matrix must be square")
    eta = 1.0e-9
    U = np.zeros((n, n))
    nullty = 0
    ifault = 0

    for icol in range(n):
        ii_diag = icol
        x = eta * eta * A[icol, icol]
        for irow in range(icol + 1):
            w = A[irow, icol]
            for k in range(irow):
                w -= U[k, irow] * U[k, icol]
            if irow == icol:
                break
            if abs(U[irow, irow]) > 1.0e-30:
                U[irow, icol] = w / U[irow, irow]
            else:
                U[irow, icol] = 0.0
                if abs(x * A[irow, icol]) < w * w:
                    ifault = 2
                    return U, nullty, ifault
        w_final = A[icol, icol]
        for k in range(icol):
            w_final -= U[k, icol] ** 2
        if abs(w_final) <= abs(eta * A[icol, icol]):
            U[icol, icol] = 0.0
            nullty += 1
        elif w_final < 0.0:
            ifault = 2
            return U, nullty, ifault
        else:
            U[icol, icol] = math.sqrt(w_final)
    return U, nullty, ifault


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b via Cholesky factorisation (fallback to least-squares
    if the matrix is rank-deficient).
    """
    U, nullty, ifault = cholesky_factor(A)
    if ifault != 0 or nullty > 0:
        # Fall back to a robust least-squares solve
        return np.linalg.lstsq(A, b, rcond=None)[0]
    # Forward substitution: U^T y = b
    n = A.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= U[k, i] * y[k]
        y[i] = s / U[i, i] if abs(U[i, i]) > 1.0e-30 else 0.0
    # Back substitution: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= U[i, k] * x[k]
        x[i] = s / U[i, i] if abs(U[i, i]) > 1.0e-30 else 0.0
    return x


# ---------------------------------------------------------------------------
#              von Neumann report (textual, no plotting)
# ---------------------------------------------------------------------------
def von_neumann_report() -> str:
    """Return a multi-line text report of the von Neumann analysis."""
    cfl_crit = critical_cfl()
    lines = [
        "=== von Neumann stability analysis ===",
        f"  Scheme         : compact-4 space + SSP-RK3 time",
        f"  Critical CFL   : {cfl_crit:.4f}",
        f"  Recommended CFL: {0.4 * cfl_crit:.4f}  (safety = 0.4)",
    ]
    theta = np.linspace(0.0, math.pi, 16)
    lines.append("  theta        |G| @ CFL=0.5  |G| @ CFL=C_crit")
    for th in theta:
        g1 = amplification_factor(np.array([th]), 0.5)[0]
        g2 = amplification_factor(np.array([th]), cfl_crit)[0]
        lines.append(f"  {th:8.5f}     {g1:8.5f}        {g2:8.5f}")
    return "\n".join(lines)
