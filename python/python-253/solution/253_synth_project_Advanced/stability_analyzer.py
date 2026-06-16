"""
stability_analyzer.py — Von Neumann stability analysis and CFL conditions.

For the model wave equation
    u_tt = c^2 u_xx
discretised with second-order centred time stepping (leapfrog / Störmer-Verlet)
and a p-th order centred spatial operator D2, the fully-discrete scheme reads

    u^{n+1}_j = 2 u^n_j - u^{n-1}_j + (c dt)^2 (D2 u^n)_j.

The von Neumann ansatz  u^n_j = g^n exp(i j theta)  leads to the
amplification-factor quadratic
    g^2 - 2 A g + 1 = 0,   where  A = 1 + (c dt/dx)^2 sigma(theta) dx^2 / 2.

Stability  |g| <= 1  requires  A in [-1, +1], which translates into the CFL
condition   dt <= dx / (c * sigma_max^{1/2}).

This module:
  * computes the CFL limit analytically for each FD order,
  * evaluates the maximum amplification factor for a given dt/dx ratio,
  * performs a sampled spectral-radius scan,
  * determines the critical Courant number for higher-order stencils,
  * reports the dissipation and dispersion errors as functions of theta.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple

from high_order_fd import STENCIL_D2, spectral_symbol_d2


# ---------------------------------------------------------------------------
#  CFL limit from the spectral minimum of the FD symbol
# ---------------------------------------------------------------------------
def cfl_limit(order: int, n_sample: int = 8192) -> float:
    """Return the critical Courant number  C_max = dt/dx  for stability.

    For a p-th order centred second-derivative operator with symbol sigma(theta),
    the CFL limit is
        C_max = 1 / sqrt( max_theta | sigma(theta) dx^2 | ).
    Since sigma(theta) dx^2 is non-dimensional we denote mu_max = -min sigma dx^2
    and  C_max = 1 / sqrt(mu_max).
    """
    theta = np.linspace(0.0, math.pi, n_sample)
    sigma_dx2 = np.zeros_like(theta)
    c = STENCIL_D2[order]
    s = len(c) // 2
    # sigma dx^2 = c_0 + 2 sum c_k cos(k theta)
    sigma_dx2 += c[s]
    for k in range(1, s + 1):
        sigma_dx2 += 2.0 * c[s + k] * np.cos(k * theta)
    # sigma is non-positive; we need its minimum (= most negative value)
    sigma_min = float(np.min(sigma_dx2))
    if sigma_min >= 0.0:
        return float("inf")
    return 1.0 / math.sqrt(-sigma_min)


def max_amplification(order: int, courant: float,
                      n_sample: int = 4096) -> float:
    """Compute max_{theta} |g(theta)| for a given Courant number.

    Returns
    -------
    g_max : maximum amplification factor; stable iff g_max <= 1 + eps.
    """
    theta = np.linspace(0.0, math.pi, n_sample)
    c = STENCIL_D2[order]
    s = len(c) // 2
    sigma_dx2 = c[s] + sum(2.0 * c[s + k] * np.cos(k * theta)
                           for k in range(1, s + 1))
    # A(theta) = 1 + 0.5 C^2 sigma_dx2
    A = 1.0 + 0.5 * courant ** 2 * sigma_dx2
    # roots of g^2 - 2 A g + 1 = 0
    disc = A ** 2 - 1.0
    # when disc < 0, roots are complex conjugates with |g| = 1
    # when disc >= 0, |g| = |A| + sqrt(disc)  for the growing root
    g_growing = np.where(disc >= 0.0,
                         np.abs(A) + np.sqrt(np.maximum(disc, 0.0)),
                         1.0)
    return float(np.max(g_growing))


# ---------------------------------------------------------------------------
#  Sampled spectral scan
# ---------------------------------------------------------------------------
def spectral_scan(order: int, courant: float,
                  n_sample: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return (theta, |g(theta)|) over the Brillouin zone [0, pi]."""
    theta = np.linspace(0.0, math.pi, n_sample)
    c = STENCIL_D2[order]
    s = len(c) // 2
    sigma_dx2 = c[s] + sum(2.0 * c[s + k] * np.cos(k * theta)
                           for k in range(1, s + 1))
    A = 1.0 + 0.5 * courant ** 2 * sigma_dx2
    disc = A ** 2 - 1.0
    g_growing = np.where(disc >= 0.0,
                         np.abs(A) + np.sqrt(np.maximum(disc, 0.0)),
                         1.0)
    return theta, g_growing


# ---------------------------------------------------------------------------
#  Dispersion and dissipation errors
# ---------------------------------------------------------------------------
def numerical_phase_speed(order: int, courant: float,
                          theta: np.ndarray) -> np.ndarray:
    """Compute the numerical phase speed  c_num / c  as a function of theta.

    For the exact wave  omega = c k  and theta = k dx, so
    omega_num dt = arccos(A(theta))  and
    c_num / c = omega_num / (c k) = arccos(A) / (C theta).
    """
    c = STENCIL_D2[order]
    s = len(c) // 2
    sigma_dx2 = c[s] + sum(2.0 * c[s + k] * np.cos(k * theta)
                           for k in range(1, s + 1))
    A = 1.0 + 0.5 * courant ** 2 * sigma_dx2
    A_safe = np.clip(A, -1.0, 1.0)
    omega_num_dt = np.arccos(A_safe)
    # avoid div-by-zero at theta = 0
    safe = theta > 1.0e-12
    ratio = np.zeros_like(theta)
    ratio[safe] = omega_num_dt[safe] / (courant * theta[safe])
    return ratio


def dissipation_rate(order: int, courant: float,
                     theta: np.ndarray) -> np.ndarray:
    """Return Im(omega_num) dt, which should be zero for a non-dissipative scheme.

    For the leapfrog scheme with a symmetric stencil this is identically zero
    as long as |A| <= 1.  When |A| > 1 we have purely growing/decaying modes.
    """
    c = STENCIL_D2[order]
    s = len(c) // 2
    sigma_dx2 = c[s] + sum(2.0 * c[s + k] * np.cos(k * theta)
                           for k in range(1, s + 1))
    A = 1.0 + 0.5 * courant ** 2 * sigma_dx2
    # when |A| > 1, |g| = |A| + sqrt(A^2-1), so Im(omega) = log|g| / dt
    disc = np.maximum(A ** 2 - 1.0, 0.0)
    g_abs = np.where(np.abs(A) <= 1.0, 1.0, np.abs(A) + np.sqrt(disc))
    return np.log(g_abs + 1.0e-300)


# ---------------------------------------------------------------------------
#  Report helper
# ---------------------------------------------------------------------------
def stability_report(orders: Tuple[int, ...] = (2, 4, 6, 8, 10),
                     courant: float = 0.5) -> Dict[str, float]:
    """Produce a summary dict with CFL limits and max amplification."""
    report = {}
    for p in orders:
        cmax = cfl_limit(p)
        gmax = max_amplification(p, courant)
        report[f"order_{p}"] = {"cfl_max": cmax,
                                "g_max_at_C={courant}": gmax}
    return report


# ---------------------------------------------------------------------------
#  CFL table for direct use by the time integrator
# ---------------------------------------------------------------------------
def cfl_table() -> Dict[int, float]:
    """Return {order: C_max} for the supported FD orders."""
    return {p: cfl_limit(p) for p in (2, 4, 6, 8, 10)}
