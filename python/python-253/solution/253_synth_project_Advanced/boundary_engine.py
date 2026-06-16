"""
boundary_engine.py — Boundary treatments, data interpolation, and auxiliary ODEs.

This module collects boundary and ancillary algorithms needed around the
main Regge-Wheeler time evolution:

  1. BOUNDARY CONDITIONS
     * Sommerfeld outgoing-wave condition  u_t + u_r* = 0  at the outer edge.
     * Hyperboloidal compactification mapping r_* in (-inf, +inf) to a
       finite interval, enabling exact outgoing BCs at a finite grid point.
     * Absorbing layer (smooth sponge) of width W near each boundary.

  2. DATA INTERPOLATION  (mapped from hand_data and area_under_curve)
     * Piecewise-linear and cubic interpolation of discrete waveform data.
     * Numerical quadrature of the strain for the integrated energy flux.
     * Area-under-the-curve computation for the total radiated energy.

  3. NONLINEAR OSCILLATOR  (mapped from duffing_ode)
     * Perturbed ringdown modelled by a Duffing-type oscillator:
           x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t).
       This captures amplitude-dependent frequency shifts that arise at
       second order in black-hole perturbation theory.

  4. LOGISTIC AMPLITUDE SATURATION  (mapped from logistic_exact)
     * Amplitude envelope of the inspiral governed by a logistic equation
           dA/dt = r A (1 - A / K),
       modelling the approach to the merger cutoff.

  5. PIECEWISE CAPPED ENVELOPE  (mapped from taxshield / call_spread)
     * Post-merger amplitude modelled as a capped long-call spread:
           A(t) = tau * max(0, min(t - t0, L)).
       This provides a smooth transition between ringdown and noise floor.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Callable, List, Optional


# ---------------------------------------------------------------------------
#  Sommerfeld outgoing-wave boundary condition
# ---------------------------------------------------------------------------
def sommerfeld_update(u: np.ndarray, dudt: np.ndarray,
                      dr: float, c: float = 1.0) -> np.ndarray:
    """Apply outgoing Sommerfeld BC at both ends.

    Left:   u_t = + c u_r   (outgoing to the left)
    Right:  u_t = - c u_r   (outgoing to the right)
    """
    dudt_new = dudt.copy()
    dudt_new[0] = c * (u[1] - u[0]) / dr
    dudt_new[-1] = -c * (u[-1] - u[-2]) / dr
    return dudt_new


# ---------------------------------------------------------------------------
#  Hyperboloidal compactification
# ---------------------------------------------------------------------------
def hyperboloidal_map(rstar: np.ndarray, R: float) -> np.ndarray:
    """Map r_* in (-inf, +inf) to rho in (0, 1) via  rho = 1 / (1 + (R/r_*)^2).

    The parameter R sets the compactification scale; rho = 0.5 at r_* = R,
    rho -> 0 at r_* -> -inf, rho -> 1 at r_* -> +inf.
    """
    return 1.0 / (1.0 + (R / (rstar + 1.0e-300)) ** 2)


def hyperboloidal_inverse(rho: np.ndarray, R: float) -> np.ndarray:
    """Inverse of the hyperboloidal map."""
    return R * np.sqrt(rho / (1.0 - rho + 1.0e-300))


# ---------------------------------------------------------------------------
#  Absorbing sponge layer
# ---------------------------------------------------------------------------
def sponge_profile(rstar: np.ndarray,
                   rstar_left: float, rstar_right: float,
                   width: float, amplitude: float = 5.0) -> np.ndarray:
    """Smooth sponge sigma(r*) that damps the solution near the boundaries.

    sigma = amplitude * ((x - x_left)/width)^2  for x in [x_left, x_left+width],
          = amplitude * ((x_right - x)/width)^2 for x in [x_right-width, x_right],
          = 0                                    otherwise.
    """
    sigma = np.zeros_like(rstar)
    left_zone = (rstar >= rstar_left) & (rstar < rstar_left + width)
    right_zone = (rstar > rstar_right - width) & (rstar <= rstar_right)
    sigma[left_zone] = amplitude * ((rstar[left_zone] - rstar_left) / width) ** 2
    sigma[right_zone] = amplitude * ((rstar_right - rstar[right_zone]) / width) ** 2
    return sigma


# ---------------------------------------------------------------------------
#  Piecewise-linear and cubic interpolation (hand_data style)
# ---------------------------------------------------------------------------
def piecewise_linear_interp(x: np.ndarray, y: np.ndarray,
                            x_new: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of data (x, y) at points x_new."""
    return np.interp(x_new, x, y)


def cubic_spline_coefficients(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Natural cubic spline coefficients for each interval.

    Returns list of (a, b, c, d) for each interval [x_i, x_{i+1}] such that
        S_i(t) = a + b t + c t^2 + d t^3,  t = x - x_i.
    """
    n = len(x) - 1
    h = np.diff(x)
    # set up the tridiagonal system for the second derivatives
    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i] * (y[i + 1] - y[i])
                    - 3.0 / h[i - 1] * (y[i] - y[i - 1]))
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    for i in range(1, n):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[-1] = 1.0
    z[-1] = 0.0
    c_arr = np.zeros(n + 1)
    b_arr = np.zeros(n)
    d_arr = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c_arr[j] = z[j] - mu[j] * c_arr[j + 1]
        b_arr[j] = ((y[j + 1] - y[j]) / h[j]
                    - h[j] * (c_arr[j + 1] + 2.0 * c_arr[j]) / 3.0)
        d_arr[j] = (c_arr[j + 1] - c_arr[j]) / (3.0 * h[j])
    coeffs = [(y[i], b_arr[i], c_arr[i], d_arr[i]) for i in range(n)]
    return coeffs


def cubic_spline_eval(x: np.ndarray, coeffs: List[Tuple[float, float, float, float]],
                      x_new: np.ndarray) -> np.ndarray:
    """Evaluate the cubic spline at new points."""
    out = np.zeros_like(x_new)
    for k, xk in enumerate(x_new):
        # find interval
        idx = int(np.searchsorted(x, xk)) - 1
        idx = max(0, min(idx, len(coeffs) - 1))
        a, b, c, d = coeffs[idx]
        t = xk - x[idx]
        out[k] = a + b * t + c * t ** 2 + d * t ** 3
    return out


# ---------------------------------------------------------------------------
#  Numerical quadrature (area under curve)
# ---------------------------------------------------------------------------
def area_under_curve(y: np.ndarray, x: np.ndarray) -> float:
    """Compute the area under the curve y(x) using the composite Simpson rule.

    Falls back to trapezoidal rule if the number of points is even.
    """
    n = len(x)
    if n < 2:
        return 0.0
    if n % 2 == 1 and n >= 3:
        # composite Simpson's 1/3 rule
        h = x[1] - x[0]
        return float((h / 3.0) * (y[0] + y[-1]
                                   + 4.0 * np.sum(y[1:-1:2])
                                   + 2.0 * np.sum(y[2:-2:2])))
    # trapezoidal fallback
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
#  Duffing oscillator (perturbed ringdown model)
# ---------------------------------------------------------------------------
def duffing_rhs(t: float, y: np.ndarray,
                alpha: float = 1.0, beta: float = 0.2,
                gamma: float = 0.3, delta: float = 0.1,
                omega: float = 1.0) -> np.ndarray:
    """Right-hand side of the Duffing oscillator:
        x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t).
    y = [x, x'].
    """
    x, v = y[0], y[1]
    dx = v
    dv = -delta * v - alpha * x - beta * x ** 3 + gamma * math.cos(omega * t)
    return np.array([dx, dv])


def integrate_duffing(t_span: Tuple[float, float], y0: np.ndarray,
                      n_steps: int = 1000, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """4th-order Runge-Kutta integration of the Duffing oscillator."""
    t = np.linspace(t_span[0], t_span[1], n_steps)
    dt = t[1] - t[0]
    y = np.zeros((n_steps, 2))
    y[0] = y0
    for i in range(n_steps - 1):
        k1 = duffing_rhs(t[i], y[i], **kwargs)
        k2 = duffing_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1, **kwargs)
        k3 = duffing_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k2, **kwargs)
        k4 = duffing_rhs(t[i] + dt, y[i] + dt * k3, **kwargs)
        y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


# ---------------------------------------------------------------------------
#  Logistic amplitude saturation
# ---------------------------------------------------------------------------
def logistic_amplitude(t: np.ndarray, r: float, K: float,
                       t0: float, A0: float) -> np.ndarray:
    """Logistic growth for the inspiral amplitude envelope.

        A(t) = K A0 exp(r (t - t0)) / (K + A0 (exp(r (t - t0)) - 1)).
    """
    ex = np.exp(np.clip(r * (t - t0), -500.0, 500.0))
    return K * A0 * ex / (K + A0 * (ex - 1.0))


# ---------------------------------------------------------------------------
#  Piecewise capped spread (post-merger envelope, from taxshield)
# ---------------------------------------------------------------------------
def capped_spread(t: np.ndarray, t0: float, L: float, tau: float) -> np.ndarray:
    """Capped linear ramp:  A(t) = tau * max(0, min(t - t0, L))."""
    return tau * np.clip(t - t0, 0.0, L)
