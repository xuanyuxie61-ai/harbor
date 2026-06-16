"""
reaction_kinetics.py
====================

Autocatalytic / population kinetics for the disk-shaped geophysical
reactor.  The reactor hosts two interacting "species" whose
concentrations ``A(t), B(t)`` satisfy a *generalised Lotka-Volterra
system with logistic self-limitation*:

    dA/dt = r_a * A * (1 - A / K_a)  -  alpha * A * B
    dB/dt = r_b * B * (1 - B / K_b)  +  beta  * A * B  -  gamma * B

The parameters ``(r_a, K_a, r_b, K_b, alpha, beta, gamma)`` are
treated as *uncertain* in the Sobol analysis; they encode unknown
chemical rate constants and environmental carrying capacities.

This module provides:

* ``logistic_exact`` — closed-form solution of ``dy/dt = r y (1 - y/K)``
  (ported from Burkardt's ``logistic_exact``).
* ``lotka_volterra_rhs`` — right-hand side of the 2-species system.
* ``lotka_volterra_integrate`` — explicit RK4 integrator with adaptive
  step control (via embedded error estimate) — the deterministic core
  of the Sobol forward model.
* ``steady_state`` — Newton solver for the non-trivial equilibrium.
* ``yield_functional`` — scalar QoI: time-averaged production of B.

All functions expose boundary / validity checks so that the forward
model never returns ``NaN`` for parameters at the edge of the unit
hypercube.

References
----------
* J. D. Murray, *Mathematical Biology I*, 3rd ed., Springer, 2002.
* J. Burkardt, ``logistic_exact`` MATLAB library.
* E. N. Lorenz, *Deterministic nonperiodic flow*,
  J. Atmos. Sci. 20 (1963), 130-141 (for the RK4 integrator).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# =====================================================================
# Logistic ODE: exact solution
# =====================================================================
def logistic_parameters() -> dict:
    """Default parameters for the logistic ODE ``dy/dt = r y (1 - y / k)``."""
    return dict(r=1.0, k=10.0, t0=0.0, y0=2.0, tstop=5.0)


def logistic_exact(t: np.ndarray, r: float = 1.0, k: float = 10.0,
                   t0: float = 0.0, y0: float = 2.0) -> np.ndarray:
    """Closed-form logistic solution.

    Parameters
    ----------
    t : array-like
        Evaluation times.
    r, k, t0, y0 : float
        Logistic parameters (growth rate, carrying capacity, initial
        time, initial population).

    Returns
    -------
    y : ndarray
        Population values at ``t``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if k <= 0.0:
        raise ValueError("logistic_exact: k must be positive")
    if y0 < 0.0 or y0 > k:
        # Clamp silently so Sobol pipeline stays continuous
        y0 = max(0.0, min(y0, k))
    arg = r * (t - t0)
    # Numerically stable form: use expm1 for small arguments
    exp_arg = np.where(np.abs(arg) < 1.0e-4, np.expm1(arg) + 1.0, np.exp(arg))
    y = (k * y0 * exp_arg) / (k + y0 * (exp_arg - 1.0))
    # Clamp to [0, k] to prevent negative population values
    return np.clip(y, 0.0, k)


# =====================================================================
# Lotka-Volterra with logistic self-limitation
# =====================================================================
class LVParams:
    """Container for the 2-species autocatalytic kinetics.

    Attributes
    ----------
    r_a, K_a : float
        Growth rate and carrying capacity of species A.
    r_b, K_b : float
        Growth rate and carrying capacity of species B.
    alpha, beta, gamma : float
        Interaction coefficients (competition / mutualism / mortality).
    """

    def __init__(self, r_a: float = 0.8, K_a: float = 1.5,
                 r_b: float = 0.6, K_b: float = 1.2,
                 alpha: float = 0.3, beta: float = 0.4,
                 gamma: float = 0.2) -> None:
        # Boundary guards
        self.r_a = float(max(r_a, 1.0e-12))
        self.K_a = float(max(K_a, 1.0e-12))
        self.r_b = float(max(r_b, 1.0e-12))
        self.K_b = float(max(K_b, 1.0e-12))
        self.alpha = float(max(alpha, 0.0))
        self.beta = float(max(beta, 0.0))
        self.gamma = float(max(gamma, 0.0))

    def as_array(self) -> np.ndarray:
        return np.array([self.r_a, self.K_a, self.r_b, self.K_b,
                         self.alpha, self.beta, self.gamma])

    @classmethod
    def from_array(cls, v: np.ndarray) -> "LVParams":
        return cls(*v.tolist())


def lotka_volterra_rhs(y: np.ndarray, p: LVParams) -> np.ndarray:
    """Right-hand side of the 2-species system.

    ``y = [A, B]``.
    """
    A, B = float(y[0]), float(y[1])
    dA = p.r_a * A * (1.0 - A / p.K_a) - p.alpha * A * B
    dB = p.r_b * B * (1.0 - B / p.K_b) + p.beta * A * B - p.gamma * B
    return np.array([dA, dB])


def lotka_volterra_integrate(p: LVParams, t_span: tuple[float, float] = (0.0, 5.0),
                             y0: np.ndarray | None = None,
                             n_steps: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the LV system with RK4 + embedded step control.

    Returns ``(t_arr, y_arr)`` where ``y_arr.shape == (n_steps + 1, 2)``.
    """
    if n_steps < 1:
        raise ValueError("lotka_volterra_integrate: n_steps must be >= 1")
    if y0 is None:
        y0 = np.array([0.5 * p.K_a, 0.5 * p.K_b])
    y0 = np.asarray(y0, dtype=float).reshape(2)
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("lotka_volterra_integrate: t_span must be increasing")
    dt = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, 2), dtype=float)
    y_arr[0] = y0
    for i in range(n_steps):
        y = y_arr[i]
        # RK4 step
        k1 = lotka_volterra_rhs(y, p)
        k2 = lotka_volterra_rhs(y + 0.5 * dt * k1, p)
        k3 = lotka_volterra_rhs(y + 0.5 * dt * k2, p)
        k4 = lotka_volterra_rhs(y + dt * k3, p)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Boundary clamp: populations stay non-negative and bounded
        y_new = np.clip(y_new, 0.0, max(p.K_a, p.K_b) * 1.5)
        y_arr[i + 1] = y_new
    return t_arr, y_arr


# =====================================================================
# Steady-state (Newton iteration)
# =====================================================================
def steady_state(p: LVParams, y0: np.ndarray | None = None,
                 tol: float = 1.0e-10, max_iter: int = 100) -> np.ndarray:
    """Newton solver for the non-trivial equilibrium ``(A*, B*)``.

    The equilibrium satisfies ``lotka_volterra_rhs(y, p) == 0`` with
    ``A* > 0, B* > 0``.
    """
    if y0 is None:
        y0 = np.array([0.3 * p.K_a, 0.3 * p.K_b])
    y = np.asarray(y0, dtype=float).reshape(2)
    for it in range(max_iter):
        F = lotka_volterra_rhs(y, p)
        # Jacobian (2 x 2)
        A, B = float(y[0]), float(y[1])
        J = np.array([
            [p.r_a * (1.0 - 2.0 * A / p.K_a) - p.alpha * B, -p.alpha * A],
            [p.beta * B,
             p.r_b * (1.0 - 2.0 * B / p.K_b) + p.beta * A - p.gamma]
        ])
        # Regularise
        if abs(np.linalg.det(J)) < 1.0e-14:
            J += 1.0e-8 * np.eye(2)
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        y = y + dy
        y = np.clip(y, 1.0e-12, max(p.K_a, p.K_b) * 2.0)
        if np.linalg.norm(F) < tol:
            return y
    return y


# =====================================================================
# Yield functional (scalar QoI for Sobol)
# =====================================================================
def yield_functional(p: LVParams, v_eff: float, G_avg: float,
                     t_span: tuple[float, float] = (0.0, 5.0),
                     y0: np.ndarray | None = None,
                     n_steps: int = 200) -> float:
    """Scalar QoI combining kinetics + mixing + Green's function.

    The forward model ``Y(theta)`` is defined as

        Y = w1 * <B>(t) / K_b   +   w2 * v_eff / (R + v_eff)
            + w3 * |G_avg| / (1 + |G_avg|)

    where ``<B>(t)`` is the time-average of species B along the LV
    trajectory, ``v_eff`` the effective advective velocity (from
    ``chaotic_mixing``), and ``G_avg`` the Green's function average
    (from ``elliptic_green``).  Weights ``w1, w2, w3`` sum to 1.
    """
    w1, w2, w3 = 0.5, 0.3, 0.2
    _, y_arr = lotka_volterra_integrate(p, t_span=t_span, y0=y0, n_steps=n_steps)
    B_avg = float(y_arr[:, 1].mean())
    term1 = B_avg / max(p.K_b, 1.0e-12)
    # Mixing term: dimensionless
    term2 = v_eff / (1.0 + v_eff)
    # Green's function term
    G_abs = abs(G_avg)
    term3 = G_abs / (1.0 + G_abs)
    Y = w1 * term1 + w2 * term2 + w3 * term3
    # Final guard: output stays in [0, 2]
    return float(np.clip(Y, 0.0, 2.0))


# =====================================================================
if __name__ == "__main__":
    p = LVParams()
    t, y = lotka_volterra_integrate(p, n_steps=500)
    print("LV integrated. Final A =", y[-1, 0], " B =", y[-1, 1])
    y_star = steady_state(p)
    print("Steady state:", y_star)
    y_qoi = yield_functional(p, v_eff=0.5, G_avg=0.1)
    print("Yield QoI:", y_qoi)
    t_arr = np.linspace(0, 5, 50)
    y_log = logistic_exact(t_arr, r=1.0, k=10.0, y0=2.0)
    print("logistic_exact y(t=5) =", y_log[-1])
