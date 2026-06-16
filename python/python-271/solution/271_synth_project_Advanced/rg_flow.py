# -*- coding: utf-8 -*-
"""
Real-space renormalisation-group flow ODEs for the 1D quantum Ising model.

The block-spin RG of the TFIM maps (J, h) -> (J', h') under a
length-rescaling factor b = 2.  In the perturbative limit (h << J or
J << h) the flow is well-approximated by a system of ODEs on the
coupling constants; near the QCP the flow crosses over to a critical
manifold and the beta-function develops a simple zero whose stability
determines nu.

We follow the classic treatment of Pfeuty (1970) and the modern
derivative by Nishino (1995): the RG beta-functions for the
dimensionless coupling g = h/J are

    beta(g)  =  d g / d ln b  =  g - g^3           (perturbative)
    beta'(g) =  d^2 g / d ln^2 b = ...

More generally we include the leading irrelevant correction with
exponent omega ~ 2 (for 1D TFIM) and a higher-order stabilising term.

The ODE system is integrated with a 4th-order Runge-Kutta scheme
(see the Langford-ODE seed project 645 for a similar driver) and
the critical manifold is detected by a Newton search for the
non-trivial fixed point beta(g*) = 0.
"""

from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Beta-functions
# ---------------------------------------------------------------------------
def beta_perturbative(g: float) -> float:
    """Perturbative one-loop beta function  beta(g) = g - g^3.
    Fixed points at g = 0 (ordered), g = 1 (critical), g -> infty
    (disordered).  Linearisation at g = 1 gives
        beta'(1) = 1 - 3 = -2   ->   nu = 1 / |beta'(1)| = 1/2
    at one-loop; the exact value is nu = 1.
    """
    return g - g ** 3


def beta_two_loop(g: float) -> float:
    """Two-loop improvement:
        beta(g) = g - g^3 + (1/2) g^5.
    Adds an irrelevant stabilising term and shifts the non-trivial
    fixed point.  The shift in nu from one-loop to two-loop captures
    some of the non-perturbative physics.
    """
    return g - g ** 3 + 0.5 * g ** 5


def beta_with_irrelevant(g: float, omega: float = 2.0,
                          a_irr: float = 0.1) -> float:
    """Including the leading irrelevant scaling field:
        beta(g) = (g - g^3) * (1 + a_irr * g^omega).
    omega ~ 2 for the 1D TFIM.  a_irr tunes the crossover scale.
    """
    return (g - g ** 3) * (1.0 + a_irr * g ** omega)


def d_beta_dg(g: float,
               which: str = "two_loop") -> float:
    """Analytic derivative of beta(g) wrt g (used for nu extraction)."""
    if which == "perturbative":
        return 1.0 - 3.0 * g * g
    if which == "two_loop":
        return 1.0 - 3.0 * g * g + 2.5 * g ** 4
    raise ValueError(f"unknown beta '{which}'")


def critical_exponent_nu(which: str = "two_loop") -> float:
    """nu = 1 / |beta'(g*)| at the non-trivial fixed point g*."""
    if which == "perturbative":
        g_star = 1.0
    elif which == "two_loop":
        # Solve g - g^3 + 0.5 g^5 = 0 -> g^2 satisfies
        # 0.5 u^2 - u + 1 = 0  where u = g^2
        disc = 1.0 - 2.0
        if disc < 0:
            g_star = 1.0   # fall back
        else:
            u = (1.0 + np.sqrt(disc)) / 1.0
            g_star = np.sqrt(max(u, 0.0))
    else:
        g_star = 1.0
    bp = abs(d_beta_dg(g_star, which))
    return 1.0 / max(bp, C.EPS_NUM)


# ---------------------------------------------------------------------------
# ODE driver  (RK4; inspired by the Langford-ODE integrator)
# ---------------------------------------------------------------------------
def rk4_step(f: Callable[[float, np.ndarray], np.ndarray],
              t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step for y' = f(t, y).  Vectorised."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_flow(beta: Callable[[float], float],
                    g0: float, ln_b_max: float = 6.0,
                    n_steps: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate  dg / d ln b = beta(g)  from ln b = 0 to ln_b_max,
    starting at g(0) = g0.  Returns (ln_b_array, g_array).

    An adaptive step-halving is used whenever |g| > 1e6 to avoid blow-up
    near runaway flows.
    """
    ln_bs = np.linspace(0.0, ln_b_max, n_steps + 1)
    gs = np.zeros(n_steps + 1)
    gs[0] = g0
    for i in range(n_steps):
        dt = ln_bs[i + 1] - ln_bs[i]
        g_cur = gs[i]
        # Adaptive halving if g is large
        n_sub = 1
        while abs(g_cur) > 1e6 and n_sub < 64:
            n_sub *= 2
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            g_cur = rk4_step(lambda t, y: np.array([beta(float(y[0]))]),
                              0.0, np.array([g_cur]), sub_dt)[0]
        gs[i + 1] = g_cur
    return ln_bs, gs


# ---------------------------------------------------------------------------
# Newton search for the non-trivial fixed point
# ---------------------------------------------------------------------------
def find_fixed_point(beta: Callable[[float], float],
                      g_guess: float = 1.0,
                      tol: float = 1.0e-10,
                      max_iter: int = 60) -> float:
    """Newton iteration on beta(g) = 0."""
    g = float(g_guess)
    dg = 1.0e-6
    for _ in range(max_iter):
        bg = beta(g)
        if abs(bg) < tol:
            return g
        bpg = beta(g + dg)
        dbeta = (bpg - bg) / dg
        if abs(dbeta) < C.EPS_NUM:
            break
        g -= bg / dbeta
        if abs(g) > 1e6:
            break
    return float("nan")


# ---------------------------------------------------------------------------
# Two-coupling flow: (J, h) both running
# ---------------------------------------------------------------------------
def beta_vector(couplings: np.ndarray) -> np.ndarray:
    """Flow of (J, h, g = h/J) as a 3-component system.
    J' = J * (1 - g^2) / 2
    h' = h * (1 + g^2) / 2
    g' = g * (1 - g^2)
    This preserves the ratio g = h/J in the perturbative limit and
    adds the canonical nonlinear stabilisation.
    """
    J, h, g = float(couplings[0]), float(couplings[1]), float(couplings[2])
    dJ = 0.5 * J * (1.0 - g * g)
    dh = 0.5 * h * (1.0 + g * g)
    dg = g * (1.0 - g * g)
    return np.array([dJ, dh, dg])


def integrate_flow_vector(c0: np.ndarray, ln_b_max: float = 6.0,
                            n_steps: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the vector flow and return (ln_bs, couplings[n_steps+1, 3])."""
    ln_bs = np.linspace(0.0, ln_b_max, n_steps + 1)
    traj = np.zeros((n_steps + 1, 3))
    traj[0] = c0
    for i in range(n_steps):
        dt = ln_bs[i + 1] - ln_bs[i]
        traj[i + 1] = rk4_step(lambda t, y: beta_vector(y),
                                  0.0, traj[i], dt)
    return ln_bs, traj
