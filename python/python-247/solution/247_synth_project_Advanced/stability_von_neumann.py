"""
stability_von_neumann.py
========================
Von Neumann stability analysis of the high-order finite-difference
schemes used throughout the halo pipeline.

Scope
-----
* 5th-order upwind advection (used in convection_vortex_halo.py)
* Equilibrium-dispersive column model (multizone_accretion.py)
* Implicit trapezoidal ODE integrator (halo_ode_systems.py)

For each scheme we derive the amplification factor G(k) as a
function of the dimensionless wavenumber k dx and verify that
|G(k)| <= 1 for the chosen time step, or else compute the maximum
stable time step (CFL condition).
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


# ---------- 5th-order upwind advection --------------------------------------

def amplification_5th_upwind(cfl: float, n_theta: int = 256
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the amplification factor |G(theta)| for the 5th-order
    upwind advection scheme applied to u_t + a u_x = 0 with
    CFL = a dt / dx.  theta = k dx in [0, 2 pi]."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    # 5th-order upwind stencil coefficients (backward-biased):
    # f'_i = (2 f_{i-3} - 15 f_{i-2} + 60 f_{i-1}
    #         - 20 f_i - 30 f_{i+1} + 3 f_{i+2}) / 60
    coeffs = {
        -3:  2.0 / 60.0,
        -2: -15.0 / 60.0,
        -1:  60.0 / 60.0,
         0: -20.0 / 60.0,
        +1: -30.0 / 60.0,
        +2:  3.0 / 60.0,
    }
    # Symbol: sum_s c_s exp(i s theta)
    symbol = np.zeros_like(theta, dtype=complex)
    for s, c in coeffs.items():
        symbol += c * np.exp(1j * s * theta)
    # Euler time step: G = 1 - CFL * symbol
    G = 1.0 - cfl * symbol
    return theta, np.abs(G)


def max_stable_cfl_5th_upwind(n_theta: int = 256,
                              tol: float = 1e-8) -> float:
    """Find the maximum CFL for which |G(theta)| <= 1 + tol for all theta."""
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, Gmag = amplification_5th_upwind(mid, n_theta)
        if Gmag.max() <= 1.0 + tol:
            lo = mid
        else:
            hi = mid
    return lo


# ---------- Equilibrium-dispersive column -----------------------------------

def amplification_eq_dispersive(u: float, D: float, k_ads: float,
                                dx: float, dt: float,
                                n_theta: int = 256
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Amplification factor of the explicit upwind + central-difference
    discretisation of the equilibrium-dispersive equation, ignoring
    the adsorption coupling (k_ads = 0)."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    cfl_u = u * dt / dx
    cfl_d = D * dt / dx ** 2
    # upwind advection: G_adv = 1 - cfl_u (1 - exp(-i theta))
    # central diffusion:  G_diff = 1 - 2 cfl_d (1 - cos theta)
    G = (1.0 - cfl_u * (1.0 - np.exp(-1j * theta))
         - 2.0 * cfl_d * (1.0 - np.cos(theta)))
    return theta, np.abs(G)


# ---------- Implicit trapezoidal --------------------------------------------

def amplification_trapezoidal(lambda_real: float, dt: float
                              ) -> complex:
    """Amplification factor of the trapezoidal rule applied to
    y' = lambda y.   G = (1 + 0.5 h lambda) / (1 - 0.5 h lambda)."""
    z = lambda_real * dt
    return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)


# ---------- Convenience reports ---------------------------------------------

def stability_report() -> dict:
    cfl_max = max_stable_cfl_5th_upwind()
    _, G_05 = amplification_5th_upwind(0.5)
    _, G_10 = amplification_5th_upwind(1.0)
    theta, G_eq = amplification_eq_dispersive(u=1.0, D=0.05,
                                              k_ads=0.0, dx=0.1, dt=0.01)
    G_trap = amplification_trapezoidal(lambda_real=-1.0, dt=0.5)
    return dict(
        cfl_5th_upwind_max=cfl_max,
        max_G_at_cfl_half=float(G_05.max()),
        max_G_at_cfl_one=float(G_10.max()),
        max_G_eq_dispersive=float(G_eq.max()),
        amplification_trapezoidal_lambda_minus_one=abs(G_trap),
    )


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    return stability_report()
