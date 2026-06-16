# -*- coding: utf-8 -*-
"""
ion_transport_kinetics.py
-------------------------
Coupled kinetic rate equations for the electron, hole, and phonon population
densities in a multi-band superconductor.  The structure mirrors the
ocular-surface ion-transport model but with *electrons*, *Cooper pairs*,
and *Einstein phonons* replacing Na+, K+, Cl-, and organic osmolytes.

Scientific origin of the fused algorithms
-----------------------------------------
* Ocular surface ion transport  (seed project 1106)
    -> nonlinear steady-state solver for coupled flux-balance equations
       with activity-coefficient corrections.
    -> adapted: the flux-balance equations are
          d n_e / dt = - Gamma_{pair} n_e^2 + Gamma_{break} n_p + S_e
          d n_p / dt = + Gamma_{pair} n_e^2 - Gamma_{break} n_p
          d n_q / dt = - Gamma_{em} n_q (1 + N_q) + Gamma_{abs} n_q N_q
       where n_e = electron density, n_p = Cooper-pair density, n_q =
       phonon occupation.  Activity coefficients (analogous to gamma=0.76
       for ions) account for many-body correlations.

Core physics / mathematics
--------------------------
* Pairing rate (BCS-like):
        Gamma_{pair}(T) = Gamma_0 * max(0, 1 - T/T_c0)
* Pair-breaking rate (thermal phonons):
        Gamma_{break}(T) = Gamma_b0 * N_B(2 Delta / T)
  where N_B(x) = 1/(e^x - 1) is the Bose function.
* Phonon emission / absorption:
        Gamma_{em} = g_ep^2 * (1 + N_B(omega_E / T))
        Gamma_{abs} = g_ep^2 * N_B(omega_E / T)
* Activity coefficient  alpha  (0 < alpha <= 1)  rescales the *effective*
  densities:  n_eff = alpha * n.

Stability / boundary notes
--------------------------
* Densities are clamped to non-negative values after every Newton step.
* The Jacobian of the residual is computed analytically (not by finite
  differences) to avoid numerical noise in the small-signal regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# 1.  Physical parameters  (cf. steady_state.py of seed project 1106)
# ---------------------------------------------------------------------------
@dataclass
class KineticParameters:
    """Physical parameters for the coupled electron-phonon kinetics."""
    alpha: float = 0.76             # activity coefficient (many-body correlation)
    T: float = 4.2                  # temperature (K)
    T_c0: float = 9.3               # mean-field Tc (K)  (Nb reference)
    Delta_0: float = 1.5e-3         # zero-T gap (eV)
    omega_E: float = 2.0e-2         # Einstein phonon energy (eV)
    Gamma_0: float = 1.0e10         # pairing prefactor (1/s)
    Gamma_b0: float = 5.0e9         # pair-breaking prefactor (1/s)
    g_ep: float = 2.0e9             # electron-phonon coupling (1/s)
    S_e: float = 1.0e12             # electron source (1/cm^3/s)
    n_e_init: float = 1.0e20        # initial electron density (1/cm^3)
    n_p_init: float = 1.0e18        # initial Cooper-pair density (1/cm^3)
    n_q_init: float = 1.0e17        # initial phonon density (1/cm^3)


def bose(x: float) -> float:
    """Bose function N_B(x) = 1/(e^x - 1), with safe handling of x near 0."""
    if x <= 0.0:
        return float("inf")
    if x > 500.0:
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)


# ---------------------------------------------------------------------------
# 2.  Residual of the steady-state kinetic equations
# ---------------------------------------------------------------------------
def kinetic_residual(
    x: np.ndarray,
    params: KineticParameters,
) -> np.ndarray:
    """Residual  F(x) = 0  of the steady-state kinetic equations.

    x = [n_e, n_p, n_q]  (effective densities after activity rescaling).
    """
    n_e, n_p, n_q = x
    p = params
    kB_eV = 8.617333e-5          # Boltzmann constant in eV/K
    T_eV = p.T * kB_eV
    Delta_T = p.Delta_0 * max(0.0, math.sqrt(max(0.0, 1.0 - (p.T / p.T_c0) ** 2)))
    Gamma_pair = p.Gamma_0 * max(0.0, 1.0 - p.T / p.T_c0)
    Gamma_break = p.Gamma_b0 * bose(2.0 * Delta_T / max(T_eV, 1e-300))
    N_B_E = bose(p.omega_E / max(T_eV, 1e-300))
    Gamma_em = p.g_ep ** 2 * (1.0 + N_B_E) / max(p.g_ep, 1.0)
    Gamma_abs = p.g_ep ** 2 * N_B_E / max(p.g_ep, 1.0)

    # Activity-rescaled densities
    ne_eff = p.alpha * n_e
    np_eff = p.alpha * n_p
    nq_eff = p.alpha * n_q

    # Steady-state residuals
    r1 = -Gamma_pair * ne_eff ** 2 + Gamma_break * np_eff + p.S_e
    r2 = Gamma_pair * ne_eff ** 2 - Gamma_break * np_eff
    r3 = -Gamma_em * nq_eff * (1.0 + N_B_E) + Gamma_abs * nq_eff * N_B_E

    return np.array([r1, r2, r3])


def kinetic_jacobian(
    x: np.ndarray,
    params: KineticParameters,
) -> np.ndarray:
    """Analytical Jacobian dF/dx of the kinetic residual."""
    n_e, n_p, n_q = x
    p = params
    kB_eV = 8.617333e-5
    T_eV = p.T * kB_eV
    Delta_T = p.Delta_0 * max(0.0, math.sqrt(max(0.0, 1.0 - (p.T / p.T_c0) ** 2)))
    Gamma_pair = p.Gamma_0 * max(0.0, 1.0 - p.T / p.T_c0)
    Gamma_break = p.Gamma_b0 * bose(2.0 * Delta_T / max(T_eV, 1e-300))
    N_B_E = bose(p.omega_E / max(T_eV, 1e-300))
    Gamma_em = p.g_ep ** 2 * (1.0 + N_B_E) / max(p.g_ep, 1.0)
    Gamma_abs = p.g_ep ** 2 * N_B_E / max(p.g_ep, 1.0)

    a = p.alpha
    J = np.zeros((3, 3))
    J[0, 0] = -2.0 * Gamma_pair * a ** 2 * n_e
    J[0, 1] = Gamma_break * a
    J[0, 2] = 0.0
    J[1, 0] = 2.0 * Gamma_pair * a ** 2 * n_e
    J[1, 1] = -Gamma_break * a
    J[1, 2] = 0.0
    J[2, 0] = 0.0
    J[2, 1] = 0.0
    J[2, 2] = -Gamma_em * (1.0 + N_B_E) * a + Gamma_abs * N_B_E * a
    return J


# ---------------------------------------------------------------------------
# 3.  Steady-state solver
# ---------------------------------------------------------------------------
@dataclass
class KineticSteadyState:
    """Solution of the coupled kinetic steady-state problem."""
    n_e: float
    n_p: float
    n_q: float
    residual_norm: float
    n_jac_eval: int
    pairing_rate: float
    break_rate: float


def solve_kinetic_steady_state(
    params: KineticParameters,
    max_iter: int = 500,
) -> KineticSteadyState:
    """Solve the kinetic steady-state equations using least-squares."""
    x0 = np.array([params.n_e_init, params.n_p_init, params.n_q_init])
    res = least_squares(
        kinetic_residual,
        x0,
        args=(params,),
        jac=lambda x, p: kinetic_jacobian(x, p),
        bounds=(0.0, np.inf),
        max_nfev=max_iter,
        ftol=1e-12,
        xtol=1e-12,
    )
    n_e, n_p, n_q = res.x
    kB_eV = 8.617333e-5
    T_eV = params.T * kB_eV
    Gamma_pair = params.Gamma_0 * max(0.0, 1.0 - params.T / params.T_c0)
    Gamma_break = params.Gamma_b0 * bose(
        2.0 * params.Delta_0 * max(0.0, math.sqrt(max(0.0, 1.0 - (params.T / params.T_c0) ** 2))) / max(T_eV, 1e-300)
    )
    return KineticSteadyState(
        n_e=float(n_e),
        n_p=float(n_p),
        n_q=float(n_q),
        residual_norm=float(res.cost),
        n_jac_eval=res.njev if hasattr(res, "njev") else 0,
        pairing_rate=Gamma_pair * (params.alpha * n_e) ** 2,
        break_rate=Gamma_break * params.alpha * n_p,
    )


# ---------------------------------------------------------------------------
# 4.  Temperature sweep of the steady state
# ---------------------------------------------------------------------------
@dataclass
class KineticSweep:
    """Temperature-sweep results of the kinetic steady state."""
    temperatures: np.ndarray
    n_e: np.ndarray
    n_p: np.ndarray
    n_q: np.ndarray
    pairing_rates: np.ndarray
    break_rates: np.ndarray


def sweep_kinetic_vs_temperature(
    T_range: Tuple[float, float] = (1.0, 15.0),
    n_T: int = 20,
    base_params: KineticParameters = None,
) -> KineticSweep:
    """Solve the kinetic steady state at each temperature in T_range."""
    if base_params is None:
        base_params = KineticParameters()
    temps = np.linspace(T_range[0], T_range[1], n_T)
    ne_arr, np_arr, nq_arr = [], [], []
    pair_arr, break_arr = [], []
    for T in temps:
        p = KineticParameters(
            alpha=base_params.alpha,
            T=T,
            T_c0=base_params.T_c0,
            Delta_0=base_params.Delta_0,
            omega_E=base_params.omega_E,
            Gamma_0=base_params.Gamma_0,
            Gamma_b0=base_params.Gamma_b0,
            g_ep=base_params.g_ep,
            S_e=base_params.S_e,
            n_e_init=base_params.n_e_init,
            n_p_init=base_params.n_p_init,
            n_q_init=base_params.n_q_init,
        )
        try:
            sol = solve_kinetic_steady_state(p)
            ne_arr.append(sol.n_e)
            np_arr.append(sol.n_p)
            nq_arr.append(sol.n_q)
            pair_arr.append(sol.pairing_rate)
            break_arr.append(sol.break_rate)
        except Exception:
            ne_arr.append(np.nan)
            np_arr.append(np.nan)
            nq_arr.append(np.nan)
            pair_arr.append(np.nan)
            break_arr.append(np.nan)
    return KineticSweep(
        temperatures=temps,
        n_e=np.array(ne_arr),
        n_p=np.array(np_arr),
        n_q=np.array(nq_arr),
        pairing_rates=np.array(pair_arr),
        break_rates=np.array(break_arr),
    )
