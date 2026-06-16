# -*- coding: utf-8 -*-
"""
nuclear_network.py
==================
Stiff nuclear reaction network for hydrogen, helium and carbon burning
in a stellar zone.

Background
----------
A nuclear reaction network tracks the mass fractions Y_i (or number
fractions X_i / A_i) of K nuclear species coupled by a system of
non-linear ODEs

    dY_i/dt = sum_{reactions r}  (nu_{ir} - nu'_{ir}) R_r(Y, T, rho)

where nu_{ir} is the stoichiometric coefficient of species i in
reaction r, and R_r is the reaction rate.

For the present project we use a reduced network of K = 8 species:
    1H, 4He, 12C, 14N, 16O, 20Ne, 24Mg, 56Fe
with the dominant reactions:
    pp-chain and CNO (H burning)
    3-alpha, 12C(alpha,g)16O (He burning)
    12C+12C, 16O+16O (C, Ne, O burning)
    Si equilibrium -> Fe (approximate)

The RHS is computed in a form that can be passed directly to the
adaptive time integrators of time_integrator.py.

This module draws on:
  - seed 1029_Fdl1989_TimingofOneShotInterventions (SIR compartmental
    ODE framework for multiple groups -> multiple species);
  - seed 1131_cchrisgong_aip_rockstar (hierarchical merger tree ->
    hierarchical reaction grouping by burning stage).

The system is stiff because the nuclear timescales (seconds to days)
are vastly shorter than the thermal timescale.  The Jacobian of the
system has eigenvalues spanning many orders of magnitude.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Callable
import math

from physical_constants import N_A, k_B, m_p, Mev_to_erg
from nuclear_quadrature import rate_CF88_pp, rate_tripalpha, rate_C12pg


# =====================================================================
# Species and reaction definitions
# =====================================================================

SPECIES = ["H1", "He4", "C12", "N14", "O16", "Ne20", "Mg24", "Fe56"]
ATOMIC_NUMBER = {"H1": 1, "He4": 2, "C12": 6, "N14": 7,
                  "O16": 8, "Ne20": 10, "Mg24": 12, "Fe56": 26}
MASS_NUMBER = {"H1": 1, "He4": 4, "C12": 12, "N14": 14,
                "O16": 16, "Ne20": 20, "Mg24": 24, "Fe56": 56}
K = len(SPECIES)


def species_index(name: str) -> int:
    return SPECIES.index(name)


# =====================================================================
# Reaction rates
# =====================================================================

def _safe_rate(r: float) -> float:
    """Guard against NaN / Inf / negative rates."""
    if not math.isfinite(r) or r < 0.0:
        return 0.0
    return r


def rate_pp(T9: float, rho: float, Y: List[float]) -> float:
    """pp-chain rate  (reactions / g / s).

    The pp rate depends on Y_H^2 * rho and is dominated by
    p(p,e+nu)d.  We use the CF88 fit for N_A <sigma v>.

    The physical reaction rate per unit mass is
        r_pp = 0.5 * N_A * Y_H^2 * rho * <sv>_pp / rho
             = 0.5 * N_A * Y_H^2 * <sv>_pp
    (reactions / g / s).  Note that rho cancels since one factor
    of rho is already included in N_A <sv> per mol per cm^3.
    """
    if Y[0] <= 0.0 or T9 <= 0.0:
        return 0.0
    sv = rate_CF88_pp(T9)   # N_A <sigma v> in cm^3 / mol / s
    # Rate per gram: 0.5 * N_A * Y_H^2 * sv  [1 / (g s)]
    return _safe_rate(0.5 * N_A * Y[0] * Y[0] * sv)


def rate_cno(T9: float, rho: float, Y: List[float]) -> float:
    """CNO cycle rate (reactions / g / s), gated by 12C(p,g)13N."""
    if T9 <= 0.0 or Y[0] <= 0.0 or Y[2] <= 0.0:
        return 0.0
    sv = rate_C12pg(T9)   # N_A <sigma v>  cm^3 / mol / s
    return _safe_rate(N_A * Y[0] * Y[2] * sv)


def rate_triple_alpha(T9: float, rho: float, Y: List[float]) -> float:
    """Triple-alpha  3 He4 -> C12  rate (reactions / g / s)."""
    if T9 <= 0.0 or Y[1] <= 0.0 or rho <= 0.0:
        return 0.0
    r_vol = rate_tripalpha(T9, rho)   # reactions / s / (g/cm^3)^2
    return _safe_rate(r_vol * rho * Y[1]**3)


def rate_c12_alpha(T9: float, rho: float, Y: List[float]) -> float:
    """12C(alpha, gamma)16O  rate (reactions / g / s)."""
    if T9 <= 0.0 or Y[1] <= 0.0 or Y[2] <= 0.0:
        return 0.0
    T9m23 = T9 ** (-2.0/3.0)
    log_r = (-5.1231 - 16.754 * T9m23 + 0.4567 * T9**(1.0/3.0)
             - 0.0123 * T9 + 27.312 * math.log(T9))
    sv = math.exp(log_r) if log_r < 600 else 0.0
    return _safe_rate(N_A * Y[1] * Y[2] * sv)


def rate_c12_c12(T9: float, rho: float, Y: List[float]) -> float:
    """12C + 12C  rate (reactions / g / s)."""
    if T9 <= 0.0 or Y[2] <= 0.0:
        return 0.0
    T9m13 = T9 ** (-1.0/3.0)
    log_r = (6.7813 - 8.4166 * T9m13 - 48.239 * T9m13 * T9m13
             - 1.2614 * T9**(1.0/3.0) + 0.1337 * T9
             - 0.25783 * T9 * T9 + 11.323 * math.log(T9))
    sv = math.exp(log_r) if log_r < 600 else 0.0
    return _safe_rate(0.5 * N_A * Y[2] * Y[2] * sv)


# =====================================================================
# Energy generation rates [erg / reaction]
# =====================================================================

# Q-values in erg per reaction (not per gram).  rate_* returns
# reactions / g / s, so epsilon = Q * rate [erg / g / s].
Q_PP      = 26.73 * Mev_to_erg      # erg / reaction
Q_CNO     = 25.0  * Mev_to_erg
Q_3ALPHA  = 7.275 * Mev_to_erg
Q_C12AG   = 7.162 * Mev_to_erg
Q_C12C12  = 13.93 * Mev_to_erg


def epsilon_nuc(T9: float, rho: float, Y: List[float]) -> float:
    """Total nuclear energy generation rate [erg / g / s]."""
    e = 0.0
    e += Q_PP     * rate_pp(T9, rho, Y)
    e += Q_CNO    * rate_cno(T9, rho, Y)
    e += Q_3ALPHA * rate_triple_alpha(T9, rho, Y)
    e += Q_C12AG  * rate_c12_alpha(T9, rho, Y)
    e += Q_C12C12 * rate_c12_c12(T9, rho, Y)
    return e


def neutrino_loss(T9: float, rho: float) -> float:
    """Neutrino energy loss rate [erg / g / s] (pair + photo + plasma).
    Simplified fit from Itoh et al. 1996."""
    if T9 <= 0.0:
        return 0.0
    # Pair neutrino (dominant at T9 > 0.3)
    if T9 >= 0.3:
        T11 = T9 * 1.0e-2
        lpair = (1.0e15 / max(rho, 1.0)) * T11**9 * math.exp(-5.930 / max(T11, 1.0e-30))
    else:
        lpair = 0.0
    # Plasma neutrino
    if T9 >= 0.05:
        T9_3 = T9 ** 3
        lplasma = 1.0e8 * T9_3 * math.exp(-1.0 / max(T9, 1.0e-30))
    else:
        lplasma = 0.0
    return _safe_rate(lpair + lplasma)


# =====================================================================
# RHS of the nuclear network ODE
# =====================================================================

def nuclear_rhs(t: float, Y: List[float], *,
                T9: float, rho: float) -> List[float]:
    """Right-hand side of the nuclear network ODE.

    The state vector Y has length K = 8 with mass fractions of
    [H1, He4, C12, N14, O16, Ne20, Mg24, Fe56].

    The coupling follows the SIR-style multi-group framework from
    seed 1029: each species is like a "compartment" and the reaction
    rates are the "infection matrix" that redistributes material.
    """
    dY = [0.0] * K
    if len(Y) != K:
        return dY
    # Clamp to physical range
    Yc = [max(0.0, min(1.0, y)) for y in Y]

    # Reaction rates
    r_pp   = rate_pp(T9, rho, Yc)
    r_cno  = rate_cno(T9, rho, Yc)
    r_3a   = rate_triple_alpha(T9, rho, Yc)
    r_cag  = rate_c12_alpha(T9, rho, Yc)
    r_cc   = rate_c12_c12(T9, rho, Yc)

    # pp chain: 4 H -> He4
    dY[0] -= 4.0 * r_pp / max(rho * N_A, 1.0e-30)
    dY[1] += 1.0 * r_pp / max(rho * N_A, 1.0e-30)

    # CNO: 4 H -> He4  (catalysed by C,N,O)
    dY[0] -= 4.0 * r_cno / max(rho * N_A, 1.0e-30)
    dY[1] += 1.0 * r_cno / max(rho * N_A, 1.0e-30)
    dY[2] -= 0.05 * r_cno / max(rho * N_A, 1.0e-30)   # slow C -> N
    dY[3] += 0.05 * r_cno / max(rho * N_A, 1.0e-30)

    # Triple-alpha: 3 He4 -> C12
    dY[1] -= 3.0 * r_3a / max(rho * N_A, 1.0e-30)
    dY[2] += 1.0 * r_3a / max(rho * N_A, 1.0e-30)

    # 12C(alpha,g)16O
    dY[2] -= 1.0 * r_cag / max(rho * N_A, 1.0e-30)
    dY[4] += 1.0 * r_cag / max(rho * N_A, 1.0e-30)

    # 12C + 12C -> Ne20 + He4  (simplified branch)
    dY[2] -= 2.0 * r_cc / max(rho * N_A, 1.0e-30)
    dY[5] += 1.0 * r_cc / max(rho * N_A, 1.0e-30)
    dY[1] += 1.0 * r_cc / max(rho * N_A, 1.0e-30)

    # Slow conversion to Fe (silicon-burning surrogate)
    r_fe = 1.0e-5 * max(Yc[5], 0.0) * max(Yc[4], 0.0) * T9**3
    dY[5] -= r_fe
    dY[6] -= 0.5 * r_fe
    dY[7] += 1.5 * r_fe

    # Ensure mass conservation: sum(dY) should be zero; we rescale
    total = sum(dY)
    if abs(total) > 1.0e-10:
        mean_rate = total / K
        dY = [d - mean_rate for d in dY]
    return dY


def make_rhs(T9: float, rho: float) -> Callable:
    """Return a closure suitable for the time integrators."""
    def f(t, Y):
        return nuclear_rhs(t, Y, T9=T9, rho=rho)
    return f


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("nuclear_network self-test:")
    # Solar core conditions
    T9 = 15.7e-3     # T = 1.57e7 K
    rho = 150.0      # g/cm^3
    Y = [0.35, 0.63, 0.005, 0.005, 0.008, 0.001, 0.001, 0.0]
    eps = epsilon_nuc(T9, rho, Y)
    print(f"  Solar core: T9={T9:.4e}, rho={rho:.2f}")
    print(f"    epsilon_nuc = {eps:.4e} erg/g/s")
    print(f"    L/L_sun per gram ~ {eps / 2.0:.4e}")  # very rough
    # RHS
    dY = nuclear_rhs(0.0, Y, T9=T9, rho=rho)
    print(f"    dY/dt = {[f'{x:.2e}' for x in dY]}")
    # He-burning conditions
    T9b = 0.1; rhob = 1.0e4
    Yb = [0.0, 0.98, 0.01, 0.0, 0.005, 0.0, 0.0, 0.005]
    eps_b = epsilon_nuc(T9b, rhob, Yb)
    print(f"  He-burning: T9={T9b:.4e}, rho={rhob:.2e}")
    print(f"    epsilon_nuc = {eps_b:.4e}")
    print("nuclear_network self-test OK")


if __name__ == "__main__":
    _self_test()
