"""
nuclear_physics.py - Liquid-drop mass, Saha equilibrium, partition functions.
Derived from the physics module of FootlooseCalvingMechanism (mass-balance /
stiff boundary-value patterns) and the Truncated-Normal statistical library.
"""
from __future__ import annotations
import math
from typing import Tuple
from physical_constants import (
    K_BOLTZMANN, M_NEUTRON, M_PROTON, M_U, MEV_ERG,
    SYMMETRY_COEFF, SURFACE_ENERGY_COEFF, COULOMB_COEFF, PAIRING_COEFF,
    TINY,
)

# ----------------------------------------------------------------------
# Liquid-drop binding energy  B(A, Z)  in MeV
#   B = a_v A - a_s A^{2/3} - a_c Z(Z-1) A^{-1/3}
#       - a_sym (A - 2Z)^2 / A + delta(A, Z)
# ----------------------------------------------------------------------
A_VOL = 15.67
A_SYM = SYMMETRY_COEFF
A_SURF = SURFACE_ENERGY_COEFF
A_COUL = COULOMB_COEFF


def pairing_term(A: int, Z: int) -> float:
    """delta  (MeV)  with sign depending on even/odd parity."""
    if A <= 0:
        return 0.0
    if A % 2 == 1:
        return 0.0
    if Z % 2 == 0:
        return +PAIRING_COEFF / math.sqrt(A)
    return -PAIRING_COEFF / math.sqrt(A)


def binding_energy(A: int, Z: int) -> float:
    """Liquid-drop binding energy B(A, Z) in MeV."""
    if A <= 0 or Z < 0 or Z > A:
        return 0.0
    vol = A_VOL * A
    surf = A_SURF * A ** (2.0 / 3.0)
    coul = A_COUL * Z * (Z - 1) * A ** (-1.0 / 3.0) if A > 0 else 0.0
    sym = A_SYM * (A - 2 * Z) ** 2 / A if A > 0 else 0.0
    delta = pairing_term(A, Z)
    return vol - surf - coul - sym + delta


def nuclear_mass(A: int, Z: int) -> float:
    """Nuclear mass in amu  m(A,Z) = Z m_H + (A-Z) m_n - B/c^2."""
    mH = 1.00782503223       # amu, atomic hydrogen
    mn = 1.00866491588       # amu, neutron
    B = binding_energy(A, Z)
    return Z * mH + (A - Z) * mn - B / 931.49410242


def separation_energy(A: int, Z: int) -> Tuple[float, float]:
    """Return (S_n, S_p) in MeV for nucleus (A, Z)."""
    B0 = binding_energy(A, Z)
    Bn = binding_energy(A - 1, Z) if A > 1 else 0.0
    Bp = binding_energy(A - 1, Z - 1) if A > 1 and Z > 0 else 0.0
    Sn = B0 - Bn
    Sp = B0 - Bp
    return Sn, Sp


# ----------------------------------------------------------------------
# Nuclear statistical equilibrium (NSE) / Saha abundance  (Cameron 1957)
#   Y(A,Z) = G(A,Z) * (A)^{3/2} * (Y_n)^A * (Y_p)^Z
#            * (2 pi hbar^2 / (m_u k T))^{3(A-1)/2}
#            * exp( B(A,Z) / (k T) )
# where G is the partition function.  All quantities in CGS internally.
# ----------------------------------------------------------------------

def nse_prefactor(T: float) -> float:
    """(2 pi hbar^2 / (m_u k T))^{3/2}  in  cm^{-3}"""
    from physical_constants import H_BAR
    num = 2.0 * math.pi * H_BAR * H_BAR
    den = M_U * K_BOLTZMANN * T
    if den < TINY:
        return 0.0
    return (num / den) ** 1.5


def saha_factor(A: int, Z: int, T: float, Yn: float, Yp: float,
                G: float = 1.0) -> float:
    """
    Return the Saha equilibrium abundance Y(A, Z) given free neutron and
    proton fractions Yn, Yp (number fraction relative to baryons) and a
    (lumped) internal partition function G.  Returns a non-negative float
    clamped to 0 on underflow.
    """
    if A <= 0 or T <= 0.0:
        return 0.0
    B_erg = binding_energy(A, Z) * MEV_ERG
    pref = nse_prefactor(T)
    A_factor = A ** 1.5
    power = 3.0 * (A - 1.0) / 2.0
    if pref == 0.0:
        return 0.0
    try:
        exp_arg = B_erg / (K_BOLTZMANN * T)
        # Guard against overflow: exp_arg can exceed 700 for heavy nuclei at low T
        if exp_arg > 500.0:
            exp_arg = 500.0
        Y = (G * A_factor
             * (max(Yn, TINY)) ** A
             * (max(Yp, TINY)) ** Z
             * pref ** (A - 1)
             * math.exp(exp_arg))
    except OverflowError:
        Y = 0.0
    if not math.isfinite(Y) or Y < 0.0:
        return 0.0
    return Y


def partition_function_simple(A: int, Z: int, T: float) -> float:
    """
    Crude single-resonance partition function:
        G(T) = g_gs + g_1 * exp(-E_1 / kT)
    with g_gs = 2J+1 guessed from parity of Z, N; E_1 = 1.0 MeV.
    """
    N = A - Z
    J_guess = 0.5 if ((Z % 2) + (N % 2)) == 1 else 0.0
    g_gs = max(1, int(2 * J_guess + 1))
    E1 = 1.0 * MEV_ERG
    g_1 = 3
    try:
        G = g_gs + g_1 * math.exp(-E1 / (K_BOLTZMANN * T))
    except OverflowError:
        G = float(g_gs)
    return max(1.0, G)


# ----------------------------------------------------------------------
# Truncated-normal sampling of rate uncertainties  (from 1360_truncated_normal)
# ----------------------------------------------------------------------

def _erf(x: float) -> float:
    return math.erf(x)


def std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def truncated_normal_mean(mu: float, sigma: float,
                          a: float, b: float) -> float:
    """
    Mean of a truncated normal on [a, b] with base N(mu, sigma^2).
    alpha = (a - mu) / sigma,  beta = (b - mu) / sigma
    E[X] = mu + sigma * (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha))
    """
    if sigma <= 0.0:
        return mu
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    den = std_normal_cdf(beta) - std_normal_cdf(alpha)
    if den < 1.0e-14:
        return mu
    return mu + sigma * (std_normal_pdf(alpha) - std_normal_pdf(beta)) / den


def sample_rate_uncertainty(rate_base: float,
                            fractional_sigma: float = 0.2,
                            seed: int = 1) -> float:
    """
    Return a rate multiplier drawn from a truncated-normal on [0.1, 10.0]
    with mean 1.0 and width fractional_sigma.  Deterministic pseudo-draw
    using a simple linear congruential sequence.
    """
    m = (1103515245 * seed + 12345) & 0x7FFFFFFF
    u = (m / 0x7FFFFFFF)
    z = (u - 0.5) * 6.0          # crude N(0,1)-like, clamped below
    z = max(-3.0, min(3.0, z))
    mu, sigma = 1.0, max(0.05, fractional_sigma)
    x = truncated_normal_mean(mu + sigma * z, sigma, 0.1, 10.0)
    return max(1.0e-3, rate_base * x)


__all__ = [
    "binding_energy", "nuclear_mass", "separation_energy",
    "saha_factor", "partition_function_simple",
    "truncated_normal_mean", "sample_rate_uncertainty",
]
