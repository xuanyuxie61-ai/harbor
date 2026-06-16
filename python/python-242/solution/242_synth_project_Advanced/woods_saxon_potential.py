"""
woods_saxon_potential.py  --  Central and spin-orbit Woods-Saxon potential
==========================================================================
The Woods-Saxon (WS) form is the empirical parametrisation of the
mean-field single-particle potential felt by a valence nucleon in a
medium-to-heavy nucleus.

    V_WS(r)   = -V_0 / (1 + exp[(r - R)/a])            (central)
    V_SO(r)   = V_so * (1/r) * d/dr[ 1/(1 + exp[(r-R)/a]) ]
                * (l . s)                              (spin-orbit)
    V_C(r)    = (Z_c e^2)/(2 R_c) * (3 - (r/R_c)^2),  r < R_c   (Coulomb)
                (Z_c e^2)/r,                            r >= R_c

The expectation <l . s> = (j(j+1) - l(l+1) - s(s+1))/2 for j = l +/- 1/2.

References:
    Woods & Saxon, Nucl. Phys. (1954)
    Satchler, Direct Nuclear Reactions (1983)
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import (
    V0_CENTRAL, V_SO, R0_FM, DIFFUSIVENESS_A, V_COULOMB_R0,
    E_CHARGED_SQ, R_EPSILON,
)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """Overflow-safe exp; clip to [-700, 700] to avoid np.inf."""
    return np.exp(np.clip(x, -700.0, 700.0))


def woods_saxon_f(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """WS form factor f(r) = 1 / (1 + exp[(r - R)/a]).

    Returns values in [0, 1]; smooths from 1 (r << R) to 0 (r >> R).
    """
    r = np.asarray(r, dtype=np.float64)
    # Use logistic identity: 1 / (1 + e^x) = 0.5 * (1 - tanh(x/2))
    # This form is numerically stable for |x| up to very large values.
    return 0.5 * (1.0 - np.tanh((r - R) / (2.0 * a)))


def d_woods_saxon_dr(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """Derivative of the WS form factor with respect to r.

    d f / dr = - (1/a) * exp[(r-R)/a] / (1 + exp[(r-R)/a])^2
             = - 1 / (4 a cosh^2[(r - R)/(2a)])

    The second form is numerically preferable to avoid 0*inf.
    """
    r = np.asarray(r, dtype=np.float64)
    arg = (r - R) / (2.0 * a)
    # tanh has saturation but cosh^{-2} decays to 0 naturally
    ch = np.cosh(np.clip(arg, -250.0, 250.0))
    return -1.0 / (4.0 * a * ch * ch)


def d2_woods_saxon_dr2(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """Second derivative of the WS form factor.

    d^2 f / dr^2 = (1/(2a^2)) * sinh[(r-R)/(2a)] / cosh^3[(r-R)/(2a)]
    """
    r = np.asarray(r, dtype=np.float64)
    arg = (r - R) / (2.0 * a)
    arg_c = np.clip(arg, -250.0, 250.0)
    sh = np.sinh(arg_c)
    ch = np.cosh(arg_c)
    return sh / (2.0 * a * a * ch * ch * ch)


def central_potential(r: np.ndarray, A: int, V0: float = V0_CENTRAL) -> np.ndarray:
    """Central Woods-Saxon well V(r) = -V0 * f(r; R, a).

    Args:
        r  : radial grid in fm
        A  : mass number (determines R = R0 A^{1/3})
        V0 : depth in MeV
    """
    R = R0_FM * (A ** (1.0 / 3.0))
    return -V0 * woods_saxon_f(r, R, DIFFUSIVENESS_A)


def spin_orbit_potential(
    r: np.ndarray,
    A: int,
    l_q: int,
    j_q: float,
    Vso: float = V_SO,
) -> np.ndarray:
    r"""Spin-orbit term:
        V_so(r) = V_so * (1/r) * (d f / dr) * <l . s>

    where <l . s> = [j(j+1) - l(l+1) - 3/4] / 2.

    Note the derivative is NEGATIVE, so V_so > 0 lowers the j = l+1/2
    partner (spin-orbit splitting pulls j_> below j_<).
    """
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.where(r < R_EPSILON, R_EPSILON, r)
    s_half = 0.5
    l_dot_s = 0.5 * (j_q * (j_q + 1.0) - l_q * (l_q + 1.0) - s_half * (s_half + 1.0))
    df_dr = d_woods_saxon_dr(r_safe, R0_FM * (A ** (1.0 / 3.0)), DIFFUSIVENESS_A)
    return Vso * (1.0 / r_safe) * df_dr * l_dot_s


def coulomb_potential(r: np.ndarray, Z_core: int, A_core: int) -> np.ndarray:
    """Uniform-sphere Coulomb potential for a proton on a (Z_core, A_core) core.

    V_C(r) = Z_c e^2 / (2 R_c) * (3 - (r/R_c)^2),  r < R_c
    V_C(r) = Z_c e^2 / r,                           r >= R_c
    """
    r = np.asarray(r, dtype=np.float64)
    R_c = V_COULOMB_R0 * (A_core ** (1.0 / 3.0))
    V = np.where(
        r < R_c,
        E_CHARGED_SQ * Z_core / (2.0 * R_c) * (3.0 - (r / R_c) ** 2),
        E_CHARGED_SQ * Z_core / np.maximum(r, R_EPSILON),
    )
    return V


def total_single_particle_potential(
    r: np.ndarray,
    A: int,
    Z_core: int,
    l_q: int,
    j_q: float,
    is_proton: bool,
) -> np.ndarray:
    """Sum of central + spin-orbit + (optional) Coulomb.

    This is the radial Hamiltonian kernel entering the 1D Schrödinger eq.

        H u(r) = [ - hbar^2/(2 m_red) d^2/dr^2 + V(r) + l(l+1) hbar^2/(2 m_red r^2) ] u(r)

    where u(r) = r R(r) is the reduced radial wavefunction.
    """
    V = central_potential(r, A)
    V = V + spin_orbit_potential(r, A, l_q, j_q)
    if is_proton:
        V = V + coulomb_potential(r, Z_core, A - 1)
    return V


def centrifugal_barrier(r: np.ndarray, l_q: int, hbar2_over_2m: float) -> np.ndarray:
    """l(l+1) hbar^2 / (2 m r^2) barrier term acting on u(r)."""
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.where(r < R_EPSILON, R_EPSILON, r)
    return hbar2_over_2m * l_q * (l_q + 1.0) / (r_safe ** 2)


def surface_peaking_factor(r: np.ndarray, A: int) -> np.ndarray:
    """-df/dr measures the surface peaking of derivative couplings.

    Used when computing transition matrix elements that peak at the
    nuclear surface (e.g., collective E2 excitations).
    """
    R = R0_FM * (A ** (1.0 / 3.0))
    return -d_woods_saxon_dr(r, R, DIFFUSIVENESS_A)
