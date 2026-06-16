"""
eshelby_strain.py — Eshelby-inclusion elastic correction for finite-size effects
================================================================================

A point defect in a crystal acts as an elastic inclusion: the surrounding
lattice is strained, and the long-range 1/r² decay of the strain field
interacts with the periodic images of the supercell. The *Eshelby inclusion*
model gives an analytical correction that must be added to the raw
formation energy to remove this finite-size artefact.

For a circular (2-D) or ellipsoidal (3-D) inclusion of volume Ω_def with
misfit strain ε* in an isotropic medium of shear modulus G and Poisson
ratio ν, the elastic strain energy is
    E_elas = (1/2) Ω_def σ* : ε*
where σ* is the constrained stress. For a *circular* inclusion in 2-D,
    E_elas = 2 G (1 + ν) / (1 - ν) · Ω_def · ε*²
and for an *elliptical* inclusion with semi-axes a, b the formula involves
the complete elliptic integrals E(m) and K(m):
    E_elas^{ellip} = E_elas^{circ} · f(a/b, ν)
where
    f(a/b, ν) = [E(m) - (1 - m) K(m)] / (π/2)   with m = 1 - (b/a)²

This module implements:
  1. Circular and elliptical Eshelby strain energies.
  2. Complete elliptic integrals K(m) and E(m) — ported from
     `335_elliptic_integral/elliptic_ek.m` and `elliptic_em.m`.
  3. Finite-size scaling of E_elas with supercell size L.

The analytical Eshelby solution is used as a benchmark for the elastic
part of the formation energy; the electronic part is computed by the
DFT-like pipeline in `defect_formation_energy.py`.
"""

from __future__ import annotations
import math
from typing import Tuple


# -------------------------------------------------------------------------
# (1) Complete elliptic integrals K(m) and E(m)
#     (from 335_elliptic_integral/elliptic_ek.m, elliptic_em.m)
# -------------------------------------------------------------------------
def elliptic_K(m: float) -> float:
    """Complete elliptic integral of the first kind K(m), parameter m = k².

    Definition:
        K(m) = ∫_0^{π/2} dθ / sqrt(1 − m sin²θ)

    Computed via the arithmetic-geometric mean (AGM):
        K(m) = π / (2 · AGM(1, √(1 − m)))
    The AGM converges quadratically and gives 15-digit accuracy in ~5
    iterations.
    """
    if m < 0.0:
        m = 0.0
    if m >= 1.0:
        return float("inf")
    a = 1.0
    g = math.sqrt(1.0 - m)
    for _ in range(20):
        a_new = 0.5 * (a + g)
        g_new = math.sqrt(a * g)
        if abs(a_new - g_new) < 1e-15 * a_new:
            a, g = a_new, g_new
            break
        a, g = a_new, g_new
    return math.pi / (2.0 * a)


def elliptic_E(m: float) -> float:
    """Complete elliptic integral of the second kind E(m), parameter m = k².

    Definition:
        E(m) = ∫_0^{π/2} sqrt(1 − m sin²θ) dθ

    Computed via the AGM with running sum of c_n² terms:
        E(m) = K(m) · (1 − Σ_{n=0}^∞ 2^{n-1} c_n²)
    where c_n is the difference sequence in the AGM iteration.
    """
    if m < 0.0:
        m = 0.0
    if m >= 1.0:
        return 1.0
    a = 1.0
    g = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    s = c * c                # 2^{-1} * (2c)² = 2c² ... but let's track sum
    # We use the recurrence: sum = c² + 2(c₁² + 2c₂² + ...)
    # Starting sum = c0² = m
    pow2 = 1.0
    total = m
    for _ in range(20):
        a_new = 0.5 * (a + g)
        g_new = math.sqrt(a * g)
        c_new = 0.5 * (a - g)
        pow2 *= 2.0
        total += pow2 * c_new * c_new
        if abs(a_new - g_new) < 1e-15 * a_new:
            a, g = a_new, g_new
            break
        a, g = a_new, g_new
    K = math.pi / (2.0 * a)
    return K * (1.0 - 0.5 * total)


# -------------------------------------------------------------------------
# (2) Eshelby strain energy for circular and elliptical inclusions
# -------------------------------------------------------------------------
def eshelby_strain_energy(defect_volume: float, G: float, nu: float,
                          aspect: float = 1.0,
                          misfit_strain: float = 0.05) -> float:
    """Compute the Eshelby strain energy of a defect inclusion.

    Parameters
    ----------
    defect_volume : Ω_def, volume (area in 2-D) of the defect inclusion
    G : shear modulus (same units as desired output / volume)
    nu : Poisson ratio
    aspect : b / a for elliptical inclusion (= 1 for circular)
    misfit_strain : ε* (default 5% = 0.05, typical for vacancy in graphene)

    Returns
    -------
    E_elas : elastic strain energy in the same units as G × defect_volume.

    Formula (2-D circular inclusion):
        E_elas = 2 G (1 + ν) / (1 − ν) · Ω_def · ε*²
    For elliptical:
        E_elas^{ellip} = E_elas^{circ} · f(aspect, ν)
        f(aspect, ν) = [E(m) − (1 − m) K(m)] / (π/2)
        where m = 1 − aspect²

    References:
      * J.D. Eshelby, "The determination of the elastic field of an
        ellipsoidal inclusion", Proc. R. Soc. A 241, 376 (1957).
      * L.M. Brown & G.R. Woolhouse, "The loss of coherence of inclusions",
        Philos. Mag. 21, 53 (1970).
    """
    prefactor = 2.0 * G * (1.0 + nu) / (1.0 - nu)
    E_circ = prefactor * defect_volume * misfit_strain ** 2

    if abs(aspect - 1.0) < 1e-10:
        return E_circ

    # elliptical correction
    m_param = 1.0 - aspect ** 2
    m_param = max(0.0, min(m_param, 1.0 - 1e-10))
    Em = elliptic_E(m_param)
    Km = elliptic_K(m_param)
    f = (Em - (1.0 - m_param) * Km) / (0.5 * math.pi)
    # f is bounded: f(1) = 1, f(0) = 2/π
    return E_circ * f


# -------------------------------------------------------------------------
# (3) Finite-size scaling: how E_elas depends on supercell size L
# -------------------------------------------------------------------------
def finite_size_correction(L: float, E_elas_inf: float,
                           alpha: float = 1.0) -> float:
    """Finite-size correction to E_elas for a supercell of side L.

    The leading finite-size term is
        E_elas(L) = E_elas(∞) + α / L + O(1/L²)
    so the correction to be added to the raw formation energy is
        ΔE(L) = −α / L.
    """
    return E_elas_inf - alpha / L


# -------------------------------------------------------------------------
# (4) Validation: circular inclusion vs exact solution
# -------------------------------------------------------------------------
def validate_eshelby_circular() -> Tuple[float, float]:
    """Test: for a circular inclusion with Ω = 1.0, G = 1.0, ν = 0.3,
    ε* = 0.1, the analytical energy is
        E = 2 · 1 · 1.3 / 0.7 · 1 · 0.01 = 0.037142857...
    """
    E_num = eshelby_strain_energy(1.0, 1.0, 0.3, aspect=1.0, misfit_strain=0.1)
    E_exact = 2.0 * 1.0 * 1.3 / 0.7 * 0.01
    return E_num, E_exact


def validate_elliptic_K_E() -> Tuple[float, float, float, float]:
    """Test K(0) = π/2, E(0) = π/2, K(1/2) ≈ 1.8541, E(1/2) ≈ 1.3506."""
    K0 = elliptic_K(0.0)
    E0 = elliptic_E(0.0)
    Kh = elliptic_K(0.5)
    Eh = elliptic_E(0.5)
    return K0, E0, Kh, Eh
