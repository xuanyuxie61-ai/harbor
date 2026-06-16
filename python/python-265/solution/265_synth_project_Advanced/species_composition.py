# -*- coding: utf-8 -*-
"""
species_composition.py
----------------------
Sampling the elemental and charge composition of galactic cosmic
rays using a hypergeometric (urn) model.

The GCR source composition is drawn from a multinomial distribution
whose mean abundances follow the source abundances of
Binns et al. (2018) -- a mixture of FIP-biased coronal abundances
and pure r-process enrichment for the actinides.

The propagation process can be viewed as drawing  N_escape
particles without replacement from a finite source reservoir:
this is exactly the urn-without-replacement model.  The number of
particles of species  s  reaching the observer then follows a
hypergeometric distribution with parameters  (N_total, K_s, N_escape)
where  K_s  is the source abundance of species  s.

This module provides
    * source abundances (normalised);
    * hypergeometric sampling;
    * first-order leaky-box composition prediction including
      spallation losses:

            N_s = sum_{s'} N_{s', src} P_{s' -> s}
      with  P_{s' -> s} = (lambda_esc / (lambda_esc + lambda_sp_{s'}))
                           delta_{s s'} + ...

      where  lambda_esc = rho v tau_esc  and  lambda_sp  is the
      spallation mean free path.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple

import cosmic_ray_physics as crp


# =====================================================================
# Source abundances (normalised to Si = 10^6)
# =====================================================================
SOURCE_ABUNDANCES: Dict[str, Tuple[float, int, int]] = {
    # name : (abundance, Z, A)
    "H":  (3.11e7, 1, 1),
    "He": (2.14e6, 2, 4),
    "C":  (6.30e4, 6, 12),
    "N":  (1.63e4, 7, 14),
    "O":  (1.41e5, 8, 16),
    "Ne": (2.40e4, 10, 20),
    "Mg": (3.98e4, 12, 24),
    "Si": (1.00e6, 14, 28),
    "S":  (1.74e4, 16, 32),
    "Ar": (6.03e3, 18, 40),
    "Ca": (5.25e3, 20, 40),
    "Fe": (8.13e4, 26, 56),
    "Ni": (5.13e3, 28, 58),
}


def source_abundance_vector(names: List[str]) -> np.ndarray:
    """Return the normalised source abundance vector for the given
    list of species names.
    """
    K = np.array([SOURCE_ABUNDANCES[n][0] for n in names])
    return K / K.sum()


def source_ZA(names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z, A) arrays for the given species."""
    Z = np.array([SOURCE_ABUNDANCES[n][1] for n in names])
    A = np.array([SOURCE_ABUNDANCES[n][2] for n in names])
    return Z, A


# =====================================================================
# Urn sampling (without replacement)
# =====================================================================
def urn_sample(color_count: np.ndarray, draw_num: int,
               seed: int = 1) -> np.ndarray:
    """Draw  draw_num  items without replacement from an urn whose
    colour counts are given by  color_count.

    This is a multivariate hypergeometric draw implemented via the
    conditional-binomial algorithm.
    """
    rng = np.random.default_rng(seed)
    K = color_count.astype(np.int64)
    n = int(K.sum())
    if draw_num > n:
        raise ValueError("urn_sample: draw_num > total marble_num")
    result = np.zeros(K.size, dtype=np.int64)
    remaining = draw_num
    total_remaining = n
    for i in range(K.size - 1):
        if remaining <= 0:
            break
        # hypergeometric: draw from colour i given remaining & total
        p = K[i] / max(total_remaining, 1)
        # approximate with binomial (accurate for large counts)
        d = rng.binomial(remaining, p)
        result[i] = d
        remaining -= d
        total_remaining -= K[i]
    result[-1] = remaining
    return result


def two_color_pdf(k: int, K1: int, K2: int, n: int) -> float:
    """Exact hypergeometric pdf  P(X = k)  for  (K1, K2, n)."""
    if k < 0 or k > K1 or n - k < 0 or n - k > K2:
        return 0.0
    from math import comb
    num = comb(K1, k) * comb(K2, n - k)
    den = comb(K1 + K2, n)
    return num / den if den > 0 else 0.0


def ksub_random2(total: int, sample_size: int,
                 seed: int = 1) -> np.ndarray:
    """Select  sample_size  distinct integers from  1..total.

    (Reimplementation of the Fortran ksub_random2 routine.)
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=sample_size,
                              replace=False) + 1)


# =====================================================================
# Leaky-box composition
# =====================================================================
def spallation_cross_section(Z: int, A: int) -> float:
    """Total inelastic cross section on ISM hydrogen (Silberberg & Tsao
    1990 parametrisation, simplified):

        sigma_sp ~ 45 mb * A^{2/3}  (for A >= 4)
        sigma_sp ~ 30 mb            (for H)
        sigma_sp ~ 200 mb           (for He, empirical)
    """
    if A == 1:
        return 30.0e-3
    if A == 4:
        return 200.0e-3
    return 45.0e-27 * (A ** (2.0 / 3.0))  # [m^2]


def leaky_box_composition(names: List[str], escape_length_g_cm2: float,
                          ism_density: float = 2.4e-24 * 1.0e3) -> np.ndarray:
    """Return the leaky-box equilibrium abundances at the observer
    given the source abundances and the escape length.

    N_s = N_{s, src} * lambda_esc / (lambda_esc + lambda_sp_s)
    where  lambda_sp = m_p / sigma_sp  (in g/cm^2).
    """
    K = source_abundance_vector(names)
    Z, A = source_ZA(names)
    lambda_esc = escape_length_g_cm2  # g / cm^2
    N_out = np.zeros_like(K)
    for i, n in enumerate(names):
        sigma = spallation_cross_section(Z[i], A[i]) * 1.0e4  # m^2 -> cm^2
        lambda_sp = (crp.m_p * 1.0e3) / (sigma * 1.0)  # rough, g / cm^2
        # lambda_sp in g / cm^2 is  (A * m_p) / sigma  (with m_p in g)
        m_p_g = crp.m_p * 1.0e3
        lambda_sp = (A[i] * m_p_g) / max(sigma, 1.0e-30)
        N_out[i] = K[i] * lambda_esc / (lambda_esc + lambda_sp)
    return N_out / N_out.sum()


# =====================================================================
# Full Monte Carlo pipeline
# =====================================================================
def monte_carlo_composition(names: List[str],
                            N_total: int, N_escape: int,
                            escape_length_g_cm2: float,
                            seed: int = 3) -> Dict[str, float]:
    """Return a dictionary of species -> observed fraction.

    Pipeline:
        1. Build the source reservoir with  N_total  particles whose
           composition matches  SOURCE_ABUNDANCES.
        2. Draw  N_escape  particles without replacement (urn).
        3. Apply spallation survival (binomial) to each species.
        4. Normalise and return observed fractions.
    """
    rng = np.random.default_rng(seed)
    K = (source_abundance_vector(names) * N_total).astype(np.int64)
    # correct for rounding
    K[-1] += N_total - K.sum()
    drawn = urn_sample(K, N_escape, seed=seed)
    Z, A = source_ZA(names)
    survival = np.zeros_like(drawn, dtype=np.int64)
    for i, n in enumerate(names):
        sigma = spallation_cross_section(Z[i], A[i]) * 1.0e4
        m_p_g = crp.m_p * 1.0e3
        lambda_sp = (A[i] * m_p_g) / max(sigma, 1.0e-30)
        p_survive = escape_length_g_cm2 / (
            escape_length_g_cm2 + lambda_sp)
        survival[i] = rng.binomial(int(drawn[i]), p_survive)
    tot = survival.sum()
    return {n: float(survival[i]) / max(tot, 1)
            for i, n in enumerate(names)}


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    names = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]
    K = source_abundance_vector(names)
    print("[species_composition] Normalised source abundances:")
    for n, k in zip(names, K):
        print(f"  {n:>3s}: {k * 100:6.2f}%")
    N_total = 100_000
    N_escape = 20_000
    obs = monte_carlo_composition(names, N_total, N_escape,
                                  escape_length_g_cm2=10.0, seed=3)
    print(f"[species_composition] Observed fractions after escape + "
          f"spallation (N_escape={N_escape}):")
    for n in names:
        print(f"  {n:>3s}: {obs[n] * 100:6.2f}%")
    # hypergeometric pdf test
    p = two_color_pdf(3, K1=10, K2=20, n=5)
    print(f"[species_composition] Hypergeometric P(X=3 | K1=10, K2=20, "
          f"n=5) = {p:.5f}")


if __name__ == "__main__":
    _demo()
