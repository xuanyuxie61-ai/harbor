"""
energy_spectra.py
=================
Multigroup fusion-neutron source spectra constructed from
**Dirichlet-distributed** mixture weights.

The D-T reaction produces 14.06 MeV neutrons, but in a breeding blanket the
*effective* source seen by each energy group depends on:
  (i)  the finite energy resolution of the multigroup discretisation;
  (ii) the plasma ion-temperature broadening of the D-T peak;
  (iii) the presence of D-D, T-T and D-3He satellite reactions.

We model the group-wise source fractions  p = (p_1, ..., p_G)  as a draw
from a Dirichlet distribution  Dir(alpha)  whose concentration parameters
alpha_g are determined by folding the analytical D-T spectrum against the
group boundaries.  This follows the treatment of:

  * `dirichlet_sample`      -> sample p ~ Dir(alpha)
  * `dirichlet_mean`        -> E[p] = alpha / alpha_0
  * `dirichlet_variance`    -> Var[p_g] = a_g (a_0 - a_g) / (a_0^2 (a_0 + 1))
  * `gamma_sample`          -> used to sample Gamma(alpha_g, 1) variates
  * `gammad` / `gammainc`   -> incomplete-gamma CDF
  * `exponential_cdf_inv`   -> inverse-CDF sampling of the slowing-down tail

Adapted from seed project 053_asa266 (ASA reference routines, Burkardt).
"""

from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# Gamma / log-gamma / digamma / trigamma (Burkardt's ASA routines)
# ---------------------------------------------------------------------------
def lngamma(x: float) -> float:
    """Return ln Gamma(x) for x > 0 (Lanczos 7-term approximation)."""
    if x <= 0.0:
        raise ValueError("lngamma requires x > 0")
    g = 7
    coef = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - lngamma(1.0 - x)
    x -= 1.0
    a = coef[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += coef[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def gammad(a: float, b: float, x: float) -> float:
    """Regularised incomplete gamma  P(a, x/b)  (CDF of Gamma(a,b)).

    Uses the series expansion for x < a+1 and the continued fraction
    otherwise.  This is Burkardt's `gammainc` / `gammd` combination.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("gammad requires a, b > 0")
    if x <= 0.0:
        return 0.0
    y = x / b
    if y < a + 1.0:
        # series
        term = 1.0 / a
        total = term
        for n in range(1, 500):
            term *= y / (a + n)
            total += term
            if abs(term) < 1.0e-13 * abs(total):
                break
        return total * math.exp(-y + a * math.log(y) - lngamma(a))
    # continued fraction (Legendre)
    f = 1.0
    c = 1.0
    d = 1.0 / (y + 1.0 - a)
    f = d
    for i in range(1, 500):
        ai = i * (a - i)
        bi = y + 2 * i + 1 - a
        d = 1.0 / (bi + ai * d)
        c = bi + ai / c
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1.0e-13:
            break
    cf = f * math.exp(-y + a * math.log(y) - lngamma(a))
    return 1.0 - cf


def gamma_sample(a: float, b: float, seed_state: List[int]) -> float:
    """Sample X ~ Gamma(shape=a, scale=b) using Marsaglia-Tsang (a >= 1)
    or the Ahrens-Dieter transform (a < 1).

    The state list carries the LCG seed so the caller can reproduce the
    stream deterministically.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("gamma_sample requires a, b > 0")
    if a < 1.0:
        u = _lcg_uniform(seed_state)
        return gamma_sample(a + 1.0, b, seed_state) * u ** (1.0 / a)
    d = a - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = _normal_standard(seed_state)
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = _lcg_uniform(seed_state)
        if u < 1.0 - 0.0331 * x ** 4:
            return d * v * b
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v * b


def _lcg_uniform(state: List[int]) -> float:
    """Linear congruential generator -> U(0,1)."""
    s = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
    state[0] = s
    return (s + 0.5) / 0x100000000


def _normal_standard(state: List[int]) -> float:
    """Standard-normal sample via Box-Muller."""
    u1 = max(_lcg_uniform(state), 1.0e-300)
    u2 = _lcg_uniform(state)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Dirichlet distribution
# ---------------------------------------------------------------------------
def dirichlet_mean(alpha: Sequence[float]) -> List[float]:
    """Mean of Dir(alpha):  m_g = alpha_g / sum(alpha)."""
    s = sum(alpha)
    if s <= 0.0:
        raise ValueError("alpha must have positive sum")
    return [a / s for a in alpha]


def dirichlet_variance(alpha: Sequence[float]) -> List[float]:
    """Diagonal of the Dirichlet covariance matrix."""
    a0 = sum(alpha)
    if a0 <= 0.0:
        raise ValueError("alpha must have positive sum")
    return [a * (a0 - a) / (a0 * a0 * (a0 + 1.0)) for a in alpha]


def dirichlet_sample(alpha: Sequence[float],
                      seed: int = 271828) -> List[float]:
    """Sample a single Dirichlet vector via the gamma construction."""
    state = [seed & 0xFFFFFFFF]
    g = [gamma_sample(a, 1.0, state) for a in alpha]
    s = sum(g)
    if s <= 0.0:
        # degenerate; return the mean
        return dirichlet_mean(alpha)
    return [x / s for x in g]


# ---------------------------------------------------------------------------
# Fusion-neutron source spectrum
# ---------------------------------------------------------------------------
def dt_spectrum_bimodal(E_mev: float, T_i_keV: float = 20.0) -> float:
    """Analytical D-T neutron spectrum (Gaussian broadening).

    The Brysk formula gives the FWHM of the 14.06 MeV peak as

        Delta_E = 177 * sqrt(T_i [keV])   [keV]

    We use sigma = FWHM / (2 sqrt(2 ln 2)) and a normalised Gaussian.
    """
    E0 = pc.E_NEUTRON_DT
    fwhm_keV = 177.0 * math.sqrt(max(T_i_keV, 0.1))
    sigma_mev = (fwhm_keV * 1.0e-3) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if sigma_mev <= 0.0:
        sigma_mev = 1.0e-3
    return (
        math.exp(-0.5 * ((E_mev - E0) / sigma_mev) ** 2)
        / (sigma_mev * math.sqrt(2.0 * math.pi))
    )


def build_dt_group_source(
    T_i_keV: float = 20.0,
    n_quad: int = 8,
) -> List[float]:
    """Return the unnormalised D-T source intensity in each group.

    The Gaussian is integrated over each group by Gauss-Legendre quadrature
    of order `n_quad`.
    """
    nodes, weights = gauss_legendre(n_quad)
    q: List[float] = []
    for g in range(pc.N_GROUPS):
        E_hi = pc.GROUP_BOUNDS_MEV[g]
        E_lo = pc.GROUP_BOUNDS_MEV[g + 1]
        half = 0.5 * (E_hi - E_lo)
        mid = 0.5 * (E_hi + E_lo)
        total = 0.0
        for xi, w in zip(nodes, weights):
            E = mid + half * xi
            total += w * dt_spectrum_bimodal(E, T_i_keV)
        q.append(total * half)
    return q


def dirichlet_source_weights(
    T_i_keV: float = 20.0,
    concentration: float = 50.0,
) -> List[float]:
    """Return Dirichlet concentration vector for the group weights.

    alpha_g = concentration * (integral of D-T spectrum in group g).
    A small floor (1e-3) is added so every component is strictly positive,
    which is required for the Dirichlet sampler to work.
    """
    q = build_dt_group_source(T_i_keV)
    s = sum(q)
    if s <= 0.0:
        return [concentration / pc.N_GROUPS] * pc.N_GROUPS
    # normalise and add a small floor
    floor = 1.0e-3
    alpha = [max(concentration * x / s, floor) for x in q]
    return alpha


def sample_group_fraction(
    T_i_keV: float = 20.0,
    concentration: float = 50.0,
    seed: int = 314159,
) -> List[float]:
    """Draw a single source-fraction vector p ~ Dir(alpha(T_i))."""
    alpha = dirichlet_source_weights(T_i_keV, concentration)
    return dirichlet_sample(alpha, seed=seed)


# ---------------------------------------------------------------------------
# Slowing-down tail (inverse-CDF sampler)
# ---------------------------------------------------------------------------
def exponential_cdf_inverse(u: float, lam: float) -> float:
    """Inverse of  F(x) = 1 - exp(-lambda x).  Burkardt's routine."""
    if lam <= 0.0:
        raise ValueError("lambda must be positive")
    u = max(0.0, min(1.0 - 1.0e-15, u))
    return -math.log(1.0 - u) / lam


def sample_slowing_down_tail(group_lo_mev: float,
                              xi_mev: float = 0.5,
                              u: float = 0.5) -> float:
    """Sample a neutron energy in the 1/E slowing-down tail below E_min."""
    return group_lo_mev * math.exp(-xi_mev * exponential_cdf_inverse(u, 1.0))


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature (from 395_fem1d_pack / legendre_com)
# ---------------------------------------------------------------------------
def gauss_legendre(n: int) -> Tuple[List[float], List[float]]:
    """Return nodes and weights for Gauss-Legendre on [-1,1] of order n."""
    if n < 1:
        raise ValueError("order n must be >= 1")
    nodes: List[float] = []
    weights: List[float] = []
    m = (n + 1) // 2
    for i in range(m):
        # initial guess
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(50):
            p0 = 1.0
            p1 = z
            for k in range(1, n):
                p2 = ((2 * k + 1) * z * p1 - k * p0) / (k + 1)
                p0 = p1
                p1 = p2
            # derivative
            pp = n * (z * p1 - p0) / (z * z - 1.0) if abs(z * z - 1.0) > 1e-30 else 0.0
            if pp == 0.0:
                break
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < 1.0e-15:
                break
        nodes.append(-z)
        nodes.append(z)
        weights.append(2.0 / ((1.0 - z * z) * pp * pp))
        weights.append(2.0 / ((1.0 - z * z) * pp * pp))
    # sort and dedupe
    paired = sorted(zip(nodes, weights), key=lambda p: p[0])
    nodes_out: List[float] = []
    weights_out: List[float] = []
    for n_, w_ in paired:
        if nodes_out and abs(nodes_out[-1] - n_) < 1.0e-14:
            weights_out[-1] += w_
        else:
            nodes_out.append(n_)
            weights_out.append(w_)
    return nodes_out, weights_out
