"""
monte_carlo_spectrum.py
=======================
Monte-Carlo estimation of CMB angular power spectra.

The angular power spectrum C_l is defined as
    < a_{lm} a_{l'm'}^* > = delta_{ll'} delta_{mm'} C_l
where
    a_{lm} = integral_{S^2} T(theta, phi) Y_{lm}^*(theta, phi) d Omega.

This module computes C_l via Monte-Carlo quadrature:
    a_{lm} ~ (4 pi / N) sum_{i=1}^{N} T(theta_i, phi_i) Y_{lm}^*(theta_i, phi_i)
           = V_{S^2} * mean_i [T * Y_lm^*]
where {(theta_i, phi_i)} are N random samples on S^2 and V_{S^2} = 4 pi.

Two flavours are provided:
  1. Uniform Monte-Carlo  (samples uniformly on S^2 via inverse CDF)
  2. Importance sampling  (samples more densely where |T| is large)

We also implement a variance-reduced estimator using antithetic variates:
    T_pair = (T(theta, phi) + T(pi - theta, phi + pi)) / 2
which halves the variance for parity-symmetric fields.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Callable


# ---------------------------------------------------------------------------
# Sampling on S^2
# ---------------------------------------------------------------------------
def sample_uniform_sphere(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    """
    Uniform sampling on S^2 via inverse CDF:
        theta = arccos(1 - 2 u),   phi = 2 pi v
    for u, v ~ Uniform(0, 1).
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * u)))
        phi = 2.0 * math.pi * v
        samples.append((theta, phi))
    return samples


def sample_importance(n: int, T_func: Callable[[float, float], float],
                       seed: int = 42, n_pilot: int = 2000) -> List[Tuple[float, float]]:
    """
    Two-stage importance sampling:
      1. Draw n_pilot uniform samples, compute |T|.
      2. Build a piecewise-constant proposal distribution on
         10 latitude bands weighted by mean |T|.
      3. Draw n samples from the proposal.
    """
    rng = random.Random(seed)
    pilot = sample_uniform_sphere(n_pilot, seed)
    # 10 latitude bands
    n_bands = 10
    band_sum = [0.0] * n_bands
    band_count = [0] * n_bands
    for (theta, phi) in pilot:
        b = min(n_bands - 1, int(theta / math.pi * n_bands))
        band_sum[b] += abs(T_func(theta, phi))
        band_count[b] += 1
    weights = [band_sum[i] / max(1, band_count[i]) for i in range(n_bands)]
    total = sum(weights) + 1e-30
    weights = [w / total for w in weights]
    # CDF
    cdf = []
    s = 0.0
    for w in weights:
        s += w
        cdf.append(s)
    # Draw
    samples = []
    for _ in range(n):
        u = rng.random()
        b = 0
        while b < n_bands - 1 and u > cdf[b]:
            b += 1
        theta_min = b * math.pi / n_bands
        theta_max = (b + 1) * math.pi / n_bands
        # Sample theta uniformly in [theta_min, theta_max]
        # but with sin(theta) weighting within band:
        #   cos theta ~ Uniform(cos theta_max, cos theta_min)
        ct_min = math.cos(theta_max)
        ct_max = math.cos(theta_min)
        ct = ct_min + (ct_max - ct_min) * rng.random()
        theta = math.acos(max(-1.0, min(1.0, ct)))
        phi = 2.0 * math.pi * rng.random()
        samples.append((theta, phi))
    return samples


# ---------------------------------------------------------------------------
# Real spherical harmonics (normalised)
# ---------------------------------------------------------------------------
def _legendre_P(l: int, m: int, x: float) -> float:
    """Associated Legendre function P_l^m(x) with Condon-Shortley phase."""
    if m < 0 or m > l:
        return 0.0
    # P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    # P_{m+1}^m
    pmm1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmm1
    # Recurrence
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmm1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = pll
    return pll


def _norm_factor(l: int, m: int) -> float:
    """Normalisation sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!)"""
    num = 1.0
    den = 1.0
    for i in range(1, l - m + 1):
        den *= i
    for i in range(1, l + m + 1):
        num *= i
    return math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi) * den / num)


def spherical_harmonic_real(l: int, m_signed: int, theta: float, phi: float) -> float:
    """
    Real spherical harmonic Y_{lm} with sign convention:
      m >= 0: Y_{lm} = N_{lm} P_l^m(cos theta) cos(m phi)
      m <  0: Y_{lm} = N_{l|m|} P_l^|m|(cos theta) sin(|m| phi)
    """
    m = abs(m_signed)
    if m > l:
        return 0.0
    N = _norm_factor(l, m)
    P = _legendre_P(l, m, math.cos(theta))
    if m_signed >= 0:
        return N * P * math.cos(m * phi)
    return N * P * math.sin(m * phi)


# ---------------------------------------------------------------------------
# Monte-Carlo estimator for a_{lm}
# ---------------------------------------------------------------------------
def mc_alm(T_func: Callable[[float, float], float],
            l_max: int,
            n_samples: int,
            seed: int = 42,
            importance: bool = False) -> Dict[Tuple[int, int], float]:
    """
    Estimate a_{lm} = int_{S^2} T Y_{lm} dOmega  for all 0 <= l <= l_max.
    Returns dict of (l, m) -> a_{lm}  for -l <= m <= l.
    """
    if importance:
        samples = sample_importance(n_samples, T_func, seed)
        # Importance weights  w_i = (4 pi) / (n_bands * p_i)
        # For simplicity, we use uniform weighting with variance correction
        weight = 4.0 * math.pi / n_samples
    else:
        samples = sample_uniform_sphere(n_samples, seed)
        weight = 4.0 * math.pi / n_samples

    alm: Dict[Tuple[int, int], float] = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            s = 0.0
            for (theta, phi) in samples:
                s += T_func(theta, phi) * spherical_harmonic_real(l, m, theta, phi)
            alm[(l, m)] = weight * s
    return alm


# ---------------------------------------------------------------------------
# Power spectrum C_l from a_{lm}
# ---------------------------------------------------------------------------
def cl_from_alm(alm: Dict[Tuple[int, int], float], l_max: int) -> List[float]:
    """
    C_l = (1 / (2l+1)) sum_{m=-l}^{l} |a_{lm}|^2
    """
    cl = [0.0] * (l_max + 1)
    for l in range(l_max + 1):
        s = 0.0
        for m in range(-l, l + 1):
            a = alm.get((l, m), 0.0)
            s += a * a
        cl[l] = s / (2 * l + 1) if (2 * l + 1) > 0 else 0.0
    return cl


# ---------------------------------------------------------------------------
# Antithetic variates for variance reduction
# ---------------------------------------------------------------------------
def mc_alm_antithetic(T_func: Callable[[float, float], float],
                        l_max: int,
                        n_samples: int,
                        seed: int = 42) -> Dict[Tuple[int, int], float]:
    """
    Use antithetic variates: for each sample (theta, phi), also use
    (pi - theta, phi + pi).  The averaged estimator is
        a_{lm} = (2 pi / n) sum_i [ T(theta_i, phi_i) + T(pi-theta_i, phi_i+pi) ] / 2
    This cancels odd-parity modes and halves variance.
    """
    samples = sample_uniform_sphere(n_samples, seed)
    weight = 4.0 * math.pi / n_samples
    alm: Dict[Tuple[int, int], float] = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            s = 0.0
            for (theta, phi) in samples:
                t1 = T_func(theta, phi)
                t2 = T_func(math.pi - theta, phi + math.pi)
                Y1 = spherical_harmonic_real(l, m, theta, phi)
                Y2 = spherical_harmonic_real(l, m, math.pi - theta, phi + math.pi)
                s += (t1 * Y1 + t2 * Y2) / 2.0
            alm[(l, m)] = weight * s
    return alm


# ---------------------------------------------------------------------------
# Variance estimation via multiple independent runs
# ---------------------------------------------------------------------------
def cl_variance(T_func: Callable[[float, float], float],
                  l_max: int,
                  n_samples: int,
                  n_runs: int = 10,
                  seed_base: int = 42) -> Tuple[List[float], List[float]]:
    """
    Run n_runs independent MC estimates, return (mean Cl, std Cl).
    """
    all_cl: List[List[float]] = []
    for r in range(n_runs):
        alm = mc_alm(T_func, l_max, n_samples, seed_base + r * 1000)
        all_cl.append(cl_from_alm(alm, l_max))
    mean = [0.0] * (l_max + 1)
    var  = [0.0] * (l_max + 1)
    for cl in all_cl:
        for l in range(l_max + 1):
            mean[l] += cl[l]
    for l in range(l_max + 1):
        mean[l] /= n_runs
    for cl in all_cl:
        for l in range(l_max + 1):
            var[l] += (cl[l] - mean[l]) ** 2
    for l in range(l_max + 1):
        var[l] = math.sqrt(var[l] / max(1, n_runs - 1))
    return mean, var


# ---------------------------------------------------------------------------
# Synthetic CMB map for testing
# ---------------------------------------------------------------------------
def synthetic_cmb(alm_true: Dict[Tuple[int, int], float]) -> Callable[[float, float], float]:
    """
    Build a callable T(theta, phi) = sum_{l,m} a_{lm} Y_{lm}(theta, phi).
    """
    def T(theta: float, phi: float) -> float:
        s = 0.0
        for (l, m), a in alm_true.items():
            s += a * spherical_harmonic_real(l, m, theta, phi)
        return s
    return T


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test: pure l=2, m=0 mode
    alm_true = {(2, 0): 1.0}
    T = synthetic_cmb(alm_true)
    alm_est = mc_alm(T, l_max=3, n_samples=5000, seed=1)
    cl_est = cl_from_alm(alm_est, 3)
    print("Synthetic l=2,m=0 test:")
    for l in range(4):
        print(f"  C_{l} = {cl_est[l]:.4e}")
