"""
transport_pdf.py — Truncated log-normal statistics of turbulent transport
coefficients χ⊥(ψ), mapped from the 699_log_normal_truncated_ab project.

Scientific background
---------------------
Turbulent heat/particle transport in a tokamak is typically modelled as a
diffusive process χ⊥ with χ⊥ varying stochastically on the flux surface.
Experimental and gyrokinetic studies (e.g. Candy et al., Phys. Plasmas 2013)
show that the PDF of χ⊥ is well described by a *truncated log-normal*

    f(χ) = (1 / (χ σ √(2π))) exp(-(ln χ - μ)² / (2 σ²))
           / (Φ((ln b - μ)/σ) - Φ((ln a - μ)/σ))
           for χ ∈ [a, b]

with lower/upper bounds a, b set by physical considerations (neoclassical
floor / Bohm ceiling).

The mean, variance, CDF, inverse-CDF and sampling routines are direct
Python ports of the MATLAB scripts in 699_log_normal_truncated_ab.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Standard normal CDF / inverse CDF (Rational approximation, Abramowitz 26.2.23)
# ---------------------------------------------------------------------------

def normal_01_cdf(x: float) -> float:
    """Φ(x) = 0.5 · (1 + erf(x/√2))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_01_cdf_inv(p: float) -> float:
    """Inverse of Φ via rational approximation (Beasley-Springer-Moro)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    # Rational approximation for 0 < p < 1
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# ---------------------------------------------------------------------------
# Log-normal CDF / PDF
# ---------------------------------------------------------------------------

def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0.0:
        return 0.0
    return math.exp(-0.5 * ((math.log(x) - mu) / sigma) ** 2) / \
           (x * sigma * math.sqrt(2.0 * math.pi))


def log_normal_cdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0.0:
        return 0.0
    return normal_01_cdf((math.log(x) - mu) / sigma)


def log_normal_cdf_inv(p: float, mu: float, sigma: float) -> float:
    return math.exp(normal_01_cdf_inv(p) * sigma + mu)


def log_normal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + sigma * sigma / 2.0)


def log_normal_variance(mu: float, sigma: float) -> float:
    return (math.exp(sigma * sigma) - 1.0) * math.exp(2.0 * mu + sigma * sigma)


# ---------------------------------------------------------------------------
# Truncated log-normal on [a, b]
# ---------------------------------------------------------------------------

def tl_normalising(a: float, b: float, mu: float, sigma: float) -> float:
    """Z = Φ((ln b - μ)/σ) - Φ((ln a - μ)/σ)."""
    if a <= 0.0 or b <= a:
        return 1.0
    return normal_01_cdf((math.log(b) - mu) / sigma) - \
           normal_01_cdf((math.log(a) - mu) / sigma)


def tl_pdf(x: float, a: float, b: float, mu: float, sigma: float) -> float:
    if x < a or x > b:
        return 0.0
    return log_normal_pdf(x, mu, sigma) / tl_normalising(a, b, mu, sigma)


def tl_cdf(x: float, a: float, b: float, mu: float, sigma: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    Z = tl_normalising(a, b, mu, sigma)
    return (log_normal_cdf(x, mu, sigma) - log_normal_cdf(a, mu, sigma)) / Z


def tl_cdf_inv(p: float, a: float, b: float, mu: float, sigma: float) -> float:
    """Inverse CDF by linear interpolation in the un-truncated domain."""
    Za = log_normal_cdf(a, mu, sigma)
    Zb = log_normal_cdf(b, mu, sigma)
    target = Za + p * (Zb - Za)
    return log_normal_cdf_inv(target, mu, sigma)


def tl_mean(a: float, b: float, mu: float, sigma: float) -> float:
    """Mean of truncated log-normal via numerical quadrature (trapezoidal)."""
    n = 200
    xs = [a + i * (b - a) / (n - 1) for i in range(n)]
    ys = [x * tl_pdf(x, a, b, mu, sigma) for x in xs]
    dx = (b - a) / (n - 1)
    return sum(ys) * dx - 0.5 * dx * (ys[0] + ys[-1])


def tl_variance(a: float, b: float, mu: float, sigma: float) -> float:
    m = tl_mean(a, b, mu, sigma)
    n = 200
    xs = [a + i * (b - a) / (n - 1) for i in range(n)]
    ys = [(x - m) ** 2 * tl_pdf(x, a, b, mu, sigma) for x in xs]
    dx = (b - a) / (n - 1)
    return sum(ys) * dx - 0.5 * dx * (ys[0] + ys[-1])


def tl_sample(rng, a: float, b: float, mu: float, sigma: float) -> float:
    """Inverse-CDF sampling with the user-provided RNG."""
    u = rng.random()
    return tl_cdf_inv(u, a, b, mu, sigma)


# ---------------------------------------------------------------------------
# Application: sample a profile χ⊥(ψ_n) on [0,1]
# ---------------------------------------------------------------------------

def sample_chi_profile(n_psi: int = 32, seed: int = 42,
                       chi_neo: float = 0.05, chi_bohm: float = 5.0,
                       mu: float = 0.0, sigma: float = 0.7) -> list[float]:
    """Sample χ⊥(ψ_n) ∈ [χ_neo, χ_bohm] for n_psi equally-spaced ψ_n values
    from a truncated log-normal distribution."""
    import random
    rng = random.Random(seed)
    chi = []
    for _ in range(n_psi):
        c = tl_sample(rng, chi_neo, chi_bohm, mu, sigma)
        chi.append(c)
    return chi
