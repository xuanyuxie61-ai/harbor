"""
epidemic_distributions.py
Incomplete gamma function and epidemiological statistical distributions.

Adapted from:
  - 043_asa147: Algorithm AS 147 for the incomplete gamma integral

Role in synthesis:
  Provides cumulative distribution functions for generation intervals,
  infectious periods, and other gamma-distributed epidemiological quantities.
"""

import numpy as np
import math


def incomplete_gamma(x: float, p: float) -> tuple[float, int]:
    """
    Compute the regularized lower incomplete gamma function P(x, p) = γ(x, p) / Γ(p).

    Algorithm AS 147 (Chi Leung Lau, 1980).

    Parameters
    ----------
    x : float
        Upper limit of integration (x >= 0).
    p : float
        Shape parameter (p > 0).

    Returns
    -------
    value : float
        The regularized lower incomplete gamma value.
    ifault : int
        Error flag: 0=success, 1=invalid input, 2=underflow.
    """
    ifault = 0
    if x < 0.0:
        ifault = 1
        return 0.0, ifault
    if p <= 0.0:
        ifault = 1
        return 0.0, ifault
    if x == 0.0:
        return 0.0, ifault

    # Use gammaln for numerical stability
    try:
        arg = p * np.log(x) - x - math.lgamma(p)
    except (OverflowError, ValueError):
        ifault = 2
        return 0.0, ifault

    if arg < np.log(np.finfo(float).tiny):
        ifault = 2
        return 0.0, ifault

    f = np.exp(arg)
    if f == 0.0:
        ifault = 2
        return 0.0, ifault

    c = 1.0
    value = 1.0
    a = p

    while True:
        a = a + 1.0
        c = c * x / a
        value = value + c
        if c / value <= 1e-9:
            break
        if a > 1e6:
            break

    value = value * f
    return value, ifault


def gamma_pdf(t: float, shape: float, scale: float) -> float:
    """
    Probability density function of Gamma(shape, scale) distribution.
    f(t) = t^{shape-1} * exp(-t/scale) / (scale^shape * Γ(shape))
    """
    if t <= 0.0 or shape <= 0.0 or scale <= 0.0:
        return 0.0
    return (t ** (shape - 1.0)) * np.exp(-t / scale) / (scale ** shape * np.exp(math.lgamma(shape)))


def gamma_cdf(t: float, shape: float, scale: float) -> float:
    """
    Cumulative distribution function of Gamma(shape, scale) using incomplete gamma.
    F(t) = P(t/scale, shape)
    """
    if t <= 0.0:
        return 0.0
    try:
        from scipy.special import gammainc
        return float(gammainc(shape, t / scale))
    except ImportError:
        val, _ = incomplete_gamma(t / scale, shape)
        # AS147 computes unregularized; normalize by Gamma(shape)
        return float(val / np.exp(math.lgamma(shape))) if val > 0 else 0.0


def generation_interval_distribution(t: np.ndarray, mean: float = 5.2, std: float = 1.7) -> np.ndarray:
    """
    Generation interval distribution for infectious diseases.
    Parameters estimated from COVID-19 literature (mean=5.2 days, std=1.7 days).
    Shape = mean^2 / std^2, scale = std^2 / mean.
    """
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    return np.array([gamma_pdf(ti, shape, scale) for ti in t])


def cumulative_generation_interval(t: np.ndarray, mean: float = 5.2, std: float = 1.7) -> np.ndarray:
    """
    Cumulative generation interval probability.
    """
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    return np.array([gamma_cdf(ti, shape, scale) for ti in t])


def infectious_period_cdf(t: float, gamma_rate: float = 1.0 / 5.0) -> float:
    """
    CDF of infectious period assuming exponential distribution with rate gamma_rate.
    F(t) = 1 - exp(-gamma_rate * t)
    """
    if t < 0.0:
        return 0.0
    return 1.0 - np.exp(-gamma_rate * t)
