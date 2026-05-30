
import numpy as np
import math


def incomplete_gamma(x: float, p: float) -> tuple[float, int]:
    ifault = 0
    if x < 0.0:
        ifault = 1
        return 0.0, ifault
    if p <= 0.0:
        ifault = 1
        return 0.0, ifault
    if x == 0.0:
        return 0.0, ifault


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
    if t <= 0.0 or shape <= 0.0 or scale <= 0.0:
        return 0.0
    return (t ** (shape - 1.0)) * np.exp(-t / scale) / (scale ** shape * np.exp(math.lgamma(shape)))


def gamma_cdf(t: float, shape: float, scale: float) -> float:
    if t <= 0.0:
        return 0.0
    try:
        from scipy.special import gammainc
        return float(gammainc(shape, t / scale))
    except ImportError:
        val, _ = incomplete_gamma(t / scale, shape)

        return float(val / np.exp(math.lgamma(shape))) if val > 0 else 0.0


def generation_interval_distribution(t: np.ndarray, mean: float = 5.2, std: float = 1.7) -> np.ndarray:
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    return np.array([gamma_pdf(ti, shape, scale) for ti in t])


def cumulative_generation_interval(t: np.ndarray, mean: float = 5.2, std: float = 1.7) -> np.ndarray:
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    return np.array([gamma_cdf(ti, shape, scale) for ti in t])


def infectious_period_cdf(t: float, gamma_rate: float = 1.0 / 5.0) -> float:
    if t < 0.0:
        return 0.0
    return 1.0 - np.exp(-gamma_rate * t)
