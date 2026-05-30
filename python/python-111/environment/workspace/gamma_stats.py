
import numpy as np
from scipy.special import gammaln
from typing import Tuple


def gammds(x: float, p: float, eps: float = 1e-9) -> Tuple[float, int]:
    if x <= 0.0 or p <= 0.0:
        return 0.0, 1
    

    arg = p * np.log(x) - gammaln(p + 1.0) - x
    if arg < -100.0:

        return 0.0, 2
    
    e = np.exp(arg)
    if e < 1e-37:
        return 0.0, 2
    
    c = 1.0
    series_sum = 1.0
    a = p
    
    while True:
        a += 1.0
        c *= x / a
        series_sum += c
        if c / series_sum < eps:
            break

        if a > p + 1e6:
            break
    
    value = e * series_sum

    value = min(max(value, 0.0), 1.0)
    return value, 0


def gamma_cdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    
    x_flat = np.atleast_1d(x)
    cdf = np.zeros_like(x_flat, dtype=float)
    for i, xi in enumerate(x_flat):
        if xi <= 0:
            cdf[i] = 0.0
        else:
            val, _ = gammds(xi / scale, shape)
            cdf[i] = val
    return cdf


def estimate_gamma_parameters(data: np.ndarray) -> Tuple[float, float]:
    from scipy.special import digamma, polygamma
    
    data = np.array(data)
    if np.any(data <= 0):
        raise ValueError("All data points must be positive")
    
    mean_log = np.mean(np.log(data))
    log_mean = np.log(np.mean(data))
    s = log_mean - mean_log
    

    alpha = 0.5 / s if s > 0 else 1.0
    

    for _ in range(100):
        f = np.log(alpha) - digamma(alpha) - s
        fp = 1.0 / alpha - polygamma(1, alpha)
        if abs(fp) < 1e-14:
            break
        alpha_new = alpha - f / fp
        if alpha_new <= 0:
            alpha_new = alpha * 0.5
        if abs(alpha_new - alpha) < 1e-10:
            break
        alpha = alpha_new
    
    beta = np.mean(data) / alpha
    return float(alpha), float(beta)


def metastable_state_residence_time_distribution(residence_times: np.ndarray) -> dict:
    data = np.array(residence_times)
    if len(data) == 0:
        return {}
    
    mean_t = float(np.mean(data))
    var_t = float(np.var(data))
    
    try:
        shape, scale = estimate_gamma_parameters(data)
    except Exception:
        shape = mean_t ** 2 / max(var_t, 1e-12)
        scale = mean_t / max(shape, 1e-12)
    
    half_life = float(scale * shape * (2 ** (1.0 / shape) - 1.0))
    
    return {
        "mean": mean_t,
        "variance": var_t,
        "gamma_shape": shape,
        "gamma_scale": scale,
        "half_life": half_life,
        "n_samples": len(data),
    }


def chi_square_pvalue(chi2_stat: float, dof: int) -> float:
    if chi2_stat < 0 or dof <= 0:
        return 0.0
    val, _ = gammds(chi2_stat / 2.0, dof / 2.0)
    pvalue = 1.0 - val
    return float(max(0.0, min(1.0, pvalue)))
