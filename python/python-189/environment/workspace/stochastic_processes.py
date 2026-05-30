
import numpy as np
from typing import Tuple






def direction_uniform_nd(d: int) -> np.ndarray:
    if d < 1:
        raise ValueError("direction_uniform_nd: d must be positive")
    z = np.random.randn(d)
    norm = np.linalg.norm(z)
    if norm < 1.0e-12:
        return direction_uniform_nd(d)
    return z / norm


def brownian_motion(n: int, d: int, sigma: float = 1.0) -> np.ndarray:
    if n < 1 or d < 1:
        raise ValueError("brownian_motion: n and d must be positive")
    X = np.zeros((n, d))
    for i in range(1, n):
        r = abs(np.random.randn())
        direction = direction_uniform_nd(d)
        X[i, :] = X[i - 1, :] + sigma * r * direction
    return X


def ornstein_uhlenbeck_process(n: int, d: int, theta: float = 0.15,
                                sigma: float = 0.2, dt: float = 0.01) -> np.ndarray:
    X = np.zeros((n, d))
    X[0, :] = np.random.randn(d) * sigma / np.sqrt(2.0 * theta)
    for i in range(1, n):
        dW = np.random.randn(d) * np.sqrt(dt)
        X[i, :] = X[i - 1, :] - theta * X[i - 1, :] * dt + sigma * dW
    return X






def multivariate_normal_distance_stats(m: int, n_samples: int = 10000) -> Tuple[float, float]:
    if m < 1:
        raise ValueError("multivariate_normal_distance_stats: m must be positive")
    t = np.zeros(n_samples)
    for i in range(n_samples):
        p = np.random.randn(m)
        q = np.random.randn(m)
        t[i] = np.linalg.norm(p - q)
    mu = float(np.mean(t))
    if n_samples > 1:
        var = float(np.var(t, ddof=1))
    else:
        var = 0.0
    return mu, var


def theoretical_chi_mean(d: int) -> float:
    from math import gamma, sqrt
    return sqrt(2.0) * gamma((d + 1) / 2.0) / gamma(d / 2.0)






def gaussian_kernel_matrix(states: np.ndarray, sigma: float = None) -> np.ndarray:
    n = states.shape[0]
    K = np.zeros((n, n))
    if sigma is None:

        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(states[i] - states[j]))
        if len(dists) == 0:
            sigma = 1.0
        else:
            sigma = float(np.median(dists)) + 1.0e-8
    for i in range(n):
        for j in range(n):
            d2 = np.sum((states[i] - states[j]) ** 2)
            K[i, j] = np.exp(-d2 / (2.0 * sigma ** 2))
    return K
