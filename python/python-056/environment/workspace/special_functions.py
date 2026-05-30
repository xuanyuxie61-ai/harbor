
import numpy as np
from typing import Tuple


def digamma(x: float) -> Tuple[float, int]:
    if x <= 0.0:
        return 0.0, 1

    euler_mascheroni = 0.57721566490153286060
    value = 0.0


    if x <= 1.0e-6:
        value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
        return value, 0


    while x < 8.5:
        value = value - 1.0 / x
        x = x + 1.0


    r = 1.0 / x
    value = value + np.log(x) - 0.5 * r
    r2 = r * r
    value -= r2 * (1.0 / 12.0
                   - r2 * (1.0 / 120.0
                           - r2 * (1.0 / 252.0
                                   - r2 * (1.0 / 240.0
                                           - r2 * (1.0 / 132.0)))))
    return value, 0


def digamma_vector(x_arr: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_arr, dtype=float)
    if np.any(x_arr <= 0):
        raise ValueError("digamma_vector: 所有输入必须大于 0")
    return np.array([digamma(x)[0] for x in x_arr.flat]).reshape(x_arr.shape)


def tidal_digamma_modulation(freq_ratio: float, n_harmonics: int = 6) -> float:
    if freq_ratio <= 0.0:
        raise ValueError("tidal_digamma_modulation: freq_ratio 必须大于 0")
    total = 0.0
    for k in range(1, n_harmonics + 1):
        psi_knu, _ = digamma(k + freq_ratio)
        psi_k, _ = digamma(float(k))
        total += (psi_knu - psi_k) / (k * k)
    return total


def polygamma2(x: float) -> float:
    if x <= 0.0:
        raise ValueError("polygamma2: x 必须大于 0")
    if x < 1.0e-3:
        return 1.0 / (x * x) + np.pi * np.pi / 6.0
    r = 1.0 / x
    r2 = r * r
    return r + 0.5 * r2 + r2 * r / 6.0 - r2 * r2 * r / 30.0
