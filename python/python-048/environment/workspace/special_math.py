
import numpy as np
from typing import Tuple


def trigamma(x: float) -> Tuple[float, int]:
    a = 1.0e-4
    b = 5.0
    b2 = 0.1666666667
    b4 = -0.03333333333
    b6 = 0.02380952381
    b8 = -0.03333333333

    if x <= 0.0:
        return 0.0, 1

    ifault = 0
    z = x


    if x <= a:
        return 1.0 / (x * x), ifault


    value = 0.0
    while z < b:
        value += 1.0 / (z * z)
        z += 1.0


    y = 1.0 / (z * z)
    value += 0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z

    return value, ifault


def trigamma_array(x_arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(x_arr, dtype=float)
    for i in range(x_arr.size):
        val, flt = trigamma(float(x_arr.flat[i]))
        out.flat[i] = val if flt == 0 else np.nan
    return out


def fracture_size_pdf(a: np.ndarray, a_min: float, D_f: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a_min <= 0:
        raise ValueError("a_min 必须为正数")
    if not (1.0 < D_f < 3.0):
        raise ValueError("分形维数 D_f 应在 (1, 3) 区间")

    pdf = np.zeros_like(a)
    mask = a >= a_min
    pdf[mask] = (D_f / a_min) * (a[mask] / a_min) ** (-(D_f + 1.0))
    return pdf


def stress_intensity_factor(sigma: float, a: float, geometry_factor: float = 1.0) -> float:
    if sigma < 0 or a < 0:
        return 0.0
    return sigma * np.sqrt(np.pi * a) * geometry_factor


def log_likelihood_fracture_size(a_obs: np.ndarray, a_min: float, D_f: float) -> float:
    a_obs = np.asarray(a_obs, dtype=float)
    if np.any(a_obs < a_min):
        return -np.inf
    N = a_obs.size
    return N * np.log(D_f / a_min) - (D_f + 1.0) * np.sum(np.log(a_obs / a_min))
