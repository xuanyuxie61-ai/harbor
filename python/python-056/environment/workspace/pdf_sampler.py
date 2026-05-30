
import numpy as np
from typing import Tuple


def set_discrete_cdf(pdf_mat: np.ndarray) -> np.ndarray:
    pdf = np.asarray(pdf_mat, dtype=float)
    if np.any(pdf < 0):
        raise ValueError("set_discrete_cdf: PDF 元素不能为负")
    total = np.sum(pdf)
    if total < 1e-14:
        raise ValueError("set_discrete_cdf: PDF 总和为零")
    pdf_norm = pdf / total
    cdf = np.cumsum(np.cumsum(pdf_norm, axis=0), axis=1)
    return cdf


def discrete_cdf_to_xy(
    m1: int,
    m2: int,
    cdf_mat: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    n: int,
    u: np.ndarray,
) -> np.ndarray:
    s = np.zeros((2, n))
    low = 0.0
    cdf = np.asarray(cdf_mat)
    for j in range(m2):
        for i in range(m1):
            high = cdf[i, j]
            mask = (low <= u) & (u <= high)
            count = np.count_nonzero(mask)
            if count > 0:
                r = np.random.rand(2, count)
                idx = np.where(mask)[0]
                s[0, idx] = (1.0 - r[0, :]) * xb[i] + r[0, :] * xb[i + 1]
                s[1, idx] = (1.0 - r[1, :]) * yb[j] + r[1, :] * yb[j + 1]
            low = high
    return s


def sample_velocity_2d(
    pdf_mat: np.ndarray,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_samples: int = 1000,
) -> np.ndarray:
    m1, m2 = pdf_mat.shape
    cdf_mat = set_discrete_cdf(pdf_mat)
    xb = np.linspace(u_range[0], u_range[1], m1 + 1)
    yb = np.linspace(v_range[0], v_range[1], m2 + 1)
    u_rand = np.random.rand(n_samples)
    samples = discrete_cdf_to_xy(m1, m2, cdf_mat, xb, yb, n_samples, u_rand)
    return samples


def estimate_power_statistics(
    pdf_mat: np.ndarray,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    turbine_area: float = 20.0,
    rho: float = 1025.0,
    n_samples: int = 5000,
) -> dict:
    samples = sample_velocity_2d(pdf_mat, u_range, v_range, n_samples)
    u = samples[0, :]
    v = samples[1, :]
    speed = np.sqrt(u ** 2 + v ** 2)


    cp = 16.0 / 27.0
    power = 0.5 * rho * turbine_area * cp * speed ** 3


    rated_speed = 2.5
    rated_power = 0.5 * rho * turbine_area * cp * rated_speed ** 3
    power = np.minimum(power, rated_power)

    mean_p = float(np.mean(power))
    std_p = float(np.std(power))
    max_p = float(np.max(power))
    cap_factor = mean_p / rated_power if rated_power > 0 else 0.0

    return {
        "mean_power": mean_p,
        "std_power": std_p,
        "max_power": max_p,
        "rated_power": rated_power,
        "capacity_factor": cap_factor,
        "samples": samples,
        "speeds": speed,
    }
