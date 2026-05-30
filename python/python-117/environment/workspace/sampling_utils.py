
import numpy as np
from typing import Tuple






def pdf_to_histogram(pdf_func: callable,
                     n_bins: int = 64,
                     x_min: float = -1.0,
                     x_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_l = np.linspace(x_min, x_max, n_bins + 1)[:-1]
    b_r = np.linspace(x_min, x_max, n_bins + 1)[1:]
    b_m = 0.5 * (b_l + b_r)
    b_p = np.array([pdf_func(xm) for xm in b_m], dtype=np.float64)

    b_p = np.clip(b_p, 0.0, None)
    return b_p, b_l, b_r


def histogram_to_cdf(b_p: np.ndarray, b_l: np.ndarray, b_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    widths = b_r - b_l
    mass = b_p * widths
    total = np.sum(mass)
    if total <= 0:
        total = 1.0
    c_x = np.concatenate(([b_l[0]], b_r))
    c_y = np.zeros(len(b_p) + 1, dtype=np.float64)
    for i in range(len(b_p)):
        c_y[i + 1] = c_y[i] + mass[i] / total
    c_y[-1] = 1.0
    return c_x, c_y


def cdf_to_sample(c_x: np.ndarray, c_y: np.ndarray, n_samples: int) -> np.ndarray:
    u = np.random.rand(n_samples)
    samples = np.empty(n_samples, dtype=np.float64)
    n = len(c_y)
    for k in range(n_samples):
        uk = u[k]

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if c_y[mid] <= uk:
                lo = mid
            else:
                hi = mid
        left = lo

        dy = c_y[left + 1] - c_y[left]
        if abs(dy) < 1e-30:
            samples[k] = c_x[left]
        else:
            frac = (uk - c_y[left]) / dy
            samples[k] = c_x[left] + frac * (c_x[left + 1] - c_x[left])
    return samples






def sphere_sample_marsaglia(n: int) -> np.ndarray:
    v = np.random.randn(3, n)
    norms = np.linalg.norm(v, axis=0)
    norms[norms == 0] = 1.0
    points = v / norms
    return points


def svd_deformation_modes(displacement_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(displacement_matrix, full_matrices=False)
    return U, S, Vt


def sample_random_orientation() -> np.ndarray:
    M = np.random.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def boltzmann_acceptance(delta_E: float, T: float = 300.0,
                         k_B: float = 8.314e-3) -> bool:
    if delta_E < 0:
        return True
    p = np.exp(-delta_E / (k_B * T))
    return np.random.rand() < p
