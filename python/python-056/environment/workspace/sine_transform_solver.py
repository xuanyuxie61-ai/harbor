
import numpy as np
from typing import Tuple


def sine_transform_data(n: int, d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).flatten()
    if d.size < n:
        raise ValueError("sine_transform_data: 输入数据长度不足")
    s = np.zeros(n)
    coeff = np.pi / (n + 1)
    scale = np.sqrt(2.0 / (n + 1))
    for k in range(1, n + 1):
        angles = coeff * k * np.arange(1, n + 1)
        s[k - 1] = np.sum(np.sin(angles) * d[:n])
    s *= scale
    return s


def inverse_sine_transform(n: int, s: np.ndarray) -> np.ndarray:
    return sine_transform_data(n, s)


def dst_fast(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    n = d.size
    y = np.zeros(2 * n + 2)
    y[1:n + 1] = d
    y[n + 2:2 * n + 2] = -d[::-1]
    y_fft = np.fft.fft(y)
    s = -np.imag(y_fft[1:n + 1]) / 2.0

    scale = np.sqrt(2.0 / (n + 1))
    return s * scale


def solve_poisson_1d(
    f: np.ndarray,
    L: float = 1.0,
) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    N = f.size
    h = L / (N + 1)
    s_f = dst_fast(f)
    k = np.arange(1, N + 1)
    lam = -4.0 / (h * h) * np.sin(np.pi * k / (2.0 * (N + 1))) ** 2

    lam_safe = np.where(np.abs(lam) < 1e-14, 1.0, lam)
    s_u = s_f / lam_safe
    u = dst_fast(s_u)
    return u


def solve_helmholtz_1d(
    f: np.ndarray,
    kappa: float,
    L: float = 1.0,
) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    N = f.size
    h = L / (N + 1)
    s_f = dst_fast(f)
    k = np.arange(1, N + 1)
    lam = -4.0 / (h * h) * np.sin(np.pi * k / (2.0 * (N + 1))) ** 2 - kappa * kappa
    s_u = s_f / lam
    u = dst_fast(s_u)
    return u


def compute_wake_potential(
    thrust_distribution: np.ndarray,
    domain_length: float = 100.0,
    viscosity_scale: float = 0.01,
) -> np.ndarray:

    rhs = -np.gradient(thrust_distribution, domain_length / len(thrust_distribution))
    return solve_poisson_1d(rhs, L=domain_length)
