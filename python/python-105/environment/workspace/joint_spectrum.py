
import numpy as np
from scipy.linalg import svd


def normalized_sinc(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    s = np.ones_like(x, dtype=np.float64)
    nz = np.abs(x) > 1e-16
    x_nz = x[nz]
    s[nz] = np.sin(np.pi * x_nz) / (np.pi * x_nz)
    return s


def phase_mismatch(omega_s: np.ndarray, omega_i: np.ndarray,
                   omega_p0: float, Lambda: float,
                   sellmeier_p: callable, sellmeier_s: callable,
                   sellmeier_i: callable) -> np.ndarray:
    c = 2.99792458e8
    omega_p = omega_s + omega_i

    kp = sellmeier_p(omega_p) * omega_p / c
    ks = sellmeier_s(omega_s) * omega_s / c
    ki = sellmeier_i(omega_i) * omega_i / c
    dk = kp - ks - ki - 2.0 * np.pi / Lambda
    return dk


def pump_envelope_gaussian(omega_sum: np.ndarray,
                           omega_p0: float,
                           sigma_p: float) -> np.ndarray:
    if sigma_p <= 0.0:
        raise ValueError("sigma_p 必须为正。")
    return np.exp(-((omega_sum - omega_p0) ** 2) / (2.0 * sigma_p ** 2))


def phase_matching_function(dk: np.ndarray, L: float) -> np.ndarray:
    if L <= 0.0:
        raise ValueError("晶体长度 L 必须为正。")
    arg = dk * L / (2.0 * np.pi)
    return normalized_sinc(arg)


def compute_jsa(omega_s: np.ndarray, omega_i: np.ndarray,
                omega_p0: float, sigma_p: float, L: float, Lambda: float,
                sellmeier_p: callable, sellmeier_s: callable,
                sellmeier_i: callable) -> np.ndarray:
    Os, Oi = np.meshgrid(omega_s, omega_i, indexing='ij')
    alpha = pump_envelope_gaussian(Os + Oi, omega_p0, sigma_p)
    dk = phase_mismatch(Os, Oi, omega_p0, Lambda,
                        sellmeier_p, sellmeier_s, sellmeier_i)
    phi = phase_matching_function(dk, L)
    jsa = alpha * phi

    norm = np.sqrt(np.sum(np.abs(jsa) ** 2))
    if norm > 1e-20:
        jsa /= norm
    return jsa


def schmidt_decomposition_jsa(jsa: np.ndarray) -> tuple:


    raise NotImplementedError("Hole 1: 请实现 Schmidt 分解")
