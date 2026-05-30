
import numpy as np
from typing import Tuple, List
from math import factorial, sqrt


def legendre_polynomial(l: int, x: np.ndarray) -> np.ndarray:
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x.copy()
    p_prev2 = np.ones_like(x)
    p_prev1 = x.copy()
    for n in range(1, l):
        p_curr = ((2.0 * n + 1.0) * x * p_prev1 - n * p_prev2) / (n + 1.0)
        p_prev2 = p_prev1
        p_prev1 = p_curr
    return p_prev1


def associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    m_abs = abs(m)
    if m_abs > l:
        return np.zeros_like(x)


    p = np.ones_like(x)
    if m_abs > 0:
        p = (-1.0) ** m_abs * (1.0 - x ** 2) ** (m_abs / 2.0)

        for k in range(1, m_abs + 1):
            p *= (2.0 * k - 1.0)

    if l == m_abs:
        p_lm = p
    else:
        p_lm_prev = p

        p_lm = x * (2.0 * m_abs + 1.0) * p_lm_prev
        for n in range(m_abs + 1, l):
            p_lm_next = ((2.0 * n + 1.0) * x * p_lm - (n + m_abs) * p_lm_prev) / (n - m_abs + 1.0)
            p_lm_prev = p_lm
            p_lm = p_lm_next

    norm = sqrt((2.0 * l + 1.0) * factorial(l - m_abs) / (4.0 * np.pi * factorial(l + m_abs)))
    return norm * p_lm


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.cos(theta)
    plm = associated_legendre(l, abs(m), x)
    if m > 0:
        y = plm * np.cos(m * phi)
    elif m < 0:
        y = plm * np.sin(abs(m) * phi)
    else:
        y = plm
    return y, plm


def spherical_harmonics_expansion(values: np.ndarray, theta: np.ndarray,
                                   phi: np.ndarray, l_max: int = 4) -> dict:
    coefficients = {}

    n_points = len(values)
    d_omega = 4.0 * np.pi / n_points
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y_lm, _ = spherical_harmonic(l, m, theta, phi)

            integrand = values * y_lm * np.sin(theta)
            c_lm = np.sum(integrand) * d_omega
            coefficients[(l, m)] = c_lm
    return coefficients


def project_to_sphere(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return data / norms


def spherical_coordinates(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = data.shape[1]
    if D != 3:

        from linear_algebra_core import jacobi_eigenvalue
        cov = np.cov(data.T)
        eigvals, eigvecs = jacobi_eigenvalue(cov)
        data_3d = data @ eigvecs[:, :3]
        data = data_3d
    r = np.linalg.norm(data, axis=1)
    r = np.where(r < 1e-15, 1.0, r)
    theta = np.arccos(np.clip(data[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(data[:, 1], data[:, 0])
    return theta, phi


def high_dim_spherical_harmonics_spectrum(data: np.ndarray, l_max: int = 4) -> np.ndarray:
    data_centered = data - np.mean(data, axis=0)
    data_sphere = project_to_sphere(data_centered)
    theta, phi = spherical_coordinates(data_sphere)

    values = np.ones(len(data))
    coeffs = spherical_harmonics_expansion(values, theta, phi, l_max)

    spectrum = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        energy = 0.0
        for m in range(-l, l + 1):
            energy += coeffs[(l, m)] ** 2
        spectrum[l] = energy
    return spectrum


def reconstruct_from_harmonics(coefficients: dict, theta: np.ndarray,
                                phi: np.ndarray, l_max: int = 4) -> np.ndarray:
    result = np.zeros_like(theta)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y_lm, _ = spherical_harmonic(l, m, theta, phi)
            result += coefficients[(l, m)] * y_lm
    return result
