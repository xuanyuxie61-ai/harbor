
import numpy as np
import math
from typing import Tuple





def r8lt_det(n: int, a: np.ndarray) -> float:
    if n <= 0:
        raise ValueError("Matrix order n must be positive.")
    a = np.asarray(a, dtype=float)
    if a.shape != (n, n):
        raise ValueError(f"Matrix shape {a.shape} does not match expected ({n}, {n}).")

    det = 1.0
    for i in range(n):
        det *= a[i, i]
    return det


def r8lt_inverse(n: int, a: np.ndarray) -> np.ndarray:
    if n <= 0:
        raise ValueError("Matrix order n must be positive.")
    a = np.asarray(a, dtype=float)
    inv = np.zeros((n, n), dtype=float)

    for i in range(n):
        if abs(a[i, i]) < 1e-30:
            raise ValueError(f"Singular matrix: diagonal element a[{i},{i}] = 0.")
        inv[i, i] = 1.0 / a[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s += a[i, k] * inv[k, j]
            inv[i, j] = -s / a[i, i]
    return inv





def hilbert_matrix(m: int, n: int = None) -> np.ndarray:
    if n is None:
        n = m
    if m <= 0 or n <= 0:
        raise ValueError("Dimensions must be positive.")

    a = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            a[i, j] = 1.0 / (i + j + 1)
    return a


def hilbert_inverse(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    from math import factorial

    inv = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sign = (-1)**(i + j)
            num = factorial(n + i) * factorial(n + j)
            den = (i + j + 1) * (factorial(i) * factorial(j))**2 * factorial(n - 1 - i) * factorial(n - 1 - j)
            inv[i, j] = sign * num / den
    return inv





def matrix_condition_number_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)

    s = np.linalg.svd(a, compute_uv=False)
    s_max = np.max(s)
    s_min = np.min(s[s > 1e-15]) if np.any(s > 1e-15) else 1e-30
    return s_max / s_min


def estimate_tov_stability_matrix(
    radius: np.ndarray,
    pressure: np.ndarray,
    mass: np.ndarray,
    energy_density: np.ndarray
) -> np.ndarray:
    n = len(radius)
    if n < 2:
        raise ValueError("Need at least 2 radial points.")

    J = np.zeros((2 * n, 2 * n))
    Gc2 = 6.67430e-11 / (2.99792458e8)**2

    for i in range(n - 1):
        r = radius[i]
        P = pressure[i]
        m = mass[i]
        eps = energy_density[i]

        if r < 1e-10:
            continue

        denom = r * (r - 2.0 * Gc2 * m)
        if abs(denom) < 1e-30:
            continue


        dr = radius[i + 1] - radius[i]
        J[2*i, 2*i] = -1.0
        J[2*i, 2*i + 2] = 1.0
        J[2*i + 1, 2*i + 1] = -1.0
        J[2*i + 1, 2*i + 3] = 1.0

    return J


def analyze_eigenvalue_stability(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    eigs = np.linalg.eigvals(a)
    real_parts = np.real(eigs)
    is_stable = np.all(real_parts < 1e-10)
    return eigs, is_stable


def test_numerical_stability_on_hilbert(n_max: int = 10) -> dict:
    results = {}
    for n in range(2, n_max + 1):
        H = hilbert_matrix(n)
        x_exact = np.ones(n)
        b = H @ x_exact

        try:
            x_numerical = np.linalg.solve(H, b)
            error = np.linalg.norm(x_numerical - x_exact)
            cond = matrix_condition_number_1d(H)
            results[n] = {
                'error': error,
                'condition_number': cond,
                'log_error': math.log10(error + 1e-20)
            }
        except np.linalg.LinAlgError:
            results[n] = {
                'error': float('inf'),
                'condition_number': float('inf'),
                'log_error': float('inf')
            }

    return results
