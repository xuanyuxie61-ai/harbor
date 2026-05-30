
import time
import numpy as np
from typing import Tuple, Optional


class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()


def check_numerical_singularity(A: np.ndarray, tol: float = 1e-12) -> bool:
    if A.size == 0:
        return True

    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        diag_abs = np.abs(np.diag(A))
        if np.any(diag_abs < tol):
            return True

    cond = np.linalg.cond(A)
    return cond > 1.0 / tol


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    if np.isclose(b, 0.0, atol=1e-15):
        return fallback
    return a / b


def robust_sqrt(x: float, eps: float = 1e-14) -> float:
    return np.sqrt(max(float(x), eps))


def clip_to_bounds(val: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(val, lower, upper)


def finite_difference_jacobian(func, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = x.size
    fx = func(x)
    m = fx.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (func(x_plus) - func(x_minus)) / (2.0 * h)
    return J


def householder_reflection(v: np.ndarray) -> np.ndarray:
    v = v.astype(float).copy()
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.eye(len(v))
    v[0] += np.sign(v[0]) * norm_v
    H = np.eye(len(v)) - 2.0 * np.outer(v, v) / np.dot(v, v)
    return H


def gershgorin_discs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square matrix")
    n = A.shape[0]
    centers = np.diag(A).copy()
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii
