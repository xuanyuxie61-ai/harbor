
import numpy as np
from typing import Tuple


def check_numerical_stability(values: np.ndarray,
                              name: str = "array",
                              tol: float = 1e10) -> bool:
    has_nan = np.any(np.isnan(values))
    has_inf = np.any(np.isinf(values))
    max_abs = np.max(np.abs(values))

    if has_nan:
        print(f"[WARNING] {name} contains NaN values")
        return False
    if has_inf:
        print(f"[WARNING] {name} contains Inf values")
        return False
    if max_abs > tol:
        print(f"[WARNING] {name} has large values: max_abs = {max_abs:.2e}")
        return False

    return True


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return a / np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)


def entropy(p: np.ndarray) -> float:
    p = np.array(p)
    p = p[p > 0]
    p = p / np.sum(p)
    return -np.sum(p * np.log(p))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.array(p)
    q = np.array(q)
    mask = (p > 1e-15) & (q > 1e-15)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))
