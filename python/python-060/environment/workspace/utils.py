
import numpy as np
from typing import Tuple, Optional


def safe_exp(x: np.ndarray, max_val: float = 700.0) -> np.ndarray:
    x_clipped = np.clip(x, -max_val, max_val)
    return np.exp(x_clipped)


def safe_log(x: np.ndarray, min_val: float = 1e-300) -> np.ndarray:
    x_safe = np.where(x > min_val, x, min_val)
    return np.log(x_safe)


def normalize_array(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-30)


def finite_difference_gradient(f: callable, x: np.ndarray,
                                h: float = 1e-6) -> np.ndarray:
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h)
    return grad


def relative_error(approx: float, exact: float) -> float:
    if abs(exact) < 1e-30:
        return abs(approx - exact)
    return abs(approx - exact) / abs(exact)


def convergence_rate(errors: np.ndarray) -> np.ndarray:
    if len(errors) < 2:
        return np.array([])
    rates = np.zeros(len(errors) - 1)
    for i in range(len(errors) - 1):
        if errors[i] > 1e-30 and errors[i + 1] > 1e-30:
            rates[i] = np.log(errors[i + 1] / errors[i]) / np.log(0.5)
        else:
            rates[i] = 0.0
    return rates


def check_array_bounds(arr: np.ndarray, name: str = "array",
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> bool:
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} 包含 NaN")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} 包含 Inf")
    if min_val is not None and np.any(arr < min_val):
        raise ValueError(f"{name} 包含小于 {min_val} 的值")
    if max_val is not None and np.any(arr > max_val):
        raise ValueError(f"{name} 包含大于 {max_val} 的值")
    return True


def house_transform(v: np.ndarray) -> np.ndarray:
    n = len(v)
    x = v.copy()
    alpha = -np.sign(x[0]) * np.linalg.norm(x)
    u = x.copy()
    u[0] -= alpha

    norm_u2 = np.dot(u, u)
    if norm_u2 < 1e-30:
        return np.eye(n)

    H = np.eye(n) - 2.0 * np.outer(u, u) / norm_u2
    return H


def print_matrix_summary(A: np.ndarray, name: str = "Matrix"):
    print(f"\n{name}:")
    print(f"  Shape: {A.shape}")
    print(f"  Min: {np.min(A):.6e}")
    print(f"  Max: {np.max(A):.6e}")
    print(f"  Mean: {np.mean(A):.6e}")
    print(f"  Frobenius norm: {np.linalg.norm(A):.6e}")
    if A.shape[0] == A.shape[1]:
        try:
            cond = np.linalg.cond(A)
            print(f"  Condition number: {cond:.6e}")
        except:
            print(f"  Condition number: N/A")
