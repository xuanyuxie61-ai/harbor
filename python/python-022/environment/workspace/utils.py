
import numpy as np
from typing import Tuple


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1.0e-300
    result[mask] = a[mask] / b[mask]
    return result


def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def clamp_array(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip(arr, vmin, vmax)


def spherical_volume(r_inner: float, r_outer: float) -> float:
    if r_outer < r_inner or r_outer < 0.0 or r_inner < 0.0:
        return 0.0
    return (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)


def spherical_surface_area(r: float) -> float:
    if r < 0.0:
        return 0.0
    return 4.0 * np.pi * r * r


def log_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    ratio = a / b
    if np.abs(ratio - 1.0) < 1.0e-6:
        return 0.5 * (a + b)
    return (a - b) / np.log(ratio)


def vector_norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * v)))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = vector_norm(v)
    if norm < 1.0e-15:
        return np.zeros_like(v)
    return v / norm


def gauss_jacobi_quadrature_standard(n: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        return np.array([]), np.array([])
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Gauss-Jacobi 参数需满足 alpha > -1 且 beta > -1")


    ab = alpha + beta
    abi = 2.0 + ab


    diag = np.zeros(n)
    diag[0] = (beta - alpha) / abi
    if n > 1:
        a2b2 = beta * beta - alpha * alpha
        for i in range(1, n):
            idx = i + 1
            ab_i = 2.0 * idx + ab
            diag[i] = a2b2 / ((ab_i - 2.0) * ab_i)


    subdiag = np.zeros(n - 1)
    if n > 1:
        subdiag[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta)
                             / ((abi + 1.0) * abi * abi))
        for i in range(1, n - 1):
            idx = i + 1
            ab_i = 2.0 * idx + ab
            subdiag[i] = np.sqrt(4.0 * idx * (idx + alpha) * (idx + beta) * (idx + ab)
                                 / ((ab_i * ab_i - 1.0) * ab_i * ab_i))



    T = np.diag(diag) + np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)


    x = eigenvalues



    mu0 = 2.0**(ab + 1.0) * np.exp(
        np.math.lgamma(alpha + 1.0) + np.math.lgamma(beta + 1.0) - np.math.lgamma(ab + 2.0)
    )
    w = mu0 * (eigenvectors[0, :]**2)

    return x, w


def scale_quadrature(x: np.ndarray, w: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    if b <= a:
        raise ValueError("区间右端点必须大于左端点")
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return x * scale + shift, w * scale
