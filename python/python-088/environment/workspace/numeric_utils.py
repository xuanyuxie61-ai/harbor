
import numpy as np
from typing import Tuple, Optional


def nextafter(x: float, direction: str = "up") -> float:
    if direction == "up":
        return np.nextafter(x, np.inf)
    else:
        return np.nextafter(x, -np.inf)


def machine_epsilon() -> float:
    return np.finfo(float).eps


def safe_divide(a: np.ndarray, b: np.ndarray, tol: float = None) -> np.ndarray:
    if tol is None:
        tol = machine_epsilon() * 10.0
    b_safe = np.where(np.abs(b) < tol, np.sign(b + tol) * tol, b)
    return a / b_safe


def condition_number_check(A: np.ndarray, threshold: float = 1e12) -> bool:
    cond = np.linalg.cond(A)
    return cond < threshold


def square_distance_pdf(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    mask1 = (d >= 0.0) & (d <= 1.0)
    mask2 = (d > 1.0) & (d <= np.sqrt(2.0))

    pdf[mask1] = 2.0 * d[mask1] * (d[mask1]**2 - 4.0 * d[mask1] + np.pi)

    sqrt_term = np.sqrt(d[mask2]**2 - 1.0)
    pdf[mask2] = 2.0 * d[mask2] * (
        4.0 * sqrt_term
        - (d[mask2]**2 + 2.0 - np.pi)
        - 4.0 * np.arctan(sqrt_term)
    )
    return pdf


def square_distance_cdf(r: float) -> float:
    if r <= 0.0:
        return 0.0
    if r >= np.sqrt(2.0):
        return 1.0


    n_points = 1000
    d_samples = np.linspace(0.0, r, n_points)
    pdf_vals = square_distance_pdf(d_samples)
    cdf = np.trapezoid(pdf_vals, d_samples)

    d_full = np.linspace(0.0, np.sqrt(2.0), n_points)
    pdf_full = square_distance_pdf(d_full)
    total = np.trapezoid(pdf_full, d_full)
    return cdf / total if total > 0 else 0.0


def compute_moments(samples: np.ndarray) -> Tuple[float, float, float, float]:
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    mu = np.mean(samples)
    if n < 2:
        return mu, 0.0, 0.0, 0.0

    var = np.var(samples, ddof=1)
    std = np.sqrt(var) if var > 0 else 1.0

    skew = np.mean(((samples - mu) / std) ** 3) if std > 0 else 0.0
    kurt = np.mean(((samples - mu) / std) ** 4) - 3.0 if std > 0 else 0.0

    return float(mu), float(var), float(skew), float(kurt)


def chebyshev_bound(mean: float, std: float, k: float) -> float:
    if k <= 0:
        return 1.0
    return 1.0 / (k ** 2)


def gershgorin_discs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii


def is_diagonally_dominant(A: np.ndarray, strict: bool = True) -> bool:
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = np.sum(np.abs(A[i, :])) - diag
        if strict:
            if diag <= off_diag:
                return False
        else:
            if diag < off_diag:
                return False
    return True


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        return np.linalg.norm(b - A @ x)
    return np.linalg.norm(b - A @ x) / norm_b
