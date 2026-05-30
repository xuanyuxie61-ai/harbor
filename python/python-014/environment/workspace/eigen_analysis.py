
import numpy as np
from typing import Tuple
from utils import EPS_MACHINE, rms_norm


def power_method(
    A: np.ndarray,
    x0: np.ndarray = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> Tuple[float, np.ndarray, int]:
    N = A.shape[0]
    if x0 is None:
        x = np.random.randn(N)
    else:
        x = x0.copy()
    x = x / (np.linalg.norm(x) + EPS_MACHINE)
    lam_old = 0.0

    for it in range(max_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)
        if norm_y < EPS_MACHINE:
            break
        x_new = y / norm_y
        lam = float(np.dot(x_new, A @ x_new))
        if abs(lam - lam_old) < tol and np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        lam_old = lam


    lam = float(np.dot(x, A @ x) / np.dot(x, x))
    return lam, x, it + 1


def inverse_iteration(
    A: np.ndarray,
    shift: float = 0.0,
    x0: np.ndarray = None,
    max_iter: int = 300,
    tol: float = 1e-12,
) -> Tuple[float, np.ndarray, int]:
    N = A.shape[0]
    if x0 is None:
        x = np.random.randn(N)
    else:
        x = x0.copy()
    x = x / (np.linalg.norm(x) + EPS_MACHINE)
    M = A - shift * np.eye(N)

    for it in range(max_iter):
        try:
            y = np.linalg.solve(M, x)
        except np.linalg.LinAlgError:

            M_reg = M + EPS_MACHINE * np.eye(N)
            y = np.linalg.solve(M_reg, x)
        norm_y = np.linalg.norm(y)
        if norm_y < EPS_MACHINE:
            break
        x_new = y / norm_y
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    lam = float(np.dot(x, A @ x) / np.dot(x, x))
    return lam, x, it + 1


def spectral_gap_and_soft_modes(
    J: np.ndarray, n_soft: int = 3, tol: float = 1e-10
) -> Tuple[float, float, np.ndarray, np.ndarray]:

    w, v = np.linalg.eigh(J)
    lambda_min = float(w[0])
    if w.size > 1:
        gap = float(w[1] - w[0])
    else:
        gap = 0.0
    return lambda_min, gap, w, v


def spin_wave_dispersion_1d(J: float, S: float, a: float, k_points: np.ndarray) -> np.ndarray:
    omega = 2.0 * J * S * (1.0 - np.cos(k_points * a))
    return omega


def correlation_length_from_gap(gap: float, J: float, a: float = 1.0) -> float:
    if gap <= EPS_MACHINE:
        return float("inf")
    return a / np.sqrt(gap / J)
