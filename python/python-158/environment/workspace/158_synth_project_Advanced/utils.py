
import numpy as np
from typing import Tuple


def safe_exp(x: float) -> float:
    if x > 700.0:
        return np.exp(700.0)
    if x < -700.0:
        return 0.0
    return np.exp(x)


def safe_log(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    return np.log(x)


def newton_raphson_scalar(
    f, df, x0: float, tol: float = 1e-12, max_iter: int = 100,
    x_min: float = -np.inf, x_max: float = np.inf
) -> Tuple[float, int, bool]:
    x = float(x0)
    for k in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, k, True
        dfx = df(x)
        if abs(dfx) < 1e-300:

            dfx = 1e-300 if fx >= 0 else -1e-300
        x_new = x - fx / dfx

        if x_min != -np.inf:
            x_new = max(x_new, x_min)
        if x_max != np.inf:
            x_new = min(x_new, x_max)
        if abs(x_new - x) < tol * (1.0 + abs(x)):
            return x_new, k + 1, True
        x = x_new
    return x, max_iter, False


def richardson_extrapolation(fh: float, fhm: float, p: float = 2.0) -> float:
    return (p * p * fhm - fh) / (p * p - 1.0)


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        return np.array([]), np.array([])
    

    x = np.cos(np.pi * (4.0 * np.arange(1, n + 1) - 1.0) / (4.0 * n + 2.0))
    
    eps = np.finfo(float).eps
    delta = 1.0
    while delta > eps:
        P0 = np.ones_like(x)
        P1 = x.copy()
        for k in range(1, n):
            P0, P1 = P1, ((2.0 * k + 1.0) * x * P1 - k * P0) / (k + 1.0)

        dP = n * (P0 - x * P1) / (1.0 - x * x)
        dx = P1 / dP
        x = x - dx
        delta = np.max(np.abs(dx))
    

    w = 2.0 / ((1.0 - x * x) * dP * dP)
    return x, w


def tridiagonal_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    n = len(diag)
    if n == 0:
        return np.array([])
    
    lower = lower.astype(float).copy()
    diag = diag.astype(float).copy()
    upper = upper.astype(float).copy()
    rhs = rhs.astype(float).copy()
    

    for i in range(1, n):
        if abs(diag[i - 1]) < 1e-300:
            diag[i - 1] = 1e-300
        m = lower[i] / diag[i - 1]
        diag[i] = diag[i] - m * upper[i - 1]
        rhs[i] = rhs[i] - m * rhs[i - 1]
    

    if abs(diag[-1]) < 1e-300:
        diag[-1] = 1e-300
    x = np.zeros(n)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        if abs(diag[i]) < 1e-300:
            diag[i] = 1e-300
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i]
    return x


def condition_estimate(A: np.ndarray) -> float:
    n = A.shape[0]
    if n == 0:
        return 1.0
    x = np.ones(n) / n
    ATA = A.T @ A + 1e-12 * np.eye(n)
    for _ in range(5):
        try:
            y = np.linalg.solve(ATA, x)
        except np.linalg.LinAlgError:
            y = x
            break
        norm_y = np.linalg.norm(y, 1)
        if norm_y < 1e-300:
            break
        x = y / norm_y
    
    norm_A = np.linalg.norm(A, 1)
    norm_Ainv = np.linalg.norm(x, 1)
    return norm_A * norm_Ainv


def adaptive_timestep(error_est: float, dt: float, atol: float = 1e-6,
                      rtol: float = 1e-4, safety: float = 0.9,
                      min_factor: float = 0.1, max_factor: float = 5.0) -> float:
    if error_est < 1e-300:
        return dt * max_factor
    factor = safety * np.sqrt(atol / error_est)
    factor = max(min_factor, min(max_factor, factor))
    return dt * factor
