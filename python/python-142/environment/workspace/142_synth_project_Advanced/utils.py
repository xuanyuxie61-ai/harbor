
import numpy as np
import math
from typing import Tuple, Optional


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("求积点数 n 必须为正整数")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def logistic_transform(z: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return a + (b - a) / (1.0 + np.exp(-z))


def inverse_logistic_transform(y: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    eps = 1e-12
    y_clip = np.clip(y, a + eps, b - eps)
    return -np.log((b - a) / (y_clip - a) - 1.0)


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0.0))


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(lambda t: math.erf(t / np.sqrt(2.0)))(x))


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def is_positive_definite(M: np.ndarray, tol: float = 1e-10) -> bool:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals > tol))


def nearest_correlation_matrix(A: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    n = A.shape[0]
    R = np.copy(A)
    dS = np.zeros_like(A)
    Y = np.copy(A)
    for k in range(max_iter):
        R = Y - dS

        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        dS = X - R
        np.fill_diagonal(dS, 0.0)
        Y = np.copy(X)
        np.fill_diagonal(Y, 1.0)
        err = np.max(np.abs(Y - X))
        if err < tol:
            break
    np.fill_diagonal(Y, 1.0)
    return Y


def cholesky_with_pivot(A: np.ndarray, tol: float = 1e-12) -> Optional[np.ndarray]:
    try:
        L = np.linalg.cholesky(A)
        return L
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, tol)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        return L


def finite_difference_1d_second_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    n = len(u)
    if n < 3:
        raise ValueError("数组长度至少为 3")
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / (dx * dx)

    d2u[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / (dx * dx)
    d2u[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]) / (dx * dx)
    return d2u


def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("三对角矩阵维度不匹配")
    cp = np.zeros(n - 1, dtype=float)
    dp = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
