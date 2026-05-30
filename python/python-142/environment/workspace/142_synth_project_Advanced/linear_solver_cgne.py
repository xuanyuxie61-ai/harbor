
import numpy as np
from typing import Tuple, Optional, Callable


def cg_ne_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    tol: float = 1e-10
) -> Tuple[np.ndarray, int, float]:
    m, n = A.shape
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A @ x
    z = A.T @ r
    d = z.copy()
    rz_old = np.dot(z, z)

    for k in range(max_iter):
        Ad = A @ d
        denom = np.dot(Ad, Ad)
        if denom < 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * d
        r -= alpha * Ad
        z = A.T @ r
        rz_new = np.dot(z, z)
        residual_norm = np.sqrt(rz_new)
        if residual_norm < tol:
            return x, k + 1, residual_norm
        beta = rz_new / rz_old
        d = z + beta * d
        rz_old = rz_new

    return x, max_iter, np.sqrt(rz_old)


def cg_ne_solve_with_regularization(
    A: np.ndarray,
    b: np.ndarray,
    lam: float = 1e-6,
    x0: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    tol: float = 1e-10
) -> Tuple[np.ndarray, int, float]:
    m, n = A.shape
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A @ x
    z = A.T @ r - lam * x
    d = z.copy()
    rz_old = np.dot(z, z)

    for k in range(max_iter):
        Ad = A @ d
        denom = np.dot(Ad, Ad) + lam * np.dot(d, d)
        if denom < 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * d
        r -= alpha * Ad
        z = A.T @ r - lam * x
        rz_new = np.dot(z, z)
        residual_norm = np.sqrt(rz_new)
        if residual_norm < tol:
            return x, k + 1, residual_norm
        beta = rz_new / rz_old
        d = z + beta * d
        rz_old = rz_new

    return x, max_iter, np.sqrt(rz_old)


def helmert_matrix(n: int) -> np.ndarray:
    H = np.zeros((n, n), dtype=float)
    H[0, :] = 1.0 / np.sqrt(n)
    for i in range(1, n):
        H[i, :i] = 1.0 / np.sqrt(i * (i + 1))
        H[i, i] = -i / np.sqrt(i * (i + 1))
    return H


def lesp_matrix(m: int, n: int) -> np.ndarray:
    A = np.zeros((m, n), dtype=float)
    for i in range(min(m, n)):
        A[i, i] = -(2.0 * (i + 1) + 3.0)
    for i in range(1, min(m, n)):
        A[i, i - 1] = 1.0 / (i + 1)
    for i in range(min(m, n - 1)):
        A[i, i + 1] = (i + 1) + 1.0
    return A


def test_cg_ne():
    np.random.seed(42)
    n = 20

    H = helmert_matrix(n)
    x_true = np.random.randn(n)
    b = H @ x_true
    x_sol, iters, res = cg_ne_solve(H, b, tol=1e-12)
    err = np.linalg.norm(x_sol - x_true)
    assert err < 1e-8, f"CG-NE 解误差过大: {err}"
    print(f"cg_ne test passed. iters={iters}, residual={res:.2e}, error={err:.2e}")


if __name__ == "__main__":
    test_cg_ne()
