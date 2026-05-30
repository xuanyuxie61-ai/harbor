# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional





def cholesky_factorize(A: np.ndarray) -> Tuple[np.ndarray, int]:
    n = A.shape[0]
    R = A.copy().astype(float)
    info = 0

    for j in range(n):

        for k in range(j):
            t = np.dot(R[:k, k], R[:k, j])
            R[k, j] = (R[k, j] - t) / R[k, k]

        t = np.dot(R[:j, j], R[:j, j])
        s = R[j, j] - t

        if s <= 0.0:
            info = j + 1
            return R, info

        R[j, j] = np.sqrt(s)


    for i in range(n):
        for j in range(i):
            R[i, j] = 0.0

    return R, info





def solve_upper_triangular(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = b.copy().astype(float)

    for j in range(n - 1, -1, -1):
        if abs(U[j, j]) < 1e-15:
            raise ValueError(f"Zero diagonal entry in upper triangular system at index {j}")
        x[j] = x[j] / U[j, j]
        for i in range(j):
            x[i] = x[i] - U[i, j] * x[j]

    return x


def solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    x = b.copy().astype(float)

    for j in range(n):
        if abs(L[j, j]) < 1e-15:
            raise ValueError(f"Zero diagonal entry in lower triangular system at index {j}")
        x[j] = x[j] / L[j, j]
        for i in range(j + 1, n):
            x[i] = x[i] - L[i, j] * x[j]

    return x





def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    R, info = cholesky_factorize(A)
    if info != 0:

        eps = 1e-10 * np.max(np.diag(A))
        R, info2 = cholesky_factorize(A + eps * np.eye(A.shape[0]))
        if info2 != 0:
            raise np.linalg.LinAlgError(f"Matrix is not positive definite (info={info})")



    y = solve_lower_triangular(R.T, b)

    x = solve_upper_triangular(R, y)
    return x





def cgs_squared(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    n = A.shape[0]
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy().astype(float)

    normb = float(np.linalg.norm(b))
    if normb == 0.0:
        normb = 1.0

    r = b - A @ x
    resid = float(np.linalg.norm(r)) / normb
    if resid <= tol:
        return x

    rtilde = r.copy()
    rho = 0.0
    u = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    q = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        rho_old = rho
        rho = float(rtilde @ r)

        if abs(rho) < 1e-30:
            break

        if _ == 0:
            u = r.copy()
            p = u.copy()
        else:
            beta = rho / rho_old
            u = r + beta * q
            p = u + beta * (q + beta * p)

        phat = p.copy()
        vhat = A @ phat
        denom = float(rtilde @ vhat)
        if abs(denom) < 1e-30:
            break

        alpha = rho / denom
        q = u - alpha * vhat
        uhat = u + q
        x = x + alpha * uhat
        qhat = A @ uhat
        r = r - alpha * qhat

        resid = float(np.linalg.norm(r)) / normb
        if resid < tol:
            break

    return x





def solve_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "auto",
    tol: float = 1e-10,
) -> np.ndarray:
    n = A.shape[0]

    if method == "auto":
        if n <= 100:
            method = "cholesky"
        else:
            method = "cgs"

    if method == "cholesky":
        try:
            return cholesky_solve(A, b)
        except np.linalg.LinAlgError:
            return cgs_squared(A, b, tol=tol)
    elif method == "cgs":
        return cgs_squared(A, b, tol=tol)
    else:
        raise ValueError(f"Unknown solver method: {method}")
