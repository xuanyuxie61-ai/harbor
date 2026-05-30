
import numpy as np
from math import sqrt
from typing import Tuple






def rref_compute(A: np.ndarray, tol: float = 1.0e-12) -> Tuple[np.ndarray, list]:
    A = A.astype(float).copy()
    m, n = A.shape
    pivot_cols = []
    row = 0
    for col in range(n):

        pivot_val = 0.0
        pivot_row = -1
        for r in range(row, m):
            if abs(A[r, col]) > pivot_val:
                pivot_val = abs(A[r, col])
                pivot_row = r
        if pivot_val <= tol:
            continue

        if pivot_row != row:
            A[[pivot_row, row], :] = A[[row, pivot_row], :]

        A[row, :] = A[row, :] / A[row, col]

        for r in range(m):
            if r != row and abs(A[r, col]) > tol:
                A[r, :] = A[r, :] - A[r, col] * A[row, :]
        pivot_cols.append(col)
        row += 1
        if row >= m:
            break
    return A, pivot_cols


def rref_solve(A: np.ndarray, b: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    A = np.atleast_2d(A).astype(float)
    b = np.asarray(b, dtype=float)
    m1, n1 = A.shape
    if b.ndim == 1:
        if len(b) == m1:
            b = b.reshape(-1, 1)
        else:
            b = b.reshape(1, -1)
    m2, n2 = b.shape
    if m1 != m2:
        raise ValueError(f"rref_solve: A has {m1} rows but b has {m2} rows")
    AI = np.hstack([A, b])
    AIRREF, pivot_cols = rref_compute(AI, tol=tol)
    x = AIRREF[:n1, n1:n1 + n2]
    return x


def rref_rank(A: np.ndarray, tol: float = 1.0e-12) -> int:
    _, pivot_cols = rref_compute(A, tol=tol)
    return len(pivot_cols)






def r83_np_fa(n: int, a: np.ndarray) -> np.ndarray:
    if n < 2:
        raise ValueError("r83_np_fa: n must be at least 2")
    a_lu = a.copy()
    for i in range(1, n):
        a_lu[2, i] = a_lu[2, i] / a_lu[1, i - 1]
        a_lu[1, i] = a_lu[1, i] - a_lu[2, i] * a_lu[0, i - 1]
    return a_lu


def r83_np_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int = 0) -> np.ndarray:
    x = b.copy().astype(float)
    if job == 0:

        for i in range(1, n):
            x[i] = x[i] - a_lu[2, i] * x[i - 1]

        for i in range(n - 1, -1, -1):
            x[i] = x[i] / a_lu[1, i]
            if i > 0:
                x[i - 1] = x[i - 1] - a_lu[0, i - 1] * x[i]
    else:

        for i in range(n):
            x[i] = x[i] / a_lu[1, i]
            if i < n - 1:
                x[i + 1] = x[i + 1] - a_lu[0, i] * x[i]
        for i in range(n - 1, 0, -1):
            x[i - 1] = x[i - 1] - a_lu[2, i] * x[i]
    return x


def r83p_fa(n: int, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if n < 3:
        raise ValueError("r83p_fa: n must be at least 3")
    a_lu = a.copy()

    a2 = np.zeros(n - 1)
    a3 = np.zeros(n - 1)
    a2[0] = a[0, 0]
    a2[n - 2] = a[2, n - 1]
    a3[0] = a[2, 0]
    a3[n - 2] = a[0, n - 2]


    a_lu_part = r83_np_fa(n - 1, a_lu[:, :n - 1])
    a_lu[:, :n - 1] = a_lu_part


    work2 = np.zeros(n - 1)
    work2[0] = a[0, 0]
    work2[n - 2] = a[2, n - 1]
    work2 = r83_np_sl(n - 1, a_lu[:, :n - 1], work2, job=0)


    work3 = np.zeros(n - 1)
    work3[0] = a[2, 0]
    work3[n - 2] = a[0, n - 2]
    work3 = r83_np_sl(n - 1, a_lu[:, :n - 1], work3, job=0)


    work4 = a[1, n - 1] - a[0, n - 1] * work2[n - 2] - a[2, n - 2] * work3[n - 2]
    if abs(work4) < 1.0e-15:
        raise ValueError("r83p_fa: singular matrix")

    return a_lu, work2, work3, work4


def r83p_sl(n: int, a_lu: np.ndarray, b: np.ndarray,
            job: int, work2: np.ndarray, work3: np.ndarray, work4: float) -> np.ndarray:
    x = b.copy().astype(float)
    if job == 0:
        x[:n - 1] = r83_np_sl(n - 1, a_lu[:, :n - 1], x[:n - 1], job=0)
        x[n - 1] = x[n - 1] - a_lu[0, 0] * x[0] - a_lu[2, n - 2] * x[n - 2]
        x[n - 1] = x[n - 1] / work4
        x[:n - 1] = x[:n - 1] - work2 * x[n - 1]
    else:
        x[:n - 1] = r83_np_sl(n - 1, a_lu[:, :n - 1], x[:n - 1], job=1)
        x[n - 1] = x[n - 1] - a_lu[2, n - 1] * x[0] - a_lu[0, n - 1] * x[n - 2]
        x[n - 1] = x[n - 1] / work4
        x[:n - 1] = x[:n - 1] - work3 * x[n - 1]
    return x


def r83p_solve(n: int, a: np.ndarray, b: np.ndarray, job: int = 0) -> np.ndarray:
    a_lu, work2, work3, work4 = r83p_fa(n, a)
    return r83p_sl(n, a_lu, b, job, work2, work3, work4)






def toeplitz_cholesky_lower(n: int, a: np.ndarray) -> np.ndarray:
    if n < 1:
        raise ValueError("toeplitz_cholesky_lower: n must be positive")
    a = np.atleast_2d(a).astype(float)
    if a.shape != (n, n):
        raise ValueError("toeplitz_cholesky_lower: a shape mismatch")

    for i in range(n):
        if a[i, i] <= 0:
            raise ValueError(f"toeplitz_cholesky_lower: non-positive diagonal at {i}")



    scale = np.sqrt(a[0, 0])
    if scale < 1.0e-12:
        raise ValueError("toeplitz_cholesky_lower: zero or near-zero diagonal")
    g = np.zeros((2, n))
    g[0, :] = a[:, 0] / scale
    g[1, 0] = 0.0
    g[1, 1:n] = a[1:n, 0] / scale

    L = np.zeros((n, n))
    L[:, 0] = g[0, :]
    g[0, 1:n] = g[0, 0:n - 1]
    g[0, 0] = 0.0

    for i in range(1, n):
        rho = -g[1, i] / g[0, i]
        denom = sqrt((1.0 - rho) * (1.0 + rho))
        if abs(denom) < 1.0e-15:
            raise ValueError("toeplitz_cholesky_lower: breakdown at step {}".format(i))
        A_mat = np.array([[1.0, rho], [rho, 1.0]])
        g[:, i:n] = (A_mat @ g[:, i:n]) / denom
        L[i:n, i] = g[0, i:n]
        if i + 1 < n:
            g[0, i + 1:n] = g[0, i:n - 1]
        g[0, i] = 0.0
    return L


def sample_from_toeplitz_covariance(n: int, first_col: np.ndarray) -> np.ndarray:
    first_col = np.asarray(first_col, dtype=float)
    if len(first_col) < n:
        raise ValueError("sample_from_toeplitz_covariance: first_col too short")
    first_col = first_col[:n]
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = first_col[abs(i - j)]

    eigvals = np.linalg.eigvalsh(T)
    if np.min(eigvals) <= 0:
        T = T + (-np.min(eigvals) + 1.0e-10) * np.eye(n)
    L = toeplitz_cholesky_lower(n, T)
    z = np.random.randn(n)
    return L @ z
