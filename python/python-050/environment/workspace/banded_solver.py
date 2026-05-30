
import numpy as np
from typing import Tuple


def solve_tridiagonal(a: np.ndarray,
                      b: np.ndarray,
                      c: np.ndarray,
                      d: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = len(b)

    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("All input arrays must have the same length.")


    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-20:
            denom = 1e-20 * np.sign(denom) if denom != 0 else 1e-20
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom


    u = np.zeros(n, dtype=np.float64)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    return u


def solve_pentadiagonal(p: np.ndarray, q: np.ndarray,
                        r: np.ndarray, s: np.ndarray,
                        t: np.ndarray, d: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = len(r)



    if n > 5000:

        try:
            from scipy.sparse import diags
            from scipy.sparse.linalg import spsolve
            A = diags([p[2:], q[1:], r, s[:-1], t[:-2]],
                      [-2, -1, 0, 1, 2], format='csc')
            return spsolve(A, d)
        except ImportError:
            pass

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = r[i]
        if i > 0:
            A[i, i - 1] = q[i]
        if i > 1:
            A[i, i - 2] = p[i]
        if i < n - 1:
            A[i, i + 1] = s[i]
        if i < n - 2:
            A[i, i + 2] = t[i]

    return np.linalg.solve(A, d)


def banded_lu_solve(ab: np.ndarray, kl: int, ku: int, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    ab = np.asarray(ab, dtype=np.float64)

    if ab.shape[0] != kl + ku + 1:
        raise ValueError("ab must have shape (kl+ku+1, n)")

    n = ab.shape[1]

    try:
        from scipy.linalg.lapack import dgbsv

        ab_lapack = np.zeros((2 * kl + ku + 1, n), dtype=np.float64)
        for j in range(n):
            for i in range(kl + ku + 1):
                row_in_lapack = i + kl
                col = j + i - ku
                if 0 <= col < n:
                    ab_lapack[row_in_lapack, col] = ab[i, col]

        if b.ndim == 1:
            b = b.reshape(-1, 1)
        _, _, x, info = dgbsv(kl, ku, ab_lapack, b)
        if info != 0:
            raise RuntimeError(f"dgbsv failed with info={info}")
        return x.ravel() if x.shape[1] == 1 else x
    except ImportError:

        A = banded_to_dense(ab, kl, ku, n)
        return np.linalg.solve(A, b)


def banded_to_dense(ab: np.ndarray, kl: int, ku: int, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(kl + ku + 1):
            row = j + ku - i
            col = j
            if 0 <= row < n:
                A[row, col] = ab[i, col]
    return A


def dense_to_banded(A: np.ndarray, kl: int, ku: int) -> np.ndarray:
    n = A.shape[0]
    ab = np.zeros((kl + ku + 1, n), dtype=np.float64)
    for j in range(n):
        i_start = max(0, ku - j)
        i_end = min(kl + ku + 1, ku + n - j)
        for i in range(i_start, i_end):
            row = j + ku - i
            ab[i, j] = A[row, j]
    return ab


def build_sia_tridiagonal(H: np.ndarray,
                          bedrock: np.ndarray,
                          dx: float,
                          A: float,
                          rho_g: float,
                          n: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=np.float64)
    n_nodes = len(H)

    a = np.zeros(n_nodes, dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)
    c = np.zeros(n_nodes, dtype=np.float64)
    rhs = np.zeros(n_nodes, dtype=np.float64)

    surface = bedrock + H
    grad_s = np.zeros(n_nodes, dtype=np.float64)
    grad_s[1:-1] = (surface[2:] - surface[:-2]) / (2.0 * dx)
    grad_s[0] = (surface[1] - surface[0]) / dx
    grad_s[-1] = (surface[-1] - surface[-2]) / dx
    grad_s = np.abs(grad_s)
    grad_s = np.maximum(grad_s, 1e-12)








    raise NotImplementedError("Hole 3: 请实现 build_sia_tridiagonal 核心公式与矩阵组装")
