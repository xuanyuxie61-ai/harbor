
import numpy as np
from typing import Optional


def r83_mv(m: int, n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).ravel()

    if a.shape != (3, n):
        raise ValueError(f"R83 矩阵形状必须是 (3, {n})，实际为 {a.shape}")
    if len(x) != n:
        raise ValueError(f"向量维度 {len(x)} 与矩阵列数 {n} 不匹配")

    b = np.zeros(m, dtype=float)
    for j in range(n):
        i_start = max(0, j - 1)
        i_end = min(m, j + 2)
        for i in range(i_start, i_end):
            b[i] += a[i - j + 1, j] * x[j]
    return b


def r83_cg(n: int, a: np.ndarray, b: np.ndarray,
           x: Optional[np.ndarray] = None,
           max_iter: Optional[int] = None,
           tol: float = 1e-10) -> np.ndarray:
    b = np.asarray(b, dtype=float).ravel()
    if len(b) != n:
        raise ValueError(f"右端项维度 {len(b)} 与 n={n} 不匹配")

    if x is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel().copy()
        if len(x) != n:
            raise ValueError("初始猜测维度不匹配")

    if max_iter is None:
        max_iter = n


    ap = r83_mv(n, n, a, x)
    r = b - ap
    p = r.copy()

    rs_old = float(r @ r)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        b_norm = 1.0

    for it in range(max_iter):
        ap = r83_mv(n, n, a, p)
        pap = float(p @ ap)

        if abs(pap) < 1e-20:
            break

        alpha = rs_old / pap
        x += alpha * p
        r -= alpha * ap

        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol * b_norm:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def construct_tridiagonal_from_dense(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")

    n = A.shape[0]
    a_r83 = np.zeros((3, n), dtype=float)

    for j in range(n):
        if j > 0:
            a_r83[2, j - 1] = A[j, j - 1]
        a_r83[1, j] = A[j, j]
        if j < n - 1:
            a_r83[0, j + 1] = A[j, j + 1]

    return a_r83


def solve_normal_equations_cg(A: np.ndarray, y: np.ndarray,
                              lambda_reg: float = 1e-6,
                              max_iter: Optional[int] = None,
                              tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape


    H = A.T @ A
    H += lambda_reg * np.eye(N)


    H_tri = np.zeros_like(H)
    for i in range(N):
        H_tri[i, i] = H[i, i]
        if i > 0:
            H_tri[i, i - 1] = H[i, i - 1]
            H_tri[i - 1, i] = H[i - 1, i]

    a_r83 = construct_tridiagonal_from_dense(H_tri)
    b = A.T @ y

    x = r83_cg(N, a_r83, b, max_iter=max_iter, tol=tol)
    return x
