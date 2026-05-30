
import numpy as np
from typing import Tuple, Optional


def gauss_elimination_partial_pivot(A: np.ndarray, b: np.ndarray,
                                    tol: Optional[float] = None) -> np.ndarray:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须为方阵。")
    n = A.shape[0]
    b = np.atleast_2d(b).T if b.ndim == 1 else np.array(b)
    if b.shape[0] != n:
        raise ValueError("b 的行数必须与 A 的维数一致。")


    Ab = np.hstack([A.astype(np.float64), b.astype(np.float64)])
    if tol is None:
        tol = np.finfo(float).eps * n * np.max(np.abs(A))


    for col in range(n):

        pivot_row = col + np.argmax(np.abs(Ab[col:, col]))
        max_val = np.abs(Ab[pivot_row, col])
        if max_val < tol:
            raise ValueError(f"矩阵在第 {col} 列无主元，可能奇异。")
        if pivot_row != col:
            Ab[[col, pivot_row], :] = Ab[[pivot_row, col], :]


        pivot = Ab[col, col]
        Ab[col, :] /= pivot


        for row in range(col + 1, n):
            factor = Ab[row, col]
            if factor != 0.0:
                Ab[row, :] -= factor * Ab[col, :]


    x = np.zeros((n, b.shape[1]), dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i, :] = Ab[i, n:] - Ab[i, i + 1:n] @ x[i + 1:, :]

    return x.squeeze() if x.shape[1] == 1 else x


def plu_decomposition(A: np.ndarray,
                      tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须为方阵。")
    n = A.shape[0]
    if tol is None:
        tol = np.finfo(float).eps * n * np.max(np.abs(A) + 1.0)

    L = np.eye(n, dtype=np.float64)
    U = A.astype(np.float64).copy()
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < tol:
            raise ValueError(f"U_{k},{k} 接近零，矩阵奇异。")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U


def solve_plu(P: np.ndarray, L: np.ndarray, U: np.ndarray,
              b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    b = np.atleast_1d(b).astype(np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    pb = P @ b
    y = np.zeros_like(pb)
    for i in range(n):
        y[i, :] = pb[i, :] - L[i, :i] @ y[:i, :]

    x = np.zeros_like(pb)
    for i in range(n - 1, -1, -1):
        denom = U[i, i]
        if abs(denom) < 1e-15:
            raise ValueError("U 对角元接近零，无法回代。")
        x[i, :] = (y[i, :] - U[i, i + 1:] @ x[i + 1:, :]) / denom

    return x.squeeze() if x.shape[1] == 1 else x


def condition_number_estimate(A: np.ndarray) -> float:
    norm_A = np.linalg.norm(A, ord=np.inf)
    try:
        norm_Ainv = np.linalg.norm(np.linalg.inv(A), ord=np.inf)
    except np.linalg.LinAlgError:
        return np.inf
    return norm_A * norm_Ainv
