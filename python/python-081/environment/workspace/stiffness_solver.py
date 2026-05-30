
import numpy as np
from typing import Tuple, Optional






def plu_decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("PLU分解要求方阵")
    U = A.copy().astype(np.float64)
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):

        max_row = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[max_row, k]) < 1e-15:
            raise ValueError("PLU分解: 矩阵奇异或接近奇异")
        if max_row != k:
            U[[k, max_row], :] = U[[max_row, k], :]
            P[[k, max_row], :] = P[[max_row, k], :]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def solve_plu(P: np.ndarray, L: np.ndarray, U: np.ndarray,
              b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    pb = P @ b

    for i in range(n):
        s = pb[i] - np.dot(L[i, :i], y[:i])
        y[i] = s / L[i, i]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise ValueError("solve_plu: U对角元接近零")
        x[i] = s / U[i, i]
    return x






def cholesky_decompose(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diag_sum = np.dot(L[i, :i], L[i, :i])
        val = A[i, i] - diag_sum
        if val <= 1e-15:

            val = 1e-12
        L[i, i] = np.sqrt(val)
        for j in range(i + 1, n):
            off_sum = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - off_sum) / L[i, i]
    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i] - np.dot(L[i, :i], y[:i])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Cholesky求解: L对角元接近零")
        y[i] = s / L[i, i]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i] - np.dot(L[i + 1:, i], x[i + 1:])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Cholesky求解: L对角元接近零")
        x[i] = s / L[i, i]
    return x






def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                        x0: Optional[np.ndarray] = None,
                        tol: float = 1e-10, max_iter: Optional[int] = None) -> np.ndarray:
    n = A.shape[0]
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-14:
        norm_b = 1.0

    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-15:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new) / norm_b < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def apply_dirichlet_to_system(K: np.ndarray, R: np.ndarray,
                               bc_dofs: np.ndarray,
                               bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K_mod = K.copy()
    R_mod = R.copy()
    for dof, val in zip(bc_dofs, bc_values):
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0
        R_mod[dof] = val
    return K_mod, R_mod
