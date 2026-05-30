
import numpy as np
from system_utils import EPS, TOL_RANK, MAX_ITER, check_finite






def r83_mv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    x = np.asarray(x)
    n = A.shape[1]
    if x.shape[0] != n:
        raise ValueError("Dimension mismatch.")
    y = A[1] * x
    if n > 1:
        y[:-1] += A[2, :-1] * x[1:]
        y[1:] += A[0, 1:] * x[:-1]
    return y


def r83_mtv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    x = np.asarray(x)
    n = A.shape[1]
    y = A[1] * x
    if n > 1:
        y[:-1] += A[0, 1:] * x[1:]
        y[1:] += A[2, :-1] * x[:-1]
    return y


def r83_dif2(n: int) -> np.ndarray:
    A = np.zeros((3, n), dtype=float)
    A[1, :] = 2.0
    if n > 1:
        A[0, 1:] = -1.0
        A[2, :-1] = -1.0
    return A


def r83_res(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - r83_mv(A, x)






def r83_cg(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
           tol: float = 1e-10, max_iter: int = None) -> np.ndarray:







    raise NotImplementedError("Hole 2: R83 CG solver 待实现")






def r83_cr_fa(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    fac = [A.copy()]
    m = n
    while m > 1:
        m_prev = m
        m = m // 2
        if m < 1:
            break
        prev = fac[-1]
        nxt = np.zeros((3, m), dtype=float)
        for i in range(m):
            i2 = 2 * i + 1

            diag = prev[1, i2]
            if i2 > 0:
                diag -= prev[0, i2] * prev[2, i2 - 1] / (prev[1, i2 - 1] + EPS)
            if i2 + 1 < m_prev:
                diag -= prev[2, i2] * prev[0, i2 + 1] / (prev[1, i2 + 1] + EPS)
            nxt[1, i] = diag
            if i > 0:
                nxt[0, i] = -prev[0, i2] * prev[0, i2 - 1] / (prev[1, i2 - 1] + EPS)
            if i + 1 < m:
                nxt[2, i] = -prev[2, i2] * prev[2, i2 + 1] / (prev[1, i2 + 1] + EPS)
        fac.append(nxt)
    return fac


def r83_cr_sl(fac: list, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    n = fac[0].shape[1]
    x = b.copy()

    levels = len(fac)
    rhs = [x]
    for lev in range(1, levels):
        m = fac[lev].shape[1]
        prev_rhs = rhs[-1]
        new_rhs = np.zeros(m, dtype=float)
        for i in range(m):
            i2 = 2 * i + 1
            val = prev_rhs[i2]
            if i2 > 0:
                val -= fac[lev - 1][0, i2] * prev_rhs[i2 - 1] / (fac[lev - 1][1, i2 - 1] + EPS)
            if i2 + 1 < len(prev_rhs):
                val -= fac[lev - 1][2, i2] * prev_rhs[i2 + 1] / (fac[lev - 1][1, i2 + 1] + EPS)
            new_rhs[i] = val
        rhs.append(new_rhs)

    x_coarse = rhs[-1] / (fac[-1][1, :] + EPS)

    sol = [x_coarse]
    for lev in range(levels - 2, -1, -1):
        m = fac[lev].shape[1]
        fine = np.zeros(m, dtype=float)
        coarse = sol[-1]
        for i in range(m):
            if i % 2 == 1:
                fine[i] = coarse[i // 2]
            else:
                fine[i] = rhs[lev][i]
                if i > 0:
                    fine[i] -= fac[lev][0, i] * fine[i - 1]
                if i + 1 < m:
                    fine[i] -= fac[lev][2, i] * fine[i + 1]
                fine[i] /= (fac[lev][1, i] + EPS)
        sol.append(fine)
    return sol[-1]






def r83_jac_sl(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
               tol: float = 1e-10, max_iter: int = None) -> np.ndarray:
    A = np.asarray(A)
    b = np.asarray(b)
    n = A.shape[1]
    if max_iter is None:
        max_iter = min(MAX_ITER, 10 * n)
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    diag = A[1, :].copy()
    diag = np.where(np.abs(diag) < EPS, EPS, diag)
    for _ in range(max_iter):
        x_new = b.copy()
        if n > 1:
            x_new[:-1] -= A[2, :-1] * x[1:]
            x_new[1:] -= A[0, 1:] * x[:-1]
        x_new /= diag
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x_new):
            return x_new
        x = x_new
    return x


def r83_gs_sl(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
              tol: float = 1e-10, max_iter: int = None) -> np.ndarray:
    A = np.asarray(A)
    b = np.asarray(b)
    n = A.shape[1]
    if max_iter is None:
        max_iter = min(MAX_ITER, 10 * n)
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    diag = A[1, :].copy()
    diag = np.where(np.abs(diag) < EPS, EPS, diag)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A[0, i] * x[i - 1]
            if i < n - 1:
                sigma += A[2, i] * x[i + 1]
            x[i] = (b[i] - sigma) / diag[i]
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x):
            return x
    return x
