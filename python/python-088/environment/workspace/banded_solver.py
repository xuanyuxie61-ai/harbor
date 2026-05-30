
import numpy as np
from typing import Tuple, Optional


def thomas_algorithm(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, b: np.ndarray
) -> np.ndarray:
    n = len(diag)
    if len(lower) != n - 1 or len(upper) != n - 1 or len(b) != n:
        raise ValueError("Dimension mismatch in tridiagonal system")


    d = diag.astype(float).copy()
    c = upper.astype(float).copy()
    l = lower.astype(float).copy()
    rhs = b.astype(float).copy()


    for i in range(1, n):
        w = l[i - 1] / d[i - 1]
        d[i] = d[i] - w * c[i - 1]
        rhs[i] = rhs[i] - w * rhs[i - 1]


    x = np.zeros(n)
    x[-1] = rhs[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - c[i] * x[i + 1]) / d[i]

    return x


def r83_cr_factor(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    n = len(diag)
    if n < 2:
        raise ValueError("Matrix order must be at least 2")


    a_cr = np.zeros((3, n))
    a_cr[0, :n - 1] = upper[:n - 1]
    a_cr[1, :] = diag[:]
    a_cr[2, 1:n] = lower[:n - 1]



    n_levels = int(np.ceil(np.log2(n)))

    for level in range(n_levels):
        step = 2 ** level
        for i in range(2 * step - 1, n, 2 * step):
            if i - step >= 0 and i + step < n:

                pivot = a_cr[1, i - step]
                if abs(pivot) < 1e-15:
                    pivot = 1e-15
                factor_lower = a_cr[2, i] / pivot
                factor_upper = a_cr[0, i - step] / pivot

                a_cr[1, i] -= factor_lower * a_cr[0, i - step]
                a_cr[2, i] = -factor_lower * a_cr[2, i - step] if i - 2 * step >= 0 else 0.0
                if i + step < n:
                    a_cr[0, i] -= factor_upper * a_cr[0, i]

    return a_cr


def r83_cr_solve(
    a_cr: np.ndarray, b: np.ndarray
) -> np.ndarray:
    n = a_cr.shape[1]
    x = b.astype(float).copy()




    lower = np.zeros(n - 1)
    diag = a_cr[1, :].copy()
    upper = np.zeros(n - 1)
    upper[:n - 1] = a_cr[0, :n - 1]
    lower[1:n - 1] = a_cr[2, 2:n]
    if n > 1:
        lower[0] = a_cr[2, 1]

    return thomas_algorithm(lower, diag, upper, x)


def conjugate_gradient_band(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
    b: np.ndarray, x0: Optional[np.ndarray] = None,
    tol: float = 1e-10, max_iter: int = None
) -> np.ndarray:
    n = len(diag)
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()


    def matvec(v):
        Av = diag * v
        if n > 1:
            Av[:-1] += upper * v[1:]
            Av[1:] += lower * v[:-1]
        return Av

    r = b - matvec(x)
    p = r.copy()
    rs_old = np.dot(r, r)

    for _ in range(max_iter):
        Ap = matvec(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def gauss_seidel_band(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
    b: np.ndarray, x0: Optional[np.ndarray] = None,
    tol: float = 1e-10, max_iter: int = 10000
) -> np.ndarray:
    n = len(diag)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += lower[i - 1] * x[i - 1]
            if i < n - 1:
                sigma += upper[i] * x_old[i + 1]
            if abs(diag[i]) < 1e-15:
                x[i] = 0.0
            else:
                x[i] = (b[i] - sigma) / diag[i]

        if np.linalg.norm(x - x_old) < tol:
            break

    return x


def banded_lower_triangular_solve(
    A_band: np.ndarray, b: np.ndarray, ml: int
) -> np.ndarray:
    n = len(b)
    x = b.astype(float).copy()

    for j in range(n):
        if abs(A_band[0, j]) < 1e-15:
            x[j] = 0.0
        else:
            x[j] = x[j] / A_band[0, j]
        ihi = min(j + ml, n - 1)
        for i in range(j + 1, ihi + 1):
            band_idx = i - j
            if band_idx < A_band.shape[0]:
                x[i] = x[i] - A_band[band_idx, j] * x[j]

    return x


def jacobi_iteration_band(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
    b: np.ndarray, x0: Optional[np.ndarray] = None,
    tol: float = 1e-10, max_iter: int = 10000
) -> np.ndarray:
    n = len(diag)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += lower[i - 1] * x[i - 1]
            if i < n - 1:
                sigma += upper[i] * x[i + 1]
            if abs(diag[i]) < 1e-15:
                x_new[i] = 0.0
            else:
                x_new[i] = (b[i] - sigma) / diag[i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    return x


def solve_sparse_symmetric_positive_definite(
    A: np.ndarray, b: np.ndarray, method: str = "auto"
) -> np.ndarray:
    n = A.shape[0]


    if n > 1:
        diag = np.diag(A)
        upper = np.diag(A, k=1)
        lower = np.diag(A, k=-1)
        off_diag_sum = np.sum(np.abs(A)) - np.sum(np.abs(diag))
        tri_diag_sum = np.sum(np.abs(upper)) + np.sum(np.abs(lower))


        if off_diag_sum < tri_diag_sum * 1.1 and method in ("auto", "direct"):
            return thomas_algorithm(lower, diag, upper, b)

    if method in ("auto", "direct"):
        return np.linalg.solve(A, b)
    elif method == "cg":
        diag = np.diag(A)
        upper = np.diag(A, k=1)
        lower = np.diag(A, k=-1)
        return conjugate_gradient_band(lower, diag, upper, b)
    elif method == "gs":
        diag = np.diag(A)
        upper = np.diag(A, k=1)
        lower = np.diag(A, k=-1)
        return gauss_seidel_band(lower, diag, upper, b)
    else:
        diag = np.diag(A)
        upper = np.diag(A, k=1)
        lower = np.diag(A, k=-1)
        return jacobi_iteration_band(lower, diag, upper, b)
