
import numpy as np
from typing import Tuple, List


class SparseMatrixCOO:

    def __init__(self, nrow: int, ncol: int, nnz: int = 0):
        self.nrow = nrow
        self.ncol = ncol
        self.row = []
        self.col = []
        self.val = []
        self._nnz_max = nnz if nnz > 0 else nrow * ncol

    def append(self, i: int, j: int, v: float):
        if abs(v) > 1e-20:
            self.row.append(i)
            self.col.append(j)
            self.val.append(v)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.nrow, self.ncol))
        for i, j, v in zip(self.row, self.col, self.val):
            A[i, j] += v
        return A

    def mv(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.nrow, dtype=float)
        for i, j, v in zip(self.row, self.col, self.val):
            y[i] += v * x[j]
        return y

    def check(self) -> bool:
        for i, j in zip(self.row, self.col):
            if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
                return False
        return True


def build_anelastic_laplacian_2d(nx: int, nz: int, dx: float, dz: float,
                                 rho: np.ndarray) -> SparseMatrixCOO:
    n = nx * nz
    A = SparseMatrixCOO(n, n, nnz=n * 5)

    def idx(i, j):
        return j * nx + i

    dx2 = dx * dx
    dz2 = dz * dz

    for j in range(nz):
        for i in range(nx):
            k = idx(i, j)

            rho_e = 0.5 * (rho[j, min(i+1, nx-1)] + rho[j, i])
            rho_w = 0.5 * (rho[j, max(i-1, 0)] + rho[j, i])
            rho_n = 0.5 * (rho[min(j+1, nz-1), i] + rho[j, i])
            rho_s = 0.5 * (rho[max(j-1, 0), i] + rho[j, i])

            coeff = 0.0

            if i > 0:
                A.append(k, idx(i-1, j), rho_w / dx2)
                coeff -= rho_w / dx2
            if i < nx - 1:
                A.append(k, idx(i+1, j), rho_e / dx2)
                coeff -= rho_e / dx2

            if j > 0:
                A.append(k, idx(i, j-1), rho_s / dz2)
                coeff -= rho_s / dz2
            if j < nz - 1:
                A.append(k, idx(i, j+1), rho_n / dz2)
                coeff -= rho_n / dz2


            A.append(k, k, coeff)

    return A


def conjugate_gradient(A: SparseMatrixCOO, b: np.ndarray, x0: np.ndarray = None,
                       tol: float = 1e-8, max_iter: int = None) -> Tuple[np.ndarray, int, float]:
    n = A.nrow
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)


    if rs_old < 1e-30:
        return x, 0, 0.0

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol * (np.linalg.norm(b) + 1.0):
            return x, k + 1, np.sqrt(rs_new)
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, np.sqrt(rs_old)


def jacobi_iteration(A: SparseMatrixCOO, b: np.ndarray, x0: np.ndarray = None,
                     tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    n = A.nrow
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()


    diag = np.zeros(n)
    for i, j, v in zip(A.row, A.col, A.val):
        if i == j:
            diag[i] = v

    diag = np.where(np.abs(diag) < 1e-20, 1.0, diag)

    for _ in range(max_iter):
        x_new = (b - A.mv(x) + diag * x) / diag
        if np.linalg.norm(x_new - x) < tol * (np.linalg.norm(x_new) + 1.0):
            return x_new
        x = x_new
    return x


def solve_anelastic_pressure(nx: int, nz: int, dx: float, dz: float,
                             rho: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    A = build_anelastic_laplacian_2d(nx, nz, dx, dz, rho)
    b = rhs.flatten()


    b -= np.mean(b)

    x0 = np.zeros_like(b)
    x, iters, res = conjugate_gradient(A, b, x0, tol=1e-10, max_iter=min(5000, A.nrow))


    if res > 1e-4:
        x = jacobi_iteration(A, b, x, tol=1e-8, max_iter=2000)

    p = x.reshape((nz, nx))

    p -= np.mean(p)
    return p
