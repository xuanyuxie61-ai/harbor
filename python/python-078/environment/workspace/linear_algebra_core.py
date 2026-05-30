
import numpy as np
from typing import Tuple, Optional






class R83TMatrix:
    def __init__(self, data: np.ndarray, n: int):
        if data.shape != (n, 3):
            raise ValueError(f"R83T data shape {data.shape} incompatible with n={n}")
        self.data = data.astype(float)
        self.n = n

    @classmethod
    def from_diagonals(cls, sub: np.ndarray, main: np.ndarray, super: np.ndarray):
        n = len(main)
        if len(sub) != n or len(super) != n:
            raise ValueError("Diagonals must have same length")
        data = np.column_stack([sub, main, super])
        return cls(data, n)

    @classmethod
    def dif2(cls, n: int) -> 'R83TMatrix':
        sub = np.full(n, -1.0)
        sub[0] = 0.0
        main = np.full(n, 2.0)
        super = np.full(n, -1.0)
        super[-1] = 0.0
        data = np.column_stack([sub, main, super])
        return cls(data, n)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i > 0:
                A[i, i - 1] = self.data[i, 0]
            A[i, i] = self.data[i, 1]
            if i < self.n - 1:
                A[i, i + 1] = self.data[i, 2]
        return A

    def eigenvalue_dif2(self, i: int) -> float:
        if i < 1 or i > self.n:
            raise ValueError("Index out of range")
        return 4.0 * np.sin(i * np.pi / (2.0 * self.n + 2.0)) ** 2


def r83t_mv(A: R83TMatrix, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != A.n:
        raise ValueError("Dimension mismatch")
    y = np.zeros(A.n)
    for i in range(A.n):
        if i > 0:
            y[i] += A.data[i, 0] * x[i - 1]
        y[i] += A.data[i, 1] * x[i]
        if i < A.n - 1:
            y[i] += A.data[i, 2] * x[i + 1]
    return y


def r83t_mtv(A: R83TMatrix, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != A.n:
        raise ValueError("Dimension mismatch")
    y = np.zeros(A.n)
    for i in range(A.n):
        if i > 0:
            y[i - 1] += A.data[i, 0] * x[i]
        y[i] += A.data[i, 1] * x[i]
        if i < A.n - 1:
            y[i + 1] += A.data[i, 2] * x[i]
    return y


def r83t_res(A: R83TMatrix, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - r83t_mv(A, x)






def r83t_jacobi_solve(A: R83TMatrix, b: np.ndarray,
                      x0: Optional[np.ndarray] = None,
                      max_iter: int = 10000,
                      tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")


    diag = A.data[:, 1].copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal element detected")

    x_new = np.zeros(n)
    for it in range(max_iter):
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A.data[i, 0] * x[i - 1]
            if i < n - 1:
                sigma += A.data[i, 2] * x[i + 1]
            x_new[i] = (b[i] - sigma) / diag[i]

        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x[:] = x_new[:]

        if diff < tol:
            r = r83t_res(A, x, b)
            return x, it + 1, float(np.linalg.norm(r))

    r = r83t_res(A, x, b)
    return x, max_iter, float(np.linalg.norm(r))


def r83t_gauss_seidel_solve(A: R83TMatrix, b: np.ndarray,
                            x0: Optional[np.ndarray] = None,
                            max_iter: int = 10000,
                            tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")

    diag = A.data[:, 1].copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal element detected")

    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A.data[i, 0] * x[i - 1]
            if i < n - 1:
                sigma += A.data[i, 2] * x[i + 1]
            x[i] = (b[i] - sigma) / diag[i]

        diff = np.linalg.norm(x - x_old, ord=np.inf)
        if diff < tol:
            r = r83t_res(A, x, b)
            return x, it + 1, float(np.linalg.norm(r))

    r = r83t_res(A, x, b)
    return x, max_iter, float(np.linalg.norm(r))


def r83t_cg_solve(A: R83TMatrix, b: np.ndarray,
                  x0: Optional[np.ndarray] = None,
                  max_iter: int = None,
                  tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")

    r = b - r83t_mv(A, x)
    p = r.copy()
    rs_old = float(np.dot(r, r))


    if rs_old < tol * tol:
        return x, 0, float(np.sqrt(rs_old))

    for it in range(max_iter):
        Ap = r83t_mv(A, p)
        pAp = float(np.dot(p, Ap))

        if abs(pAp) < 1e-15:
            raise RuntimeError("CG breakdown: p^T A p ≈ 0")

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))

        if np.sqrt(rs_new) < tol:
            return x, it + 1, float(np.sqrt(rs_new))

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, float(np.sqrt(rs_new))






def build_womersley_tridiagonal(n_r: int, alpha_w: float,
                                dt: float, dr: float,
                                kinematic_viscosity: float) -> R83TMatrix:








    raise NotImplementedError("Hole 1: build_womersley_tridiagonal 待实现")


def thomas_algorithm(A: R83TMatrix, b: np.ndarray) -> np.ndarray:
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b_vec) != n:
        raise ValueError("Dimension mismatch")

    a = A.data[:, 0].copy()
    bb = A.data[:, 1].copy()
    c = A.data[:, 2].copy()
    d = b_vec.copy()


    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / (bb[0] + 1e-15)
    dp[0] = d[0] / (bb[0] + 1e-15)

    for i in range(1, n):
        denom = bb[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom


    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x
