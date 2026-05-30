
import numpy as np
from typing import List, Tuple, Optional


class SparseMatrix:

    def __init__(self, n: int):
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []

    def add(self, i: int, j: int, v: float):
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError("index out of bounds")
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(v)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def mv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[i] += v * x[j]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[j] += v * x[i]
        return y

    def diagonal(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            if i == j:
                d[i] += v
        return d

    def residual(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        return b - self.mv(x)


def conjugate_gradient(A: SparseMatrix, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       max_iter: Optional[int] = None) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64)
    if max_iter is None:
        max_iter = n

    r = A.residual(x, b)
    p = r.copy()
    rsold = float(np.dot(r, r))

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-15:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(np.dot(r, r))
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x


def jacobi_sparse_solve(A: SparseMatrix, b: np.ndarray,
                        x0: Optional[np.ndarray] = None,
                        tol: float = 1e-10,
                        max_iter: int = 10000) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64)

    d = A.diagonal()
    if np.any(np.abs(d) < 1e-15):
        raise ValueError("Jacobi iteration requires nonzero diagonal elements")

    for _ in range(max_iter):
        r = A.residual(x, b)
        dx = r / d
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x
