
import numpy as np
from typing import Tuple, Optional


class SkylineMatrixOperator:

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("矩阵阶数必须至少为 1")
        self.n = n
        self.diag = None
        self.a = None
        self.na = 0

    def build_from_tridiagonal(
        self, lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray
    ):
        n = self.n
        if len(diagonal) != n:
            raise ValueError("对角线长度不匹配")

        diag_idx = np.zeros(n, dtype=int)
        count = 0
        for j in range(n):

            height = 1
            if j > 0 and lower[j] != 0.0:
                height = 2
            count += height
            diag_idx[j] = count - 1
        self.diag = diag_idx
        self.na = count
        self.a = np.zeros(count)
        k = 0
        for j in range(n):
            if j > 0 and lower[j] != 0.0:
                self.a[k] = lower[j]
                k += 1
            self.a[k] = diagonal[j]
            k += 1

    def build_from_dense(self, A: np.ndarray, tol: float = 1.0e-14):
        if A.shape != (self.n, self.n):
            raise ValueError("矩阵维度不匹配")
        n = self.n
        diag_idx = np.zeros(n, dtype=int)
        count = 0
        for j in range(n):

            rows = np.where(np.abs(A[:, j]) > tol)[0]
            if len(rows) == 0:
                first_row = j
            else:
                first_row = min(rows)
            first_row = min(first_row, j)
            height = j - first_row + 1
            count += height
            diag_idx[j] = count - 1
        self.diag = diag_idx
        self.na = count
        self.a = np.zeros(count)
        k = 0
        for j in range(n):
            first_row = j - (diag_idx[j] - (diag_idx[j - 1] if j > 0 else -1)) + 1
            for i in range(first_row, j + 1):
                self.a[k] = A[i, j]
                k += 1

    def multiply(self, x: np.ndarray) -> np.ndarray:
        if len(x) != self.n:
            raise ValueError("向量维度不匹配")
        if self.a is None or self.diag is None:
            raise ValueError("矩阵未初始化")

        n = self.n
        y = np.zeros(n)
        diagold = -1
        k = 0
        for j in range(n):
            height = self.diag[j] - diagold
            ilo = j + 1 - height + 1
            for i in range(ilo, j):
                y[i] += self.a[k] * x[j]
                y[j] += self.a[k] * x[i]
                k += 1
            y[j] += self.a[k] * x[j]
            k += 1
            diagold = self.diag[j]
        return y

    def to_dense(self) -> np.ndarray:
        n = self.n
        A = np.zeros((n, n))
        if self.a is None:
            return A
        diagold = -1
        k = 0
        for j in range(n):
            height = self.diag[j] - diagold
            ilo = j + 1 - height + 1
            for i in range(ilo, j):
                A[i, j] = self.a[k]
                A[j, i] = self.a[k]
                k += 1
            A[j, j] = self.a[k]
            k += 1
            diagold = self.diag[j]
        return A

    def solve_cholesky_skyline(
        self, b: np.ndarray
    ) -> np.ndarray:
        A_dense = self.to_dense()
        try:
            L = np.linalg.cholesky(A_dense)
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x
        except np.linalg.LinAlgError:

            return np.linalg.solve(A_dense, b)

    def condition_number_estimate(self) -> float:
        A_dense = self.to_dense()
        s = np.linalg.svd(A_dense, compute_uv=False)
        if s[-1] < 1.0e-15:
            return float("inf")
        return s[0] / s[-1]
