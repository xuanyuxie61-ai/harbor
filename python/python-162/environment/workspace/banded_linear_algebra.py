
import numpy as np
from typing import Tuple, Optional


class BandedMatrix:

    def __init__(self, n: int, ml: int, mu: int):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.ab = np.zeros((2 * ml + mu + 1, n), dtype=float)
        self.pivot = np.zeros(n, dtype=int)

    def set_entry(self, i: int, j: int, value: float):
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return
        if j - i > self.mu or i - j > self.ml:
            return
        row = self.ml + self.mu + i - j
        self.ab[row, j] = value

    def get_entry(self, i: int, j: int) -> float:
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return 0.0
        if j - i > self.mu or i - j > self.ml:
            return 0.0
        row = self.ml + self.mu + i - j
        return self.ab[row, j]

    def plu_factor(self) -> int:
        n = self.n
        ml = self.ml
        mu = self.mu
        m = ml + mu + 1
        info = 0
        for j in range(n):
            pivot_row = -1
            pivot_val = 0.0
            for i in range(j, min(j + ml + 1, n)):
                val = abs(self.ab[m - 1 + i - j, j])
                if val > pivot_val:
                    pivot_val = val
                    pivot_row = i
            self.pivot[j] = pivot_row
            if pivot_row == -1 or abs(self.ab[m - 1, j]) < 1e-18:
                info = j + 1
                continue
            if pivot_row != j:
                for k in range(max(0, j - mu), min(j + ml + 1, n)):
                    idx1 = m - 1 + j - k
                    idx2 = m - 1 + pivot_row - k
                    self.ab[idx1, k], self.ab[idx2, k] = self.ab[idx2, k], self.ab[idx1, k]
            for i in range(j + 1, min(j + ml + 1, n)):
                self.ab[m - 1 + i - j, j] /= self.ab[m - 1, j]
                for k in range(j + 1, min(j + mu + 1, n)):
                    self.ab[m - 1 + i - k, k] -= self.ab[m - 1 + i - j, j] * self.ab[m - 1 + j - k, k]
        return info

    def solve(self, b: np.ndarray, trans: int = 0) -> np.ndarray:
        n = self.n
        ml = self.ml
        mu = self.mu
        m = ml + mu + 1
        x = np.copy(b).astype(float)
        if trans == 0:
            for j in range(n):
                pivot_row = self.pivot[j]
                if pivot_row != j:
                    x[j], x[pivot_row] = x[pivot_row], x[j]
                for i in range(j + 1, min(j + ml + 1, n)):
                    x[i] -= self.ab[m - 1 + i - j, j] * x[j]
            for j in range(n - 1, -1, -1):
                x[j] /= self.ab[m - 1, j]
                for i in range(max(0, j - mu), j):
                    x[i] -= self.ab[m - 1 + i - j, j] * x[j]
        else:
            for j in range(n):
                x[j] /= self.ab[m - 1, j]
                for i in range(j + 1, min(j + mu + 1, n)):
                    x[i] -= self.ab[m - 1 + j - i, i] * x[j]
            for j in range(n - 1, -1, -1):
                pivot_row = self.pivot[j]
                for i in range(max(0, j - ml), j):
                    x[i] -= self.ab[m - 1 + i - j, j] * x[j]
                if pivot_row != j:
                    x[j], x[pivot_row] = x[pivot_row], x[j]
        return x

    def determinant(self) -> float:
        n = self.n
        m = self.ml + self.mu + 1
        det = 1.0
        for j in range(n):
            det *= self.ab[m - 1, j]
        swaps = sum(1 for j in range(n) if self.pivot[j] != j)
        if swaps % 2 == 1:
            det = -det
        return det


class SymmetricToeplitzSolver:

    def __init__(self, first_row: np.ndarray):
        self.first_row = np.asarray(first_row, dtype=float)
        self.n = len(first_row)

    def yule_walker(self, rhs_prefix: np.ndarray) -> np.ndarray:
        n = self.n
        r = self.first_row
        b = np.asarray(rhs_prefix, dtype=float)
        if len(b) != n:
            raise ValueError("rhs_prefix must have length n")
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)
        alpha = -r[0]
        beta = 1.0
        x[0] = b[0] / r[0]
        if n == 1:
            return x
        for k in range(1, n):
            beta = (1.0 - y[k - 1] ** 2) * beta
            if abs(beta) < 1e-18:
                raise ValueError("Toeplitz matrix singular in Durbin step")
            alpha = -(r[k] + np.dot(r[1:k][::-1], y[:k - 1])) / beta
            y[:k] = y[:k] + alpha * y[:k][::-1]
            y[k] = alpha
            x[k] = (b[k] - np.dot(r[1:k + 1][::-1], x[:k])) / (r[0] + np.dot(r[1:k + 1][::-1], y[:k]))
            x[:k] = x[:k] + x[k] * y[:k][::-1]
        return x

    def solve_general(self, b: np.ndarray) -> np.ndarray:
        n = self.n
        b_arr = np.asarray(b, dtype=float)
        if len(b_arr) != n:
            raise ValueError("b must have length n")
        r = self.first_row
        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)
        x[0] = b_arr[0] / r[0]
        if n == 1:
            return x
        y[0] = -r[1] / r[0]
        for k in range(1, n):
            denom = r[0] + np.dot(r[1:k + 1], y[:k])
            if abs(denom) < 1e-18:
                raise ValueError("Toeplitz solve denominator near zero")
            x[k] = (b_arr[k] - np.dot(r[1:k + 1][::-1], x[:k])) / denom
            x[:k] = x[:k] + x[k] * y[:k][::-1]
            if k < n - 1:
                y[k] = -(r[k + 1] + np.dot(r[1:k + 1][::-1], y[:k])) / denom
                y[:k] = y[:k] + y[k] * y[:k][::-1]
        return x

    def matvec(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        x_arr = np.asarray(x, dtype=float)
        if len(x_arr) != n:
            raise ValueError("x must have length n")
        y = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(n):
                idx = abs(i - j)
                y[i] += self.first_row[idx] * x_arr[j]
        return y


def banded_from_dense(A: np.ndarray, ml: int, mu: int) -> BandedMatrix:
    n = A.shape[0]
    bm = BandedMatrix(n, ml, mu)
    for i in range(n):
        for j in range(max(0, i - ml), min(n, i + mu + 1)):
            bm.set_entry(i, j, A[i, j])
    return bm


def build_tridiagonal_banded(n: int, lower: float, diag: float, upper: float) -> BandedMatrix:
    bm = BandedMatrix(n, 1, 1)
    for i in range(n):
        bm.set_entry(i, i, diag)
        if i > 0:
            bm.set_entry(i, i - 1, lower)
        if i < n - 1:
            bm.set_entry(i, i + 1, upper)
    return bm
