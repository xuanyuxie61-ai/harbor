"""
sparse_matrix.py
================
Sparse and banded matrix storage formats for subdomain-local solves
in domain-decomposition finite-element methods.

Integrates concepts from:
  * r8pbl (symmetric positive definite band storage)
  * hb_to_st (sparse matrix format conversion)
  * linpack_bench_backslash (dense/banded linear solve benchmarking)

Mathematical background
-----------------------
For a symmetric band matrix A with bandwidth ml (lower bandwidth),
the LAPACK-style band storage stores the lower triangle in a
2-D array of shape (ml+1, n):
    ab[k, j] = A[j+k, j]   for k = 0..ml, j = 0..n-1, with j+k < n

For the SPD case, the Cholesky factorization A = L L^T exists
uniquely with L lower triangular and positive diagonal entries.
The factorization cost is O(n ml^2), and solve cost is O(n ml).
"""

import numpy as np
from typing import Tuple, Optional


class BandedSPDMatrix:
    """
    Symmetric Positive Definite band matrix in LAPACK lower-band storage.
    ab[k, j] = A[j+k, j] for k = 0..ml, j = 0..n-1, with j+k < n.
    """

    def __init__(self, n: int, ml: int):
        if n <= 0 or ml < 0:
            raise ValueError("n must be positive and ml nonnegative.")
        if ml >= n:
            ml = n - 1
        self.n = n
        self.ml = ml
        # Storage: (ml+1) x n array, upper-right triangular part unused
        self.data = np.zeros((ml + 1, n), dtype=float)

    def set(self, i: int, j: int, val: float):
        """Set A[i,j] = val (symmetric). Only lower triangle stored."""
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError("Matrix index out of bounds.")
        if abs(i - j) > self.ml:
            return  # silently ignore out-of-band entries
        if i >= j:
            k = i - j
            self.data[k, j] = val
        # else: would be in upper triangle, stored symmetrically at (j,i)

    def get(self, i: int, j: int) -> float:
        """Get A[i,j]."""
        if abs(i - j) > self.ml:
            return 0.0
        if i >= j:
            k = i - j
            return self.data[k, j]
        else:
            k = j - i
            return self.data[k, i]

    def to_dense(self) -> np.ndarray:
        """Convert to dense n x n matrix (for verification only)."""
        A = np.zeros((self.n, self.n), dtype=float)
        for j in range(self.n):
            for k in range(self.ml + 1):
                i = j + k
                if i < self.n:
                    A[i, j] = self.data[k, j]
                    if i != j:
                        A[j, i] = self.data[k, j]
        return A

    def cholesky_band(self) -> 'BandedSPDMatrix':
        """
        In-place Cholesky factorization of a banded SPD matrix.
        Uses the standard banded Cholesky algorithm with O(n ml^2) complexity.

        For A = L L^T with L lower triangular and same lower bandwidth ml:
            L[j,j] = sqrt(A[j,j] - sum_{k=max(0,j-ml)}^{j-1} L[j,k]^2)
            L[i,j] = (A[i,j] - sum_{k=max(0,i-ml)}^{j-1} L[i,k] L[j,k]) / L[j,j]
                     for j < i <= min(n-1, j+ml)

        Returns a copy of the factor L stored in the same banded format.
        """
        L = BandedSPDMatrix(self.n, self.ml)
        for j in range(self.n):
            # Diagonal entry
            diag_sum = 0.0
            k_low = max(0, j - self.ml)
            for k in range(k_low, j):
                diag_sum += L.get(j, k) ** 2
            ajj = self.get(j, j)
            if ajj - diag_sum <= 0.0:
                # Numerical safeguard: perturb slightly to maintain SPD
                ajj = diag_sum + 1e-12
            L.set(j, j, np.sqrt(ajj - diag_sum))
            # Off-diagonal entries in column j
            i_high = min(self.n, j + self.ml + 1)
            for i in range(j + 1, i_high):
                off_sum = 0.0
                k_low2 = max(0, i - self.ml)
                k_start = max(k_low2, k_low)
                for k in range(k_start, j):
                    off_sum += L.get(i, k) * L.get(j, k)
                aij = self.get(i, j)
                ljj = L.get(j, j)
                if ljj == 0.0:
                    ljj = 1e-12
                L.set(i, j, (aij - off_sum) / ljj)
        return L

    def solve_cholesky(self, L: 'BandedSPDMatrix', b: np.ndarray) -> np.ndarray:
        """
        Solve A x = b using precomputed Cholesky factor L.
        Two triangular solves: L y = b, then L^T x = y.
        """
        if b.shape[0] != self.n:
            raise ValueError("Dimension mismatch.")
        y = np.zeros(self.n, dtype=float)
        x = np.zeros(self.n, dtype=float)

        # Forward substitution: L y = b
        for i in range(self.n):
            s = b[i]
            j_low = max(0, i - self.ml)
            for j in range(j_low, i):
                s -= L.get(i, j) * y[j]
            lii = L.get(i, i)
            if abs(lii) < 1e-15:
                lii = 1e-15
            y[i] = s / lii

        # Back substitution: L^T x = y
        for i in range(self.n - 1, -1, -1):
            s = y[i]
            j_high = min(self.n, i + self.ml + 1)
            for j in range(i + 1, j_high):
                s -= L.get(j, i) * x[j]
            lii = L.get(i, i)
            if abs(lii) < 1e-15:
                lii = 1e-15
            x[i] = s / lii
        return x

    @staticmethod
    def dif2_band(n: int) -> 'BandedSPDMatrix':
        """
        Construct the classic DIF2 (1D Laplacian) tridiagonal matrix
        with entries: main diagonal = 2, off-diagonals = -1.
        """
        A = BandedSPDMatrix(n, 1)
        for i in range(n):
            A.set(i, i, 2.0)
            if i > 0:
                A.set(i, i - 1, -1.0)
        return A


class SparseTriplet:
    """
    Sparse matrix in coordinate (triplet) format: (row, col, val).
    From HB-to-ST conversion concepts.
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.rows = []
        self.cols = []
        self.vals = []

    def add(self, i: int, j: int, v: float):
        if i < 0 or i >= self.n_rows or j < 0 or j >= self.n_cols:
            raise IndexError("Sparse index out of bounds.")
        if abs(v) > 0.0:
            self.rows.append(i)
            self.cols.append(j)
            self.vals.append(v)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n_rows, self.n_cols), dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def to_csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to CSR format: data, indices, indptr."""
        n = self.n_rows
        entries = list(zip(self.rows, self.cols, self.vals))
        entries.sort(key=lambda t: (t[0], t[1]))
        data = np.array([v for _, _, v in entries], dtype=float)
        indices = np.array([j for _, j, _ in entries], dtype=int)
        indptr = np.zeros(n + 1, dtype=int)
        for i, _, _ in entries:
            indptr[i + 1] += 1
        for i in range(n):
            indptr[i + 1] += indptr[i]
        return data, indices, indptr

    def matvec(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.n_cols:
            raise ValueError("Dimension mismatch in sparse matvec.")
        y = np.zeros(self.n_rows, dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[i] += v * x[j]
        return y


def banded_cholesky_solve(A_band: BandedSPDMatrix, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper: factorize and solve."""
    L = A_band.cholesky_band()
    return A_band.solve_cholesky(L, b)


def residual_norm(A_band: BandedSPDMatrix, x: np.ndarray, b: np.ndarray) -> float:
    """Compute infinity-norm residual ||b - A x||_inf."""
    n = A_band.n
    ax = np.zeros(n, dtype=float)
    for j in range(n):
        for i in range(j, min(n, j + A_band.ml + 1)):
            v = A_band.get(i, j)
            ax[i] += v * x[j]
            if i != j:
                ax[j] += v * x[i]
    r = b - ax
    return float(np.linalg.norm(r, ord=np.inf))
