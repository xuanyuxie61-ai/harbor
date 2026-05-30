import numpy as np
from typing import Tuple, Optional


class BandedSolver:

    def __init__(self, n: int, ml: int, mu: int, compact: bool = True):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.compact = compact
        self.m = mu + 1
        self._lu: Optional[np.ndarray] = None
        self._pivot: Optional[np.ndarray] = None
        self._info: int = 0

    def full_to_compact(self, A_full: np.ndarray) -> np.ndarray:
        A_band = np.zeros((self.ml + self.mu + 1, self.n))
        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.mu
                A_band[k, j] = A_full[i, j]
        return A_band

    def compact_to_full(self, A_band: np.ndarray) -> np.ndarray:
        A_full = np.zeros((self.n, self.n))
        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.mu
                A_full[i, j] = A_band[k, j]
        return A_full

    def factorize_np(self, A_band: np.ndarray) -> np.ndarray:



        pass

    def factorize_with_pivot(self, A_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nrow = 2 * self.ml + self.mu + 1
        alu = np.zeros((nrow, self.n))

        for j in range(self.n):
            i1 = max(0, j - self.mu)
            i2 = min(self.n - 1, j + self.ml)
            for i in range(i1, i2 + 1):
                k = i - j + self.ml + self.mu
                alu[k, j] = A_band[i - j + self.mu, j]
        pivot = np.zeros(self.n, dtype=int)
        m = self.ml + self.mu + 1
        info = 0
        jz = self.mu + 1
        j1 = min(self.n, m) - 1
        for jz_idx in range(jz, j1 + 1):
            i0 = m + 1 - jz_idx
            if i0 <= self.ml:
                alu[i0 - 1:self.ml, jz_idx - 1] = 0.0
        jz = j1
        ju = 0
        for k in range(self.n - 1):
            jz += 1
            if jz <= self.n:
                alu[0:self.ml, jz - 1] = 0.0
            lm = min(self.ml, self.n - 1 - k)
            l = m - 1
            for j in range(m, m + lm):
                if abs(alu[l, k]) < abs(alu[j, k]):
                    l = j
            pivot[k] = l + k - m + 1
            if abs(alu[l, k]) < 1e-20:
                info = k + 1
                raise ValueError(f"Zero pivot at step {k} in banded PLU factorization")

            if l != m - 1:
                t = alu[l, k]
                alu[l, k] = alu[m - 1, k]
                alu[m - 1, k] = t
            alu[m:m + lm, k] = -alu[m:m + lm, k] / alu[m - 1, k]
            ju = min(ju + self.mu + pivot[k], self.n)
            ju = min(ju, self.n)
            mm = m
            for j in range(k + 1, ju):
                l -= 1
                mm -= 1
                if l != mm:
                    t = alu[l, j]
                    alu[l, j] = alu[mm, j]
                    alu[mm, j] = t
                alu[mm:mm + lm, j] = alu[mm:mm + lm, j] + alu[mm - 1, j] * alu[m:m + lm, k]
        pivot[self.n - 1] = self.n - 1
        if abs(alu[m - 1, self.n - 1]) < 1e-20:
            info = self.n
            raise ValueError("Zero pivot at final step in banded PLU factorization")
        self._lu = alu
        self._pivot = pivot
        self._info = info
        return alu, pivot

    def solve_np(self, alu: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = b.copy()
        m = self.mu + 1

        for k in range(self.n - 1):
            lm = min(self.ml, self.n - 1 - k)
            x[k + 1:k + 1 + lm] = x[k + 1:k + 1 + lm] + alu[m:m + lm, k] * x[k]

        for k in range(self.n - 1, -1, -1):
            x[k] = x[k] / alu[m - 1, k]
            lm = min(self.ml, self.n - 1 - k)
            for i in range(1, lm + 1):
                if k - i >= 0:
                    x[k - i] = x[k - i] - alu[m - 1 + i, k - i] * x[k]
        return x

    def solve_with_pivot(self, alu: np.ndarray, pivot: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = b.copy()
        m = self.ml + self.mu + 1

        for k in range(self.n - 1):
            lm = min(self.ml, self.n - 1 - k)
            l = pivot[k]
            if l != k:
                t = x[l]
                x[l] = x[k]
                x[k] = t
            x[k + 1:k + 1 + lm] = x[k + 1:k + 1 + lm] + alu[m:m + lm, k] * x[k]

        for k in range(self.n - 1, -1, -1):
            x[k] = x[k] / alu[m - 1, k]
            lm = min(self.ml, self.n - 1 - k)
            for i in range(1, lm + 1):
                if k - i >= 0:
                    x[k - i] = x[k - i] - alu[m - 1 + i, k - i] * x[k]
        return x

    def solve_system(self, A_band: np.ndarray, b: np.ndarray, use_pivot: bool = False) -> np.ndarray:
        if use_pivot:
            alu, pivot = self.factorize_with_pivot(A_band)
            return self.solve_with_pivot(alu, pivot, b)
        else:
            alu = self.factorize_np(A_band)
            return self.solve_np(alu, b)


def extract_banded_submatrix(K: np.ndarray, contact_nodes: np.ndarray,
                              n_nodes: int, ml: int, mu: int) -> np.ndarray:
    n_c = len(contact_nodes)
    n_sub = 2 * n_c
    idx = np.zeros(n_sub, dtype=int)
    for i, node in enumerate(contact_nodes):
        idx[2 * i] = 2 * node
        idx[2 * i + 1] = 2 * node + 1
    K_sub = K[np.ix_(idx, idx)]

    solver = BandedSolver(n_sub, ml, mu, compact=True)
    return solver.full_to_compact(K_sub)
