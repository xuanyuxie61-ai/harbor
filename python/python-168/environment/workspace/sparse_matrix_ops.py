
import numpy as np


class R8GDMatrix:

    def __init__(self, n, ndiag, offsets, values):
        self.n = max(int(n), 1)
        self.ndiag = max(int(ndiag), 1)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)

        if self.values.shape != (self.n, self.ndiag):
            raise ValueError(f"values shape {self.values.shape} inconsistent with ({self.n}, {self.ndiag})")
        if self.offsets.shape[0] != self.ndiag:
            raise ValueError("offsets length mismatch")

    def mv(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError("vector dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    y[i] += self.values[i, d] * x[j]
        return y

    def mtv(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError("vector dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    y[j] += self.values[i, d] * x[i]
        return y

    def to_dense(self):
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    A[i, j] = self.values[i, d]
        return A


class SORSolver:

    def __init__(self, omega=1.5, max_iter=1000, tol=1e-10):
        self.omega = max(min(float(omega), 1.999), 0.001)
        self.max_iter = max(int(max_iter), 1)
        self.tol = max(float(tol), 1e-15)

    def solve(self, A, b, x0=None):
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]

        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()


        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-14):

            min_diag = np.min(np.abs(diag[diag != 0])) if np.any(diag != 0) else 1.0
            for i in range(n):
                if abs(diag[i]) < 1e-14:
                    A[i, i] = min_diag * 1e-6

        for it in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                if abs(A[i, i]) > 1e-15:
                    x[i] = (1.0 - self.omega) * x[i] + self.omega * (b[i] - sigma) / A[i, i]
                else:
                    x[i] = x[i]


            diff = np.linalg.norm(x - x_old)
            if diff < self.tol:
                residual = A @ x - b
                return x, np.linalg.norm(residual), it + 1, True

        residual = A @ x - b
        return x, np.linalg.norm(residual), self.max_iter, False

    def solve_sparse(self, r8gd_A, b, x0=None):
        b = np.asarray(b, dtype=np.float64)
        n = r8gd_A.n

        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()


        diag_idx = -1
        for d in range(r8gd_A.ndiag):
            if r8gd_A.offsets[d] == 0:
                diag_idx = d
                break

        if diag_idx < 0:
            raise ValueError("R8GD matrix must contain main diagonal")

        for it in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                sigma = 0.0
                for d in range(r8gd_A.ndiag):
                    j = i + r8gd_A.offsets[d]
                    if j != i and 0 <= j < n:
                        sigma += r8gd_A.values[i, d] * x[j]

                a_ii = r8gd_A.values[i, diag_idx]
                if abs(a_ii) > 1e-15:
                    x[i] = (1.0 - self.omega) * x[i] + self.omega * (b[i] - sigma) / a_ii

            diff = np.linalg.norm(x - x_old)
            if diff < self.tol:
                residual = r8gd_A.mv(x) - b
                return x, np.linalg.norm(residual), it + 1, True

        residual = r8gd_A.mv(x) - b
        return x, np.linalg.norm(residual), self.max_iter, False
