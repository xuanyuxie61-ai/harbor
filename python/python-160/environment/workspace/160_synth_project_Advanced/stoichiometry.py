
import math
import numpy as np


class StoichiometricMatrix:

    def __init__(self, biomass_formula=(1.0, 1.4, 0.6, 0.01, 0.005)):
        self.biomass_formula = tuple(float(v) for v in biomass_formula)
        self.species_names = [
            'biomass', 'H2O', 'O2', 'CO', 'CO2',
            'H2', 'CH4', 'N2', 'H2S', 'tar', 'char'
        ]


        self.A = np.zeros((5, 11), dtype=float)
        self._build_matrix()

    def _build_matrix(self):
        c, h, o, n, s = self.biomass_formula

        self.A[:, 0] = [c, h, o, n, s]

        self.A[:, 1] = [0, 2, 1, 0, 0]

        self.A[:, 2] = [0, 0, 2, 0, 0]

        self.A[:, 3] = [1, 0, 1, 0, 0]

        self.A[:, 4] = [1, 0, 2, 0, 0]

        self.A[:, 5] = [0, 2, 0, 0, 0]

        self.A[:, 6] = [1, 4, 0, 0, 0]

        self.A[:, 7] = [0, 0, 0, 2, 0]

        self.A[:, 8] = [0, 2, 0, 0, 1]

        self.A[:, 9] = [10, 10, 1, 0, 0]

        self.A[:, 10] = [1, 0, 0, 0, 0]

    def row_swap(self, i, j):
        m = self.A.shape[0]
        if i == j or i < 1 or i > m or j < 1 or j > m:
            return False
        self.A[[i - 1, j - 1], :] = self.A[[j - 1, i - 1], :]
        return True

    def row_scale(self, s, i):
        m = self.A.shape[0]
        if i < 1 or i > m or abs(s) < 1.0e-15:
            return False
        self.A[i - 1, :] *= s
        return True

    def row_axpy(self, s1, i1, s2, i2):
        m = self.A.shape[0]
        if i1 == i2 or i1 < 1 or i1 > m or i2 < 1 or i2 > m:
            return False
        self.A[i1 - 1, :] = s1 * self.A[i1 - 1, :] + s2 * self.A[i2 - 1, :]
        return True

    def gauss_jordan_elimination(self, rhs=None):
        A_work = self.A.copy()
        m, n = A_work.shape
        if rhs is not None:
            rhs = np.asarray(rhs, dtype=float).copy()
            if rhs.shape[0] != m:
                raise ValueError("RHS dimension mismatch")

        pivot_row = 0
        for col in range(n):
            if pivot_row >= m:
                break

            pivot_val = abs(A_work[pivot_row, col])
            pivot_idx = pivot_row
            for r in range(pivot_row + 1, m):
                if abs(A_work[r, col]) > pivot_val:
                    pivot_val = abs(A_work[r, col])
                    pivot_idx = r
            if pivot_val < 1.0e-12:
                continue

            if pivot_idx != pivot_row:
                A_work[[pivot_row, pivot_idx], :] = A_work[[pivot_idx, pivot_row], :]
                if rhs is not None:
                    rhs[[pivot_row, pivot_idx]] = rhs[[pivot_idx, pivot_row]]

            piv = A_work[pivot_row, col]
            A_work[pivot_row, :] /= piv
            if rhs is not None:
                rhs[pivot_row] /= piv

            for r in range(m):
                if r != pivot_row and abs(A_work[r, col]) > 1.0e-12:
                    factor = A_work[r, col]
                    A_work[r, :] -= factor * A_work[pivot_row, :]
                    if rhs is not None:
                        rhs[r] -= factor * rhs[pivot_row]
            pivot_row += 1

        if rhs is not None:
            return A_work, rhs
        return A_work

    def rank(self):
        s = np.linalg.svd(self.A, compute_uv=False)
        tol = max(self.A.shape) * np.finfo(float).eps * s[0]
        return int(np.sum(s > tol))

    def nullspace_basis(self):
        m, n = self.A.shape
        _, s, vh = np.linalg.svd(self.A)
        tol = max(m, n) * np.finfo(float).eps * s[0]
        rank = int(np.sum(s > tol))
        null_dim = n - rank
        if null_dim <= 0:
            return np.zeros((n, 1))

        basis = vh[rank:, :].T
        return basis

    def validate_balance(self, stoich_vector):
        nu = np.asarray(stoich_vector, dtype=float)
        residual = self.A.dot(nu)
        norm = np.linalg.norm(residual)
        return norm < 1.0e-8, residual


class StoichiometricReducer:

    @staticmethod
    def gcd_two(a, b):
        a, b = abs(int(round(a))), abs(int(round(b)))
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def gcd_list(values):
        vals = [abs(int(round(v))) for v in values if abs(round(v)) > 0.5]
        if not vals:
            return 1
        g = vals[0]
        for v in vals[1:]:
            g = StoichiometricReducer.gcd_two(g, v)
            if g == 1:
                break
        return g

    @staticmethod
    def reduce_coefficients(coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        g = StoichiometricReducer.gcd_list(coeffs)
        if g > 1:
            return coeffs / g
        return coeffs

    @staticmethod
    def fermat_reduce(n):
        n = int(abs(n))
        if n < 2:
            return 1, n
        n2 = int(math.isqrt(n))
        if n2 * n2 < n:
            n2 += 1
        while n2 <= n:
            n3 = n2 * n2 - n
            if n3 >= 0:
                n4 = int(math.isqrt(n3))
                if n4 * n4 == n3:
                    return n2 + n4, n2 - n4
            n2 += 1
        return 1, n
