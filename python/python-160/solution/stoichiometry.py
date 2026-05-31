"""
stoichiometry.py
================
Stoichiometric matrix operations and elemental mass balance for gasification.

Incorporates algorithms from:
  - 736_matman (elementary row operations, LU decomposition tracking)
  - 420_fermat_factor (integer factorization for coefficient reduction)

Scientific role:
  Manages the stoichiometric matrix for biomass gasification reactions.
  The general gasification reaction can be written as:
    CH_x O_y + a H₂O + b O₂ → c CO + d CO₂ + e H₂ + f CH₄ + g C_{s} + ...
  
  This module performs elementary row operations to reduce the stoichiometric
  matrix, solve for unknown coefficients, and verify elemental balances.
"""

import math
import numpy as np


class StoichiometricMatrix:
    """
    Stoichiometric matrix for gasification chemistry.

    Rows: elements (C, H, O, N, S)
    Cols: species (biomass, H2O, O2, CO, CO2, H2, CH4, N2, H2S, tar, char)
    """

    def __init__(self, biomass_formula=(1.0, 1.4, 0.6, 0.01, 0.005)):
        """
        Parameters
        ----------
        biomass_formula : tuple
            (C, H, O, N, S) atomic ratios per C atom.
        """
        self.biomass_formula = tuple(float(v) for v in biomass_formula)
        self.species_names = [
            'biomass', 'H2O', 'O2', 'CO', 'CO2',
            'H2', 'CH4', 'N2', 'H2S', 'tar', 'char'
        ]
        # Elemental composition matrix: rows = elements, cols = species
        # Each column: (C, H, O, N, S) count per mole of species
        self.A = np.zeros((5, 11), dtype=float)
        self._build_matrix()

    def _build_matrix(self):
        """Construct the elemental composition matrix."""
        c, h, o, n, s = self.biomass_formula
        # biomass
        self.A[:, 0] = [c, h, o, n, s]
        # H2O
        self.A[:, 1] = [0, 2, 1, 0, 0]
        # O2
        self.A[:, 2] = [0, 0, 2, 0, 0]
        # CO
        self.A[:, 3] = [1, 0, 1, 0, 0]
        # CO2
        self.A[:, 4] = [1, 0, 2, 0, 0]
        # H2
        self.A[:, 5] = [0, 2, 0, 0, 0]
        # CH4
        self.A[:, 6] = [1, 4, 0, 0, 0]
        # N2
        self.A[:, 7] = [0, 0, 0, 2, 0]
        # H2S
        self.A[:, 8] = [0, 2, 0, 0, 1]
        # tar (approximated as C10H10O)
        self.A[:, 9] = [10, 10, 1, 0, 0]
        # char (approximated as C)
        self.A[:, 10] = [1, 0, 0, 0, 0]

    def row_swap(self, i, j):
        """Elementary row operation: swap rows i and j (1-indexed)."""
        m = self.A.shape[0]
        if i == j or i < 1 or i > m or j < 1 or j > m:
            return False
        self.A[[i - 1, j - 1], :] = self.A[[j - 1, i - 1], :]
        return True

    def row_scale(self, s, i):
        """Elementary row operation: row i ← s * row i."""
        m = self.A.shape[0]
        if i < 1 or i > m or abs(s) < 1.0e-15:
            return False
        self.A[i - 1, :] *= s
        return True

    def row_axpy(self, s1, i1, s2, i2):
        """Elementary row operation: row i1 ← s1 * row i1 + s2 * row i2."""
        m = self.A.shape[0]
        if i1 == i2 or i1 < 1 or i1 > m or i2 < 1 or i2 > m:
            return False
        self.A[i1 - 1, :] = s1 * self.A[i1 - 1, :] + s2 * self.A[i2 - 1, :]
        return True

    def gauss_jordan_elimination(self, rhs=None):
        """
        Perform Gauss-Jordan elimination on the stoichiometric matrix.
        If rhs is provided, solve A * x = rhs for x.
        
        Returns reduced row echelon form and solution if applicable.
        """
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
            # Find pivot
            pivot_val = abs(A_work[pivot_row, col])
            pivot_idx = pivot_row
            for r in range(pivot_row + 1, m):
                if abs(A_work[r, col]) > pivot_val:
                    pivot_val = abs(A_work[r, col])
                    pivot_idx = r
            if pivot_val < 1.0e-12:
                continue
            # Swap
            if pivot_idx != pivot_row:
                A_work[[pivot_row, pivot_idx], :] = A_work[[pivot_idx, pivot_row], :]
                if rhs is not None:
                    rhs[[pivot_row, pivot_idx]] = rhs[[pivot_idx, pivot_row]]
            # Scale pivot row
            piv = A_work[pivot_row, col]
            A_work[pivot_row, :] /= piv
            if rhs is not None:
                rhs[pivot_row] /= piv
            # Eliminate other rows
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
        """Matrix rank via SVD."""
        s = np.linalg.svd(self.A, compute_uv=False)
        tol = max(self.A.shape) * np.finfo(float).eps * s[0]
        return int(np.sum(s > tol))

    def nullspace_basis(self):
        """
        Compute a basis for the nullspace of A (reaction invariants).
        Vectors v in null(A) satisfy A v = 0, meaning the reaction
        conserves all elements.
        """
        m, n = self.A.shape
        _, s, vh = np.linalg.svd(self.A)
        tol = max(m, n) * np.finfo(float).eps * s[0]
        rank = int(np.sum(s > tol))
        null_dim = n - rank
        if null_dim <= 0:
            return np.zeros((n, 1))
        # Last null_dim rows of Vh^T form nullspace basis
        basis = vh[rank:, :].T
        return basis

    def validate_balance(self, stoich_vector):
        """
        Verify that a stoichiometric vector satisfies elemental balance.
        A * ν ≈ 0
        """
        nu = np.asarray(stoich_vector, dtype=float)
        residual = self.A.dot(nu)
        norm = np.linalg.norm(residual)
        return norm < 1.0e-8, residual


class StoichiometricReducer:
    """
    Reduce stoichiometric coefficients by factoring out GCD.
    Uses Fermat factorization style search for common divisors.
    """

    @staticmethod
    def gcd_two(a, b):
        """Euclidean algorithm for GCD."""
        a, b = abs(int(round(a))), abs(int(round(b)))
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def gcd_list(values):
        """GCD of a list of integers."""
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
        """
        Divide all coefficients by their GCD to obtain smallest integers.
        """
        coeffs = np.asarray(coeffs, dtype=float)
        g = StoichiometricReducer.gcd_list(coeffs)
        if g > 1:
            return coeffs / g
        return coeffs

    @staticmethod
    def fermat_reduce(n):
        """
        Fermat factorization: find a, b such that n = a² - b².
        Useful for finding approximate square factors in reaction orders.
        """
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
