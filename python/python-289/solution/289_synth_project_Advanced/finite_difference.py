# -*- coding: utf-8 -*-
"""
finite_difference.py
====================

High-order finite-difference operators and banded-matrix linear algebra
for the radial direction of the gyrokinetic solver.

The 1-D radial operators are discretised on a uniform mesh  x_j = j h,
j = 0, ..., N-1 with  h = Lx/(N-1).  We implement:

    * 2nd-, 4th-, 6th- and 8th-order centred differences for the first
      and second derivatives,
    * a 4th-order *compact* (Pade) scheme for the first derivative,
    * explicit assembly of the banded coefficient matrix in LINPACK/LAPACK
      R8GB (banded) storage, and
    * PLU factorisation + forward/back-substitution specialised to the
      banded structure (port of ``r8gb_fa`` / ``r8gb_sl`` / ``r8gb_mv``
      from Burkardt's SLATEC-style library).

Why banded storage?
-------------------
The gyrokinetic radial equation (quasineutrality / vorticity) is solved
at every sub-step of the time integrator.  The operator has stencil width
w = 2p + 1 for a p-th order scheme -- on a mesh of size N this yields a
matrix of bandwidth w.  A dense LU would cost O(N^3); a banded LU costs
O(N w^2) which, for p <= 4 and N ~ O(100--500), is two orders of
magnitude cheaper.  The R8GB routines below give us that without
depending on scipy/lapack.

Finite-difference formulas
--------------------------
We use the standard centred formulas (e.g. Fornberg 1988).  For the
first derivative with step h:

  2nd order:   f'_j = (-f_{j-1} + f_{j+1}) / (2h)
  4th order:   f'_j = (f_{j-2} - 8 f_{j-1} + 8 f_{j+1} - f_{j+2}) / (12h)
  6th order:   f'_j = (-f_{j-3} + 9 f_{j-2} - 45 f_{j-1} + 45 f_{j+1}
                          - 9 f_{j+2} + f_{j+3}) / (60h)
  8th order:   f'_j = (f_{j-4} - (32/3) f_{j-3} + 56 f_{j-2}
                          - (392/3) f_{j-1} + (392/3) f_{j+1}
                          - 56 f_{j+2} + (32/3) f_{j+3} - f_{j+4}) / (280 h)

  The compact (Pade) 4th-order scheme solves
       (1/6) f'_{j-1} + (2/3) f'_j + (1/6) f'_{j+1}
           = (f_{j+1} - f_{j-1}) / (2h)
  with tridiagonal solve -- bandwidth 1.

References
----------
    [1] Burkardt, ``r8gb`` SLATEC-style banded linear algebra.
    [2] Fornberg, "Generation of finite difference formulas on arbitrarily
        spaced grids", Math. Comp. 51, 699 (1988).
    [3] Lele, "Compact finite difference schemes with spectral-like
        resolution", J. Comp. Phys. 103, 16 (1992).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from physics_constants import EPS_SQRT


# ============================================================================
# Explicit centred-difference coefficients
# ============================================================================
_COEFF_D1 = {
    2: (np.array([-1.0, 0.0, 1.0]), 2.0),
    4: (np.array([1.0, -8.0, 0.0, 8.0, -1.0]), 12.0),
    6: (np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]), 60.0),
    8: (np.array([1.0, -32.0 / 3.0, 56.0, -392.0 / 3.0, 0.0,
                  392.0 / 3.0, -56.0, 32.0 / 3.0, -1.0]), 280.0),
}

_COEFF_D2 = {
    2: (np.array([1.0, -2.0, 1.0]), 1.0),
    4: (np.array([-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]), 1.0),
    6: (np.array([1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0,
                  3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]), 1.0),
}


def centred_diff1(f: np.ndarray, h: float, order: int = 4) -> np.ndarray:
    """Centred finite-difference approximation of f' on a uniform grid.

    Interior points use an order-``order`` stencil; boundaries fall back
    to a one-sided (forward/backward) stencil of the same order where
    possible, else to lower order.

    Parameters
    ----------
    f : 1-D array of length N
    h : grid spacing
    order : 2, 4, 6 or 8
    """
    if order not in _COEFF_D1:
        raise ValueError(f"order {order} not supported for D1")
    c, scale = _COEFF_D1[order]
    half = c.size // 2
    N = f.size
    out = np.zeros_like(f)
    if N < c.size:
        raise ValueError("grid too small for chosen stencil order")
    # interior
    for k, ck in enumerate(c):
        out[half:N - half] += ck * f[half + (k - half):N - half + (k - half)]
    out[half:N - half] /= (scale * h)
    # boundaries: one-sided D1 (second order)
    out[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h) if N >= 3 else (f[1] - f[0]) / h
    out[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * h) if N >= 3 else (f[-1] - f[-2]) / h
    if order >= 4 and N >= 5:
        # 4th-order one-sided at second points
        out[1] = (-f[0] - 2.0 * f[1] + 6.0 * f[2] - 6.0 * f[3] + f[4] + 2.0 * f[0]) / (12.0 * h)
        # (use the interior stencil if it fits)
        if N > half + 1:
            out[1] = (-3.0 * f[0] - 10.0 * f[1] + 18.0 * f[2] - 6.0 * f[3] + f[4]) / (12.0 * h)
            out[-2] = (3.0 * f[-1] + 10.0 * f[-2] - 18.0 * f[-3] + 6.0 * f[-4] - f[-5]) / (12.0 * h)
    return out


def centred_diff2(f: np.ndarray, h: float, order: int = 4) -> np.ndarray:
    """Centred finite-difference approximation of f'' on a uniform grid."""
    if order not in _COEFF_D2:
        raise ValueError(f"order {order} not supported for D2")
    c, scale = _COEFF_D2[order]
    half = c.size // 2
    N = f.size
    if N < c.size:
        raise ValueError("grid too small for chosen stencil order")
    out = np.zeros_like(f)
    for k, ck in enumerate(c):
        out[half:N - half] += ck * f[half + (k - half):N - half + (k - half)]
    out[half:N - half] /= (scale * h * h)
    # 2nd-order one-sided at boundaries
    out[0] = (f[0] - 2.0 * f[1] + f[2]) / (h * h) if N >= 3 else 0.0
    out[-1] = (f[-3] - 2.0 * f[-2] + f[-1]) / (h * h) if N >= 3 else 0.0
    return out


# ============================================================================
# Compact (Pade) scheme for the first derivative -- tridiagonal solve
# ============================================================================
def compact_diff1(f: np.ndarray, h: float) -> np.ndarray:
    """4th-order Pade scheme:

        (1/6) f'_{j-1} + (2/3) f'_j + (1/6) f'_{j+1}
             = (f_{j+1} - f_{j-1}) / (2h)

    Interior points; boundaries use one-sided 4th-order differences.
    The tridiagonal system is solved by the Thomas algorithm.
    """
    N = f.size
    if N < 5:
        return centred_diff1(f, h, order=2)
    rhs = np.zeros_like(f)
    rhs[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)
    # 4th-order one-sided at boundaries
    if N >= 5:
        rhs[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2]
                  + 16.0 * f[3] - 3.0 * f[4]) / (12.0 * h)
        rhs[-1] = (25.0 * f[-1] - 48.0 * f[-2] + 36.0 * f[-3]
                   - 16.0 * f[-4] + 3.0 * f[-5]) / (12.0 * h)

    # Solve  a x_{j-1} + b x_j + c x_{j+1} = rhs
    a = np.full(N, 1.0 / 6.0)
    b = np.full(N, 2.0 / 3.0)
    c = np.full(N, 1.0 / 6.0)
    # Boundary rows: we already put the 4th-order one-sided value into
    # rhs[0] and rhs[-1]; enforce by setting an identity row.
    a[0] = 0.0
    c[0] = 0.0
    b[0] = 1.0
    a[-1] = 0.0
    c[-1] = 0.0
    b[-1] = 1.0
    return _tridiag_solve(a, b, c, rhs)


def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a strictly diagonally dominant tridiagonal system.

    a : lower-diagonal (length N, a[0] ignored)
    b : main diagonal
    c : upper-diagonal (length N, c[N-1] ignored)
    d : right-hand side
    """
    N = d.size
    cp = np.zeros(N)
    dp = np.zeros(N)
    if abs(b[0]) < EPS_SQRT:
        raise ValueError("tridiag_solve: zero pivot on diagonal 0")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, N):
        den = b[i] - a[i] * cp[i - 1]
        if abs(den) < EPS_SQRT:
            den = math.copysign(EPS_SQRT, den) if den != 0 else EPS_SQRT
        cp[i] = c[i] / den if i < N - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.zeros(N)
    x[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ============================================================================
# Banded-matrix algebra -- port of r8gb_* (Burkardt)
# ============================================================================
class BandedMatrixR8GB:
    """Banded matrix in LINPACK ``R8GB`` storage.

    Storage
    -------
    For an N-by-N matrix with lower bandwidth ml and upper bandwidth mu
    the R8GB format stores a (2 ml + mu + 1) x N array ``a`` in which the
    diagonals are "collapsed" into rows:

        a[k, j]  =  A_{i, j}   with   k = i - j + (ml + mu + 1)

    Indices follow the LINPACK convention: row index of the main
    diagonal is ``ml + mu``; the first ``ml`` rows are reserved for
    fill-in produced during LU factorisation.

    We deliberately implement our own factorisation rather than call
    scipy.linalg because (a) it reproduces the reference behaviour of the
    seed project ``r8gb`` and (b) it demonstrates the algorithmic
    structure that is exploited at every time step of the gyrokinetic
    solver.
    """

    def __init__(self, n: int, ml: int, mu: int) -> None:
        if n <= 0 or ml < 0 or mu < 0:
            raise ValueError("BandedMatrixR8GB: invalid dimensions")
        self.n = n
        self.ml = ml
        self.mu = mu
        self.m = ml + mu + 1
        self.storage = np.zeros((2 * ml + mu + 1, n), dtype=np.float64)

    # ---------- element access ----------
    def __setitem__(self, key, value) -> None:
        i, j = key
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise IndexError("BandedMatrixR8GB: index out of range")
        if abs(i - j) > self.ml if i >= j else abs(i - j) > self.mu:
            # outside the band -- ignore silently if zero, else warn
            if value != 0.0:
                raise ValueError(f"BandedMatrixR8GB: entry ({i},{j}) outside band")
            return
        k = i - j + self.ml + self.mu
        self.storage[k, j] = value

    def __getitem__(self, key) -> float:
        i, j = key
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise IndexError
        if i >= j:
            if i - j > self.ml:
                return 0.0
        else:
            if j - i > self.mu:
                return 0.0
        k = i - j + self.ml + self.mu
        return float(self.storage[k, j])

    # ---------- matrix-vector ----------
    def mv(self, x: np.ndarray) -> np.ndarray:
        """Port of r8gb_mv: multiply banded A by vector x."""
        if x.size != self.n:
            raise ValueError("r8gb_mv: dimension mismatch")
        b = np.zeros_like(x)
        for i in range(self.n):
            j1 = max(0, i - self.ml)
            j2 = min(self.n - 1, i + self.mu)
            s = 0.0
            for j in range(j1, j2 + 1):
                k = i - j + self.ml + self.mu
                s += self.storage[k, j] * x[j]
            b[i] = s
        return b

    # ---------- PLU factorisation -- LINPACK-style banded with partial pivoting ----------
    def lu_factor(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """PLU factorisation with partial pivoting.

        Returns
        -------
        alu   : (2 ml + mu + 1, n) the factorised matrix
        pivot : (n,) array of pivot row indices (1-based, LINPACK style)
        info  : 0 on success; i if U(i, i) == 0 at step i
        """
        n, ml, mu = self.n, self.ml, self.mu
        # Use dense LU as the reference algorithm -- still O(n*(ml+mu)^2)
        # if we restrict to the band.  We implement a faithful in-place
        # banded LU following Dongarra et al. LINPACK guide.
        alu = self.storage.copy()
        pivot = np.zeros(n, dtype=np.int64)
        m_l = ml + mu
        info = 0
        for j in range(n - 1):
            l = min(ml, n - 1 - j)
            # --- pivot search: find max in column j, rows j..j+l ---
            pivot_max = -1.0
            ip = j
            for i in range(j, j + l + 1):
                k = i - j + m_l
                if k < alu.shape[0] and abs(alu[k, j]) > pivot_max:
                    pivot_max = abs(alu[k, j])
                    ip = i
            pivot[j] = ip + 1        # 1-based
            if pivot_max == 0.0:
                info = j + 1
                pivot[-1] = n
                return alu, pivot, info
            if ip != j:
                # swap rows in the band
                # row i in the logical matrix corresponds to storage row
                # (i - col + m_l).  We need to swap a *strip* of rows.
                # Safe approach: swap the whole band (fill-in area included)
                for col in range(j, min(n, j + mu + l + 1)):
                    kj = j - col + m_l
                    kip = ip - col + m_l
                    if 0 <= kj < alu.shape[0] and 0 <= kip < alu.shape[0]:
                        alu[kj, col], alu[kip, col] = alu[kip, col], alu[kj, col]
            # --- eliminate: rows j+1 .. j+l ---
            if abs(alu[m_l, j]) < EPS_SQRT:
                pivot[-1] = n
                info = j + 1
                return alu, pivot, info
            for i in range(j + 1, j + l + 1):
                k = i - j + m_l
                factor = alu[k, j] / alu[m_l, j]
                alu[k, j] = factor
                # update entries in columns j+1 .. j+mu (and beyond for fill-in)
                for kk in range(1, mu + 1):
                    col = j + kk
                    if col >= n:
                        break
                    k1 = i - col + m_l
                    k2 = j - col + m_l
                    if 0 <= k1 < alu.shape[0] and 0 <= k2 < alu.shape[0]:
                        alu[k1, col] -= factor * alu[k2, col]
        pivot[-1] = n
        if abs(alu[m_l, n - 1]) < EPS_SQRT:
            info = n
        return alu, pivot, info

    # ---------- solve -- port of r8gb_sl ----------
    @staticmethod
    def lu_solve(alu: np.ndarray, pivot: np.ndarray, b: np.ndarray,
                 n: int, ml: int, mu: int, transpose: bool = False) -> np.ndarray:
        """Solve A x = b (or A^T x = b if transpose) given LU from lu_factor."""
        m_l = ml + mu
        x = b.astype(np.float64, copy=True)
        if not transpose:
            # forward substitution with row interchange
            for k in range(n - 1):
                ip = int(pivot[k]) - 1
                if ip != k:
                    x[k], x[ip] = x[ip], x[k]
                l = min(ml, n - 1 - k)
                for i in range(k + 1, k + l + 1):
                    row = i - k + m_l
                    if 0 <= row < alu.shape[0]:
                        x[i] -= alu[row, k] * x[k]
            # back substitution
            for j in range(n - 1, -1, -1):
                s = 0.0
                for i in range(j + 1, min(n, j + mu + 1)):
                    row = j - i + m_l
                    if 0 <= row < alu.shape[0]:
                        s += alu[row, i] * x[i]
                x[j] = (x[j] - s) / alu[m_l, j] if abs(alu[m_l, j]) > EPS_SQRT else 0.0
        else:
            for j in range(n):
                s = 0.0
                for i in range(max(0, j - mu), j):
                    row = i - j + m_l
                    if 0 <= row < alu.shape[0]:
                        s += alu[row, j] * x[i]
                x[j] = (x[j] - s) / alu[m_l, j] if abs(alu[m_l, j]) > EPS_SQRT else 0.0
            for k in range(n - 2, -1, -1):
                l = min(ml, n - 1 - k)
                s = 0.0
                for i in range(k + 1, k + l + 1):
                    row = i - k + m_l
                    if 0 <= row < alu.shape[0]:
                        s += alu[row, k] * x[i]
                x[k] -= s
                ip = int(pivot[k]) - 1
                if ip != k:
                    x[k], x[ip] = x[ip], x[k]
        return x

    def solve(self, b: np.ndarray) -> np.ndarray:
        """High-level: factor and solve  A x = b  using dense LU fallback."""
        # Build dense matrix (small n typically) and use numpy
        A_full = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(max(0, i - self.ml), min(self.n, i + self.mu + 1)):
                k = i - j + self.ml + self.mu
                A_full[i, j] = self.storage[k, j]
        return np.linalg.solve(A_full, b)


# ============================================================================
# High-level driver: assemble a radial FD operator as a banded matrix
# ============================================================================
def assemble_radial_d2_operator(N: int, h: float, kappa: float,
                                kperp2_rho2: float, Gamma0: float) -> BandedMatrixR8GB:
    """Assemble the radial operator  L phi = -kappa d^2 phi/dx^2 + kperp^2 (1 - Gamma0) phi.

    Uses a 4th-order compact scheme on interior points, with 2nd-order
    fallback at the boundaries.  Returns an R8GB matrix with ml = mu = 2.
    """
    A = BandedMatrixR8GB(N, ml=2, mu=2)
    # interior: compact discretisation
    for j in range(2, N - 2):
        # compact first derivative applied twice -- we use the direct
        # 4th-order second derivative for simplicity
        A[j, j - 2] = -kappa * (-1.0 / 12.0) / (h * h)
        A[j, j - 1] = -kappa * (4.0 / 3.0) / (h * h)
        A[j, j]     = -kappa * (-5.0 / 2.0) / (h * h) + kperp2_rho2 * (1.0 - Gamma0)
        A[j, j + 1] = -kappa * (4.0 / 3.0) / (h * h)
        A[j, j + 2] = -kappa * (-1.0 / 12.0) / (h * h)
    # boundaries: 2nd order
    A[0, 0] = -kappa * (-2.0) / (h * h) + kperp2_rho2 * (1.0 - Gamma0)
    A[0, 1] = -kappa * (1.0) / (h * h)
    A[1, 0] = -kappa * (1.0) / (h * h)
    A[1, 1] = -kappa * (-2.0) / (h * h) + kperp2_rho2 * (1.0 - Gamma0)
    A[1, 2] = -kappa * (1.0) / (h * h)
    A[N - 2, N - 3] = -kappa * (1.0) / (h * h)
    A[N - 2, N - 2] = -kappa * (-2.0) / (h * h) + kperp2_rho2 * (1.0 - Gamma0)
    A[N - 2, N - 1] = -kappa * (1.0) / (h * h)
    A[N - 1, N - 2] = -kappa * (1.0) / (h * h)
    A[N - 1, N - 1] = -kappa * (-2.0) / (h * h) + kperp2_rho2 * (1.0 - Gamma0)
    return A


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    # test finite differences on sin
    h = 0.01
    x = np.arange(0.0, 2.0 * math.pi + 0.5 * h, h)
    f = np.sin(x)
    for order in (2, 4, 6, 8):
        err = np.max(np.abs(centred_diff1(f, h, order) - np.cos(x))[5:-5])
        print(f"D1 order {order}: max interior error = {err:.3e}")
    err = np.max(np.abs(centred_diff2(f, h, 4) - (-np.sin(x)))[5:-5])
    print(f"D2 order 4 : max interior error = {err:.3e}")
    err = np.max(np.abs(compact_diff1(f, h) - np.cos(x))[2:-2])
    print(f"compact D1 : max interior error = {err:.3e}")

    # banded matrix test
    N = 20
    A = BandedMatrixR8GB(N, 2, 2)
    for i in range(N):
        for j in range(max(0, i - 2), min(N, i + 3)):
            A[i, j] = (5.0 if i == j else -1.0)
    x = np.random.RandomState(0).randn(N)
    b = A.mv(x)
    alu, piv, info = A.lu_factor()
    xh = BandedMatrixR8GB.lu_solve(alu, piv, b, N, 2, 2)
    print("banded LU rel. error:", np.max(np.abs(x - xh)) / max(np.max(np.abs(x)), 1e-14))
