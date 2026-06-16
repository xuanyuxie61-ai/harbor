"""
banded_covariance.py
====================

Banded positive-definite linear solvers for the Gaussian-process (GP)
covariance system that arises in the multi-fidelity UQ predictor.

Background
----------
A GP with a banded covariance matrix K (arising when observations lie on
a regular grid or a tensor product with local kernels) can be solved in
O(n mu^2) time rather than O(n^3), where mu is the half-bandwidth.  This
is the analog of the r8pbu (real 8-byte packed banded upper) storage
format of project 988, where only the diagonal and upper triangle of A
is stored in a compact diagonal format.

Storage convention (Burkardt r8pbu)
-----------------------------------
  A is an n x n SPD band matrix with half-bandwidth mu.
  We store a (mu+1) x n array `AB`:
    - row mu+1 : main diagonal
    - row mu   : first super-diagonal, columns 2..n
    - row mu-k : k-th super-diagonal, columns (k+1)..n

Functions provided
------------------
  - `banded_mv(n, mu, AB, x)`: matrix-vector product y = A x.
  - `banded_fa(n, mu, AB)`: Cholesky factorization (in-place, AB -> L L^T).
  - `banded_sl(n, mu, AB, b)`: solve A x = b given factored AB.
  - `banded_cg(n, mu, AB, b, x0, tol, maxit)`: conjugate-gradient solver.
  - `banded_sor(n, mu, AB, b, omega, x0, tol, maxit)`: SOR iterative solver.
  - `gp_covariance_banded(xi, xj, ell, sigma_f, mu)`: assemble a banded
    Matern-3/2 covariance matrix on a 1-D sorted grid.
  - `gp_predict_banded(X_train, y_train, X_test, ell, sigma_f, sigma_n, mu)`:
    return GP posterior mean and std using the banded solver.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# Matrix-vector product: y = A x for banded SPD A in r8pbu format.
# ----------------------------------------------------------------------
def banded_mv(n: int, mu: int, AB: List[List[float]],
              x: List[float]) -> List[float]:
    """Compute y = A x for SPD band matrix A in packed banded-upper form."""
    if len(x) != n:
        raise ValueError("banded_mv: x length mismatch.")
    y = [0.0] * n
    for i in range(n):
        # Diagonal contribution.
        y[i] += AB[mu][i] * x[i]
        # Super-diagonal contributions (symmetric: also add sub-diagonal).
        for k in range(1, mu + 1):
            if i + k < n:
                val = AB[mu - k][i + k] * x[i + k]
                y[i] += val
                y[i + k] += AB[mu - k][i + k] * x[i]
    return y


# ----------------------------------------------------------------------
# Cholesky factorization of SPD band matrix (in-place).
#
# After factorization, AB contains the upper Cholesky factor U such that
# A = U^T U.  The diagonal of AB holds diag(U); super-diagonals hold
# the corresponding super-diagonals of U.
# ----------------------------------------------------------------------
def banded_fa(n: int, mu: int, AB: List[List[float]]) -> List[List[float]]:
    """In-place Cholesky factorization of SPD band matrix.  Returns AB.

    After the call, AB stores the upper Cholesky factor U such that
    A = U^T U, in the packed banded-upper format.  Delegates to the
    textbook recurrence in `_cholesky_band_textbook`.
    """
    return _cholesky_band_textbook(n, mu, AB)


def _cholesky_band_textbook(n: int, mu: int,
                            AB_in: List[List[float]]) -> List[List[float]]:
    """Banded Cholesky via textbook recurrence.

    Computes the upper Cholesky factor U (in packed form) of SPD band
    matrix A: A = U^T U.

    Storage: AB[mu, j] = U[j, j] (diagonal)
             AB[mu - i, j + i] = U[j, j + i] for i = 1..mu (super-diags)

    Recurrence:
      U[j,j]   = sqrt(A[j,j] - sum_{k} U[k,j]^2)
      U[j,j+i] = (A[j,j+i] - sum_k U[k,j] U[k,j+i]) / U[j,j]
    where k ranges over max(0, j-mu) .. j-1, and U[k, j+i] is in band
    only if (j+i) - k <= mu.
    """
    # Working copy.
    U = [row[:] for row in AB_in]
    for j in range(n):
        # Diagonal entry.
        s = U[mu][j]
        k_start = max(0, j - mu)
        for k in range(k_start, j):
            d = j - k                     # 1 <= d <= mu
            val = U[mu - d][j]
            s -= val * val
        if s <= 0.0:
            # Regularize for numerical safety.
            s = max(s, 1.0e-14)
        U[mu][j] = math.sqrt(s)
        Ujj = U[mu][j]
        # Super-diagonal entries of column j.
        for i in range(1, mu + 1):
            if j + i >= n:
                break
            s = U[mu - i][j + i]           # original A[j, j+i]
            for k in range(k_start, j):
                d_kj = j - k
                d_kji = j + i - k
                if d_kji <= mu:            # only in-band U[k, j+i]
                    s -= U[mu - d_kj][j] * U[mu - d_kji][j + i]
            U[mu - i][j + i] = s / Ujj
    # Copy back.
    for i in range(mu + 1):
        for j in range(n):
            AB_in[i][j] = U[i][j]
    return AB_in


# ----------------------------------------------------------------------
# Triangular solves.
# ----------------------------------------------------------------------
def banded_sl(n: int, mu: int, AB: List[List[float]],
              b: List[float]) -> List[float]:
    """Solve A x = b given the factored (Cholesky) banded matrix AB."""
    # Forward solve: U^T z = b.
    z = list(b)
    for j in range(n):
        for k in range(1, mu + 1):
            if j - k >= 0:
                row = mu - k
                z[j] -= AB[row][j] * z[j - k]
        z[j] /= AB[mu][j]
    # Backward solve: U x = z.
    x = [0.0] * n
    for j in range(n - 1, -1, -1):
        s = z[j]
        for k in range(1, mu + 1):
            if j + k < n:
                row = mu - k
                s -= AB[row][j + k] * x[j + k]
        x[j] = s / AB[mu][j]
    return x


# ----------------------------------------------------------------------
# Conjugate gradient solver for SPD band systems.
# ----------------------------------------------------------------------
def banded_cg(
    n: int, mu: int, AB: List[List[float]],
    b: List[float], x0: Optional[List[float]] = None,
    tol: float = 1.0e-10, maxit: int = 400,
) -> Tuple[List[float], int, float]:
    """Solve A x = b by the conjugate gradient method.

    Returns (x, iters, residual_norm).
    """
    x = list(x0) if x0 is not None else [0.0] * n
    r = [b[i] - banded_mv(n, mu, AB, x)[i] for i in range(n)]
    p = list(r)
    rs_old = sum(v * v for v in r)
    for it in range(maxit):
        if math.sqrt(rs_old) < tol:
            return x, it, math.sqrt(rs_old)
        Ap = banded_mv(n, mu, AB, p)
        pAp = sum(p[i] * Ap[i] for i in range(n))
        if abs(pAp) < 1.0e-30:
            break
        alpha = rs_old / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rs_new = sum(v * v for v in r)
        if math.sqrt(rs_new) < tol:
            return x, it + 1, math.sqrt(rs_new)
        beta = rs_new / rs_old
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        rs_old = rs_new
    return x, maxit, math.sqrt(rs_old)


# ----------------------------------------------------------------------
# SOR solver.
# ----------------------------------------------------------------------
def banded_sor(
    n: int, mu: int, AB: List[List[float]],
    b: List[float], omega: float = 1.5,
    x0: Optional[List[float]] = None,
    tol: float = 1.0e-10, maxit: int = 400,
) -> Tuple[List[float], int, float]:
    """Successive over-relaxation for A x = b (SPD band)."""
    x = list(x0) if x0 is not None else [0.0] * n
    for it in range(maxit):
        x_old = list(x)
        for i in range(n):
            sigma = 0.0
            for k in range(1, mu + 1):
                if i - k >= 0:
                    sigma += AB[mu - k][i] * x[i - k]
                if i + k < n:
                    sigma += AB[mu - k][i + k] * x[i + k]
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sigma) / AB[mu][i]
        diff = math.sqrt(sum((x[i] - x_old[i]) ** 2 for i in range(n)))
        if diff < tol:
            return x, it + 1, diff
    return x, maxit, math.sqrt(sum((x[i] - x_old[i]) ** 2 for i in range(n)))


# ----------------------------------------------------------------------
# GP Matern-3/2 covariance (banded assembly on sorted 1-D grid).
#
# k(r) = sigma_f^2 (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)
#
# On a sorted grid x_1 < x_2 < ... < x_n, the (i, j) entry depends on
# |i - j|.  We truncate beyond bandwidth mu so that entries with
# |x_i - x_j| > mu * dx are zeroed (mu is chosen to capture the
# correlation length).
# ----------------------------------------------------------------------
def matern32(r: float, ell: float, sigma_f: float) -> float:
    """Matern-3/2 covariance kernel."""
    if ell <= 0.0:
        raise ValueError("matern32: ell must be > 0.")
    s = math.sqrt(3.0) * abs(r) / ell
    return sigma_f ** 2 * (1.0 + s) * math.exp(-s)


def gp_covariance_banded(
    x_grid: List[float], ell: float, sigma_f: float,
    sigma_n: float, mu: int,
) -> List[List[float]]:
    """Assemble the banded GP covariance matrix (with nugget) on a 1-D grid."""
    n = len(x_grid)
    AB = [[0.0] * n for _ in range(mu + 1)]
    for j in range(n):
        # Diagonal:
        AB[mu][j] = matern32(0.0, ell, sigma_f) + sigma_n ** 2
        # Super-diagonals:
        for i in range(1, mu + 1):
            if j + i >= n:
                break
            r = x_grid[j + i] - x_grid[j]
            val = matern32(r, ell, sigma_f)
            if abs(val) < 1.0e-16:
                val = 0.0
            AB[mu - i][j + i] = val
    return AB


# ----------------------------------------------------------------------
# GP predictor using banded solver.
# ----------------------------------------------------------------------
def gp_predict_banded(
    x_train: List[float], y_train: List[float],
    x_test: List[float],
    ell: float, sigma_f: float, sigma_n: float, mu: int,
) -> Tuple[List[float], List[float]]:
    """GP posterior mean and std at test points, using banded Cholesky.

    x_train must be sorted in ascending order.
    """
    n = len(x_train)
    if n != len(y_train):
        raise ValueError("gp_predict_banded: train length mismatch.")
    if n == 0:
        return [0.0] * len(x_test), [sigma_f] * len(x_test)
    # Assemble K + sigma_n^2 I.
    AB = gp_covariance_banded(x_train, ell, sigma_f, sigma_n, mu)
    # Cholesky factor.
    AB = _cholesky_band_textbook(n, mu, AB)
    # Solve alpha = (K + sigma_n^2 I)^{-1} y.
    alpha = banded_sl(n, mu, AB, y_train)
    # Posterior mean and variance at each test point.
    mu_out: List[float] = []
    var_out: List[float] = []
    for xt in x_test:
        # Build k_star = [k(x_t, x_i)]_i.
        k_star = [matern32(xt - xi, ell, sigma_f) for xi in x_train]
        m = sum(k_star[i] * alpha[i] for i in range(n))
        # Solve K v = k_star.
        v = banded_sl(n, mu, AB, k_star)
        vKv = sum(k_star[i] * v[i] for i in range(n))
        var = matern32(0.0, ell, sigma_f) - vKv
        var = max(var, 1.0e-14)
        mu_out.append(m)
        var_out.append(var)
    std_out = [math.sqrt(v) for v in var_out]
    return mu_out, std_out
