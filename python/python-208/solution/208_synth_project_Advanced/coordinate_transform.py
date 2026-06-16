"""
coordinate_transform.py
=======================

Parameter-space coordinate transformations for the multi-fidelity UQ
problem.  Adapted from the `ge_to_ccs` sparse matrix format conversion
of project 457: we convert between a "general" dense parameter vector
xi in R^d and a "compressed column" style intrinsic parameterization
eta in R^d that is aligned with the correlation structure of the GP.

The purpose
-----------
The raw inputs to the protoplanetary-disk simulator are:
    xi = [log10(alpha), log10(St), log10(Z_0), zeta]
which have very different scales and correlations.  We apply an affine
transformation

    eta = L^{-1} (xi - mu_xi)

where mu_xi is the prior mean and L is the lower-Cholesky factor of the
prior covariance Sigma_xi.  This "whitens" the input space so that the
GP kernel hyperparameters can be set to a default value (ell = 1.0,
sigma_f = 1.0) without re-tuning.

Additionally, we provide a sparse CCS-style compression:
    xi_ccs = (colptr, rowind, values)
for high-dimensional parameter vectors where most entries are zero
(e.g. when embedding a low-dimensional physical parameter vector into
a higher-dimensional polynomial chaos space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Affine whitening transform.
# ----------------------------------------------------------------------
@dataclass
class WhiteningTransform:
    """Stores the whitening transform parameters."""
    mu: List[float]              # prior mean
    L: List[List[float]]         # lower-Cholesky factor of prior cov
    Linv: List[List[float]]      # inverse of L (for encoding)


def cholesky_lower(S: List[List[float]]) -> List[List[float]]:
    """Lower-Cholesky factor L of SPD matrix S (S = L L^T)."""
    n = len(S)
    L = [[0.0] * n for _ in range(n)]
    for j in range(n):
        s = S[j][j]
        for k in range(j):
            s -= L[j][k] ** 2
        if s <= 0.0:
            s = max(s, 1.0e-14)
        L[j][j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = S[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            L[i][j] = s / L[j][j]
    return L


def invert_lower_triangular(L: List[List[float]]) -> List[List[float]]:
    """Invert a lower-triangular matrix L by forward substitution."""
    n = len(L)
    Linv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        if abs(L[j][j]) < 1.0e-30:
            raise ValueError("invert_lower_triangular: zero diagonal.")
        Linv[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j, i):
                s += L[i][k] * Linv[k][j]
            Linv[i][j] = -s / L[i][i]
    return Linv


def build_whitening(xi_samples: List[List[float]]) -> WhiteningTransform:
    """Build a whitening transform from a set of parameter samples."""
    n = len(xi_samples)
    if n < 2:
        raise ValueError("build_whitening: need >= 2 samples.")
    d = len(xi_samples[0])
    mu = [sum(xi[i] for xi in xi_samples) / n for i in range(d)]
    # Sample covariance.
    S = [[0.0] * d for _ in range(d)]
    for xi in xi_samples:
        for i in range(d):
            for j in range(d):
                S[i][j] += (xi[i] - mu[i]) * (xi[j] - mu[j])
    for i in range(d):
        for j in range(d):
            S[i][j] /= (n - 1)
    # Robust regularization: floor the diagonal at a minimum variance
    # equal to 1% of the max diagonal, plus a small absolute floor.
    max_diag = max(S[i][i] for i in range(d))
    min_diag = max(0.01 * max_diag, 1.0e-4)
    for i in range(d):
        if S[i][i] < min_diag:
            S[i][i] = min_diag
        S[i][i] += 1.0e-6
    L = cholesky_lower(S)
    Linv = invert_lower_triangular(L)
    return WhiteningTransform(mu=mu, L=L, Linv=Linv)


def encode(xi: List[float], W: WhiteningTransform) -> List[float]:
    """Encode xi -> eta via eta = Linv (xi - mu)."""
    d = len(xi)
    eta = [0.0] * d
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += W.Linv[i][j] * (xi[j] - W.mu[j])
        eta[i] = s
    return eta


def decode(eta: List[float], W: WhiteningTransform) -> List[float]:
    """Decode eta -> xi via xi = L eta + mu."""
    d = len(eta)
    xi = [0.0] * d
    for i in range(d):
        s = W.mu[i]
        for j in range(d):
            s += W.L[i][j] * eta[j]
        xi[i] = s
    return xi


# ----------------------------------------------------------------------
# GE-to-CCS sparse compression (following project 457).
# ----------------------------------------------------------------------
def ge_to_ccs(A: List[List[float]]) -> Tuple[int, List[int], List[int], List[float]]:
    """Convert a dense matrix A (m x n) to compressed column storage.

    Returns (nz_num, colptr, rowind, Accs) such that:
      - colptr[j] is the index in rowind/Accs of the first nonzero in column j,
      - colptr[n] = nz_num,
      - rowind[k] is the row index of the k-th nonzero,
      - Accs[k] is its value.
    """
    m = len(A)
    if m == 0:
        return 0, [0], [], []
    n = len(A[0])
    colptr = [0] * (n + 1)
    rowind_list: List[int] = []
    Accs: List[float] = []
    for j in range(n):
        colptr[j] = len(rowind_list)
        for i in range(m):
            if abs(A[i][j]) > 1.0e-30:
                rowind_list.append(i)
                Accs.append(A[i][j])
    colptr[n] = len(rowind_list)
    return len(Accs), colptr, rowind_list, Accs


def ccs_to_ge(
    nz_num: int, colptr: List[int], rowind: List[int], Accs: List[float],
    m: int, n: int,
) -> List[List[float]]:
    """Inverse: reconstruct the dense m x n matrix from CCS."""
    A = [[0.0] * n for _ in range(m)]
    for j in range(n):
        for k in range(colptr[j], colptr[j + 1]):
            A[rowind[k]][j] = Accs[k]
    return A


# ----------------------------------------------------------------------
# Parameter-space volume element (Jacobian of whitening transform).
# ----------------------------------------------------------------------
def whitening_log_jacobian(W: WhiteningTransform) -> float:
    """Log |det L|, the log-volume element of the whitening transform."""
    d = len(W.mu)
    log_det = 0.0
    for i in range(d):
        if W.L[i][i] <= 0.0:
            raise ValueError("whitening_log_jacobian: non-positive diagonal.")
        log_det += math.log(W.L[i][i])
    return log_det


# ----------------------------------------------------------------------
# Prior-density evaluation (multivariate normal in xi, via whitened eta).
# ----------------------------------------------------------------------
def log_prior_pdf(xi: List[float], W: WhiteningTransform) -> float:
    """Evaluate log p(xi) for xi ~ N(mu, L L^T)."""
    eta = encode(xi, W)
    d = len(eta)
    log_det = whitening_log_jacobian(W)
    quad = sum(e * e for e in eta)
    return -0.5 * (d * math.log(2.0 * math.pi) + 2.0 * log_det + quad)
