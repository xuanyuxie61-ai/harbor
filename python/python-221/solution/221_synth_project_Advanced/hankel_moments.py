"""
hankel_moments.py
=================

Hankel matrix methods for moment-space resummation and Cholesky
factorization:
- Hankel SPD Cholesky factorization (from 504_hankel_cholesky)

Scientific context:
-------------------
In moment space, the DGLAP evolution becomes multiplicative:

    q(N, mu^2) = q(N, mu0^2) * exp(int alpha_s/(2*pi) gamma_N dt)

The Mellin moments of the PDF are:
    q(N) = int_0^1 x^{N-1} q(x) dx

The moment matrix M_{ij} = q(i+j-2) forms a Hankel matrix (constant
along antidiagonals). For a positive-definite PDF, this Hankel matrix
is SPD and can be Cholesky-factored:

    M = L * L^T

The Cholesky factor L has a special structure: its elements satisfy
a recurrence relation that ensures the Hankel property is maintained.

Reference: Al-Homidan & Alshahrani, "Positive Definite Hankel Matrices
Using Cholesky Factorization", Comput. Methods Appl. Math. 9 (2009) 221.
"""

import math
from typing import List, Tuple


# ===========================================================================
# Section 1: Hankel Cholesky factorization (from 504_hankel_cholesky)
# ===========================================================================

def hankel_spd_cholesky_lower(n: int, lii: List[float],
                              liim1: List[float]) -> List[List[float]]:
    """
    Construct the lower Cholesky factor L such that H = L * L^T is
    a Hankel SPD matrix.

    The Hankel matrix H has the property H[i+j] = h(k-1) for
    1 <= i, j <= n, where k = i+j-1.

    The Cholesky factor L is constructed by specifying:
        L[i, i] = lii[i]      for 1 <= i <= n
        L[i+1, i] = liim1[i]  for 1 <= i <= n-1

    and the remaining elements are determined by the recurrence:

        L[i, j] = (alpha - beta) / L[j, j]

    where:
        alpha = sum_{s=1}^{q} L[q, s] * L[r, s]
        beta = sum_{t=1}^{j-1} L[i, t] * L[j, t]

    and q = (i+j)/2, r = (i+j)/2 if i+j even,
        q = (i+j-1)/2, r = q+1 if i+j odd.

    Parameters
    ----------
    n : int
        Order of the matrix.
    lii : list of float
        Diagonal elements L[i, i], length n.
    liim1 : list of float
        Subdiagonal elements L[i+1, i], length n-1.

    Returns
    -------
    list of lists: lower triangular Cholesky factor L (n x n).
    """
    if n < 1:
        raise ValueError(f"hankel_spd_cholesky_lower: n={n} < 1")
    if len(lii) != n:
        raise ValueError(f"hankel_spd_cholesky_lower: len(lii)={len(lii)} != n={n}")
    if len(liim1) != n - 1:
        raise ValueError(f"hankel_spd_cholesky_lower: len(liim1)={len(liim1)} != n-1={n-1}")
    L = [[0.0] * n for _ in range(n)]
    # Set diagonal
    for i in range(n):
        L[i][i] = lii[i]
    # Set subdiagonal
    for i in range(n - 1):
        L[i + 1][i] = liim1[i]
    # Fill remaining elements using the recurrence
    for i in range(2, n):
        for j in range(i - 1):
            # Determine q, r
            if (i + j) % 2 == 0:
                q = (i + j) // 2
                r = q
            else:
                q = (i + j - 1) // 2
                r = q + 1
            # Compute alpha
            alpha = 0.0
            for s in range(q + 1):
                alpha += L[q][s] * L[r][s]
            # Compute beta
            beta = 0.0
            for t in range(j):
                beta += L[i][t] * L[j][t]
            # L[i, j]
            if abs(L[j][j]) < 1e-30:
                L[i][j] = 0.0
            else:
                L[i][j] = (alpha - beta) / L[j][j]
            # Symmetry: L[j, i] = 0 (lower triangular)
    return L


def hankel_from_cholesky(L: List[List[float]]) -> List[List[float]]:
    """
    Compute H = L * L^T from the Cholesky factor.
    """
    n = len(L)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j + 1):
                s += L[i][k] * L[j][k]
            H[i][j] = s
            H[j][i] = s
    return H


def check_hankel_property(H: List[List[float]], tol: float = 1e-8) -> bool:
    """
    Check if a matrix H has the Hankel property:
        H[i, j] = h[i+j] for some sequence h.

    This means H is constant along anti-diagonals.
    """
    n = len(H)
    if n < 2:
        return True
    for k in range(2 * n - 1):
        # All (i, j) with i+j = k should have the same value
        val = None
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                if val is None:
                    val = H[i][j]
                elif abs(H[i][j] - val) > tol:
                    return False
    return True


def moment_hankel_matrix(moments: List[float], n: int) -> List[List[float]]:
    """
    Build the Hankel matrix of moments:

        H[i, j] = moments[i + j]

    for 0 <= i, j < n.

    In the resummation context, moments[i] = q(N=i+1) are the
    Mellin moments of the parton distribution function.
    """
    if len(moments) < 2 * n - 1:
        raise ValueError(f"moment_hankel_matrix: need {2*n-1} moments, got {len(moments)}")
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            H[i][j] = moments[i + j]
    return H


def cholesky_decompose_general(A: List[List[float]]) -> List[List[float]]:
    """
    Standard Cholesky decomposition A = L * L^T for SPD matrix A.

    Uses the standard algorithm:
        L[i, i] = sqrt(A[i, i] - sum_{k<i} L[i, k]^2)
        L[j, i] = (A[j, i] - sum_{k<i} L[j, k] * L[i, k]) / L[i, i]
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = sum(L[i][k] ** 2 for k in range(i))
        diag = A[i][i] - s
        if diag <= 0.0:
            diag = 1e-10  # regularization for near-singular
        L[i][i] = math.sqrt(diag)
        for j in range(i + 1, n):
            s = sum(L[j][k] * L[i][k] for k in range(i))
            if abs(L[i][i]) < 1e-30:
                L[j][i] = 0.0
            else:
                L[j][i] = (A[j][i] - s) / L[i][i]
    return L


def resummation_exponent(moment_n: float, alpha_s: float,
                         a_coeff: float = 1.0, b_coeff: float = 0.0
                         ) -> float:
    """
    Compute the resummed exponent for threshold resummation at NLL:

        g_1(alpha_s * ln(N)) = a_coeff / (alpha_s * b_coeff)
                                * [2*ln(1 - 2*alpha_s*b_coeff*ln(N))
                                   + ln(1 - 2*alpha_s*b_coeff*ln(N))]

    Simplified form for the leading-log tower:
        exp_g1 = a_coeff * alpha_s * ln(N)^2 / 2

    This is used in the exponent of the resummed cross-section:
        sigma_N ~ exp(g_1(lambda) + alpha_s * g_2(lambda) + ...)
    where lambda = alpha_s * beta0 * ln(N).
    """
    if moment_n < 1.0:
        moment_n = 1.0
    ln_n = math.log(moment_n)
    # Leading log
    g1 = a_coeff * alpha_s * ln_n * ln_n / 2.0
    # Next-to-leading log
    g2 = b_coeff * alpha_s * ln_n
    return g1 + g2
