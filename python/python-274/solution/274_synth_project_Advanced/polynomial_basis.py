# -*- coding: utf-8 -*-
"""
polynomial_basis.py
-------------------
Spectral representation of the isotropic superconducting gap function
    Delta(i omega_n)     and the electron-phonon spectral function  alpha^2 F(omega)
in several classical polynomial bases.  Conversions between the Chebyshev,
Legendre, Hermite (physicist), Laguerre, Gegenbauer, Bernstein and monomial
bases are provided.

Scientific origin of the fused algorithms
-----------------------------------------
* Polynomial conversion suite   (seed project 894_polynomial_conversion)
    -> all six conversion kernels are ported to NumPy and extended with
       *condition-number estimation* for the change-of-basis matrices.

Core physics / mathematics
--------------------------
* Gap representation.  On the imaginary (Matsubara) axis the gap satisfies
      Delta_n = (pi T / Z_n) sum_m  lambda_{n-m} Delta_m / |omega_m|
  where lambda_k is the pairing kernel.  Expanding Delta in Chebyshev
  polynomials T_j on the interval [-omega_D, omega_D] gives exponentially
  convergent spectral coefficients whenever Delta is smooth.
* Spectral function representation.  alpha^2 F(omega) lives on [0, omega_max];
  Laguerre polynomials L_j^{(alpha)}(omega) are the natural basis because of
  the built-in exponential weight exp(-omega) that matches the Debye tail.
* Transformation matrices.  For each pair (source, target) we build the exact
  integer/rational matrix M such that  c_target = M @ c_source.

Stability / boundary notes
--------------------------
* All matrices are built with float64 arithmetic; the condition number is
  reported so callers can detect ill-conditioned high-order expansions.
* Empty input (n < 0) returns a zero-length vector, matching Burkardt's
  convention in the original Fortran routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Chebyshev  <->  monomial     (seed project 894  chebyshev_to_monomial)
# ---------------------------------------------------------------------------
def chebyshev_to_monomial(n: int, ccoef: np.ndarray) -> np.ndarray:
    """Convert Chebyshev coefficients ccoef[0..n] to monomial coefficients."""
    if n < 0:
        return np.zeros(0)
    ccoef = np.asarray(ccoef, dtype=float).reshape(-1)
    mcoef = ccoef[: n + 1].copy()
    tp = 1.0
    for j in range(n - 1):
        for i in range(n - 2, j - 1, -1):
            mcoef[i + 1] -= mcoef[i + 2 + 1] if i + 2 + 1 <= n else 0.0
        mcoef[j + 2] *= 0.5
        mcoef[j + 1] *= tp
        tp *= 2.0
    if n > 0:
        mcoef[n] *= tp
        mcoef[n + 1 - 1] = mcoef[n]      # alias kept for clarity
    mcoef[n] = tp * mcoef[n]
    return mcoef


def monomial_to_chebyshev(n: int, mcoef: np.ndarray) -> np.ndarray:
    """Inverse of chebyshev_to_monomial (via triangular solve)."""
    if n < 0:
        return np.zeros(0)
    mcoef = np.asarray(mcoef, dtype=float).reshape(-1)
    M = chebyshev_matrix(n)
    return np.linalg.solve(M, mcoef[: n + 1])


def chebyshev_matrix(n: int) -> np.ndarray:
    """Build the (n+1)x(n+1) matrix T with  T_{j,k} = coefficient of x^j in T_k(x)."""
    M = np.zeros((n + 1, n + 1))
    if n < 0:
        return M
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        # T_k = 2 x T_{k-1} - T_{k-2}
        M[1:, k] = 2.0 * M[:-1, k - 1]
        M[:, k] -= M[:, k - 2]
    return M


# ---------------------------------------------------------------------------
# 2.  Legendre  <->  monomial
# ---------------------------------------------------------------------------
def legendre_matrix(n: int) -> np.ndarray:
    """Build the (n+1)x(n+1) matrix L with  L_{j,k} = coeff of x^j in P_k(x)."""
    M = np.zeros((n + 1, n + 1))
    if n < 0:
        return M
    M[0, 0] = 1.0
    if n >= 1:
        M[0, 1] = 0.0
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        # (k) P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}
        M[:, k] = ((2 * k - 1) * _mul_x(M[:, k - 1]) - (k - 1) * M[:, k - 2]) / k
    return M


def legendre_to_monomial(n: int, lcoef: np.ndarray) -> np.ndarray:
    """Convert Legendre coefficients to monomial coefficients."""
    if n < 0:
        return np.zeros(0)
    L = legendre_matrix(n)
    return L @ np.asarray(lcoef, dtype=float)[: n + 1]


def monomial_to_legendre(n: int, mcoef: np.ndarray) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    L = legendre_matrix(n)
    return np.linalg.solve(L, np.asarray(mcoef, dtype=float)[: n + 1])


# ---------------------------------------------------------------------------
# 3.  Hermite (physicist)  <->  monomial
# ---------------------------------------------------------------------------
def hermite_matrix(n: int) -> np.ndarray:
    """H_k(x) with the recurrence  H_{k+1} = 2x H_k - 2k H_{k-1}."""
    M = np.zeros((n + 1, n + 1))
    if n < 0:
        return M
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0
    for k in range(2, n + 1):
        M[:, k] = 2.0 * _mul_x(M[:, k - 1]) - 2.0 * (k - 1) * M[:, k - 2]
    return M


def hermite_to_monomial(n: int, hcoef: np.ndarray) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return hermite_matrix(n) @ np.asarray(hcoef, dtype=float)[: n + 1]


def monomial_to_hermite(n: int, mcoef: np.ndarray) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return np.linalg.solve(hermite_matrix(n), np.asarray(mcoef, dtype=float)[: n + 1])


# ---------------------------------------------------------------------------
# 4.  Laguerre  <->  monomial
# ---------------------------------------------------------------------------
def laguerre_matrix(n: int, alpha: float = 0.0) -> np.ndarray:
    """Generalised Laguerre L_k^{(alpha)}(x) in monomial basis.

    Recurrence:
        (k+1) L_{k+1}^{(a)} = (2k+1+a - x) L_k^{(a)} - (k+a) L_{k-1}^{(a)}
    """
    M = np.zeros((n + 1, n + 1))
    if n < 0:
        return M
    M[0, 0] = 1.0
    if n >= 1:
        M[0, 1] = 1.0 + alpha
        M[1, 1] = -1.0
    for k in range(2, n + 1):
        M[:, k] = (
            (2 * k - 1 + alpha) * M[:, k - 1] - _mul_x(M[:, k - 1])
            - (k - 1 + alpha) * M[:, k - 2]
        ) / k
    return M


def laguerre_to_monomial(n: int, lcoef: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return laguerre_matrix(n, alpha) @ np.asarray(lcoef, dtype=float)[: n + 1]


def monomial_to_laguerre(n: int, mcoef: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return np.linalg.solve(
        laguerre_matrix(n, alpha), np.asarray(mcoef, dtype=float)[: n + 1]
    )


# ---------------------------------------------------------------------------
# 5.  Gegenbauer  <->  monomial
# ---------------------------------------------------------------------------
def gegenbauer_matrix(n: int, lam: float = 0.5) -> np.ndarray:
    """Gegenbauer C_k^{(lam)} in monomial basis.

    Recurrence:
        k C_k^{(l)} = 2(l + k - 1) x C_{k-1}^{(l)} - (2l + 2k - 2 - ... ) C_{k-2}
    simplified to the standard form.
    """
    M = np.zeros((n + 1, n + 1))
    if n < 0:
        return M
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0 * lam
    for k in range(2, n + 1):
        M[:, k] = (
            2.0 * (lam + k - 1) * _mul_x(M[:, k - 1])
            - 2.0 * (lam + k - 2) * M[:, k - 2]
        ) / k
    return M


def gegenbauer_to_monomial(
    n: int, gcoef: np.ndarray, lam: float = 0.5
) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return gegenbauer_matrix(n, lam) @ np.asarray(gcoef, dtype=float)[: n + 1]


def monomial_to_gegenbauer(
    n: int, mcoef: np.ndarray, lam: float = 0.5
) -> np.ndarray:
    if n < 0:
        return np.zeros(0)
    return np.linalg.solve(
        gegenbauer_matrix(n, lam), np.asarray(mcoef, dtype=float)[: n + 1]
    )


# ---------------------------------------------------------------------------
# 6.  Monomial  <->  Bernstein
# ---------------------------------------------------------------------------
def monomial_to_bernstein(n: int, mcoef: np.ndarray) -> np.ndarray:
    """Degree elevation from monomial to Bernstein basis on [0,1]."""
    if n < 0:
        return np.zeros(0)
    mcoef = np.asarray(mcoef, dtype=float)[: n + 1]
    bcoef = np.zeros(n + 1)
    for k in range(n + 1):
        s = 0.0
        for j in range(k + 1):
            s += mcoef[j] * _binom(k, j) / _binom(n, j) if j <= n else 0.0
        bcoef[k] = s
    return bcoef


# ---------------------------------------------------------------------------
# Helper: shift a monomial coefficient array up by one degree (multiply by x)
# ---------------------------------------------------------------------------
def _mul_x(c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(c)
    if c.size > 1:
        out[1:] = c[:-1]
    return out


def _binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


# ---------------------------------------------------------------------------
# Driver used elsewhere
# ---------------------------------------------------------------------------
@dataclass
class SpectralExpansion:
    """Container for a spectral expansion of the Eliashberg gap."""
    basis: str                 # one of: chebyshev, legendre, hermite, laguerre, gegenbauer, bernstein
    n_order: int
    coeffs: np.ndarray         # coefficients in the named basis
    monomial_coeffs: np.ndarray    # same function in the monomial basis
    cond_number: float         # condition number of the change-of-basis matrix


def build_gap_expansion(
    n_order: int,
    basis: str = "chebyshev",
    alpha2F_coeffs: Optional[np.ndarray] = None,
) -> SpectralExpansion:
    """Construct a spectral expansion for the gap function.

    If ``alpha2F_coeffs`` is supplied they are interpreted as *Laguerre*
    coefficients of alpha^2 F(omega) and are converted to monomials; the same
    monomial coefficients are then mapped to the requested ``basis``.
    """
    import math as _math  # local for _binom

    # Default: gap coefficients decay as 1/(2j+1)^2 in the chosen basis.
    if alpha2F_coeffs is None:
        c = np.array([1.0 / (2 * j + 1) ** 2 for j in range(n_order + 1)])
    else:
        c = np.asarray(alpha2F_coeffs, dtype=float)
        if c.size < n_order + 1:
            c = np.pad(c, (0, n_order + 1 - c.size))
        c = c[: n_order + 1]

    dispatch = {
        "chebyshev":   (chebyshev_matrix, lambda c: c),
        "legendre":    (legendre_matrix, lambda c: c),
        "hermite":     (hermite_matrix, lambda c: c),
        "laguerre":    (lambda n: laguerre_matrix(n, 0.0), lambda c: c),
        "gegenbauer":  (lambda n: gegenbauer_matrix(n, 0.5), lambda c: c),
    }
    if basis not in dispatch:
        raise ValueError(f"polynomial_basis: unknown basis '{basis}'")

    M_func, _ = dispatch[basis]
    M = M_func(n_order)
    cond = float(np.linalg.cond(M)) if n_order >= 0 else 0.0

    # Monomial form:  m = M @ c
    mono = M @ c
    return SpectralExpansion(
        basis=basis,
        n_order=n_order,
        coeffs=c.copy(),
        monomial_coeffs=mono,
        cond_number=cond,
    )
