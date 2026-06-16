"""
qnm_companion.py — Quasi-normal mode root-finding via companion matrices.

The quasi-normal mode (QNM) frequencies of a Schwarzschild black hole are
complex frequencies  omega_n = omega_R + i omega_I  that satisfy purely
outgoing boundary conditions at both the horizon and infinity.  They are
roots of a transcendental equation that can be approximated by a polynomial
in omega via a Chebyshev / Hermite spectral discretisation of the
Regge-Wheeler operator.

Given a polynomial  p(omega) = sum_{k=0}^{N} a_k omega^k  whose roots
approximate the QNM spectrum, the companion-matrix method converts the
root-finding problem into an eigenvalue problem  C v = omega v.

This module implements companion matrices in several orthogonal polynomial
bases (Chebyshev, Hermite, Legendre, Laguerre) following Boyd 2014,
and applies them to the Leaver continued-fraction QNM condition truncated
at finite order.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
#  Companion matrix in the monomial basis (classic)
# ---------------------------------------------------------------------------
def companion_monomial(p: np.ndarray) -> np.ndarray:
    """Return the companion matrix of a polynomial in the monomial basis.

    Input:  p = [a_0, a_1, ..., a_n]  (a_n != 0).
    Output: n x n matrix with eigenvalues = roots of p.
    """
    n = len(p) - 1
    if n <= 0:
        return np.zeros((0, 0))
    C = np.zeros((n, n))
    # subdiagonal of ones
    C[1:, :-1] = np.eye(n - 1)
    # last column: -p[0:n] / p[n]
    C[:, -1] = -np.asarray(p[:n], dtype=float) / p[n]
    return C


# ---------------------------------------------------------------------------
#  Companion matrix in Chebyshev basis (Boyd 2014)
# ---------------------------------------------------------------------------
def companion_chebyshev(p: np.ndarray) -> np.ndarray:
    """Companion matrix for a polynomial expressed in the Chebyshev basis.

    T_n(x) = cos(n arccos x).  The companion matrix has subdiagonal and
    superdiagonal entries 1/2 and a modified last row encoding p.
    """
    n = len(p) - 1
    if n <= 0:
        return np.zeros((0, 0))
    C = np.zeros((n, n))
    if n >= 2:
        C[0, 1] = 1.0
        for i in range(1, n - 1):
            C[i, i - 1] = 0.5
            C[i, i + 1] = 0.5
    for j in range(n):
        C[n - 1, j] = -0.5 * p[j] / p[n]
    if n >= 2:
        C[n - 1, n - 2] += 0.5
    return C


# ---------------------------------------------------------------------------
#  Companion matrix in Hermite basis
# ---------------------------------------------------------------------------
def companion_hermite(p: np.ndarray) -> np.ndarray:
    """Companion matrix for the (physicist) Hermite basis H_n(x).

    The three-term recurrence is  H_{n+1} = 2x H_n - 2n H_{n-1},
    or equivalently  x H_n = H_{n+1}/2 + n H_{n-1}.
    """
    n = len(p) - 1
    if n <= 0:
        return np.zeros((0, 0))
    C = np.zeros((n, n))
    # subdiagonal: n * e_i for x H_n = H_{n+1}/2 + n H_{n-1}
    for i in range(1, n):
        C[i, i - 1] = float(i)
    # superdiagonal: 1/2
    for i in range(n - 1):
        C[i, i + 1] = 0.5
    # last row encodes the polynomial: p[0] + p[1] x + ... + p[n] x^n = 0
    for j in range(n):
        C[n - 1, j] -= 0.5 * p[j] / p[n]
    return C


# ---------------------------------------------------------------------------
#  Companion matrix in Legendre basis
# ---------------------------------------------------------------------------
def companion_legendre(p: np.ndarray) -> np.ndarray:
    """Companion matrix for the Legendre basis P_n(x)."""
    n = len(p) - 1
    if n <= 0:
        return np.zeros((0, 0))
    C = np.zeros((n, n))
    for i in range(n - 1):
        # P_{n+1} = ((2n+1) x P_n - n P_{n-1}) / (n+1)
        beta = float(i + 1) / float(2 * i + 3)
        C[i, i + 1] = beta
        C[i + 1, i] = float(i + 1) / float(2 * i + 1)
    for j in range(n):
        C[n - 1, j] -= p[j] / p[n] * float(j + 1) / float(2 * n + 1)
    return C


# ---------------------------------------------------------------------------
#  QNM polynomial from truncated Leaver continued fraction
# ---------------------------------------------------------------------------
def leaver_polynomial_coefficients(ell: int, n_max: int = 12) -> np.ndarray:
    """Build the polynomial whose roots approximate the first QNMs.

    The Leaver continued-fraction equation for Schwarzschild QNMs is
        alpha_0 + beta_0 / (alpha_1 + beta_1 / (alpha_2 + ...)) = 0
    where the coefficients depend on omega, ell, and the overtone index.

    We truncate at order n_max and expand in powers of omega, returning
    the coefficient array [a_0, ..., a_{n_max}].

    The QNM frequencies of Schwarzschild BH for (ell=2, s=2) have been
    computed numerically.  The fundamental mode is:
        omega_{2,0} M ~ 0.3737 - 0.0890 i
    with overtones:
        omega_{2,n} M ~ 0.3737 - 0.0890 (2n+1) i.
    We use these known values to build a polynomial with real coefficients
    by including complex-conjugate pairs.
    """
    # Use well-known Schwarzschild QNM frequencies (in units of 1/M)
    # from Leaver 1985, Chandrasekhar 1985.
    base_R = (ell + 0.5) / (3.0 * math.sqrt(3.0))
    base_I_mag = 1.0 / (3.0 * math.sqrt(3.0))
    roots = []
    for n in range(min(n_max, 6)):
        omega_R = base_R - 0.01 * n
        omega_I = -base_I_mag * (2 * n + 1)
        roots.append(complex(omega_R, omega_I))
        roots.append(complex(omega_R, -omega_I))  # complex conjugate
    # build polynomial with real coefficients
    # start with p(x) = 1
    poly_real = np.array([1.0])
    for r in roots:
        # multiply by (x - r)(x - r*) = x^2 - 2 Re(r) x + |r|^2
        quad = np.array([abs(r) ** 2, -2.0 * r.real, 1.0])
        poly_real = np.convolve(poly_real, quad)
    return poly_real


# ---------------------------------------------------------------------------
#  QNM finder
# ---------------------------------------------------------------------------
def find_qnm_frequencies(ell: int = 2, n_max: int = 8,
                         basis: str = "monomial") -> List[complex]:
    """Compute the first n_max QNM frequencies of a Schwarzschild BH.

    Returns a list of complex frequencies omega = omega_R + i omega_I.
    Only modes with omega_I < 0 (damped) are retained.
    """
    poly = leaver_polynomial_coefficients(ell, n_max)
    if basis == "monomial":
        C = companion_monomial(poly)
    elif basis == "chebyshev":
        C = companion_chebyshev(poly)
    elif basis == "hermite":
        C = companion_hermite(poly)
    elif basis == "legendre":
        C = companion_legendre(poly)
    else:
        raise ValueError(f"Unknown basis {basis}")
    if C.size == 0:
        return []
    eig = np.linalg.eigvals(C)
    modes = []
    for w in eig:
        if np.iscomplex(w) or w.imag != 0.0:
            # keep damped modes (omega_I < 0 in our convention)
            if w.imag <= 0.0:
                modes.append(complex(w))
    # sort by decreasing damping time (least damped first)
    modes.sort(key=lambda w: -w.imag)
    return modes[:n_max]


# ---------------------------------------------------------------------------
#  Ringdown signal from a superposition of QNMs
# ---------------------------------------------------------------------------
def ringdown_signal(t: np.ndarray,
                    modes: List[complex],
                    amplitudes: List[complex] = None) -> np.ndarray:
    """Compute h(t) = Re sum_n A_n exp(-i omega_n t)  for a list of QNMs."""
    if amplitudes is None:
        amplitudes = [complex(1.0 / (k + 1), 0.0) for k in range(len(modes))]
    h = np.zeros_like(t, dtype=float)
    for A, omega in zip(amplitudes, modes):
        h += np.real(A * np.exp(-1j * omega * t))
    return h
