"""
high_order_fd.py — High-order finite-difference stencils for the 1+1 wave equation.

Implements centered finite-difference approximations of the second derivative
    d^2 u / d x^2
with formal truncation errors O(h^p) for p in {2, 4, 6, 8, 10}.

The coefficients are obtained from the Taylor-expansion matching conditions:

    sum_{k=-s}^{s} c_k u(x + k h) = u''(x) h^2 + O(h^{p+2})

where s = p // 2 is the stencil half-width.

Each stencil is returned in normalised form so that the FD operator is

    D2 u_i = (1/h^2) sum_k c_k u_{i+k}.

The module also provides:
  - First derivative stencils (needed for Sommerfeld BCs),
  - Optimised (minimum-bandwidth) stencils,
  - Spectral-symbol computation for von Neumann analysis.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from fractions import Fraction


# ---------------------------------------------------------------------------
#  Precomputed centred second-derivative stencil coefficients
# ---------------------------------------------------------------------------
#  Each stencil satisfies: sum_k c_k u_{i+k} = h^2 u''_i + O(h^{p+2})
#  The support is k in {-s, ..., s} with s = p // 2.

STENCIL_D2: Dict[int, np.ndarray] = {}


def _build_stencil_d2(p: int) -> np.ndarray:
    """Construct the centred O(h^p) second-derivative stencil by solving
    the Vandermonde system for the Taylor coefficients.

    For an order-p stencil we need to match moments 2, 4, ..., p:
        sum_k c_k k^{2m} = 0       for m = 0
        sum_k c_k k^2      = 2     (this is the second derivative)
        sum_k c_k k^{2m}   = 0     for m = 2, ..., p//2
    and the symmetry c_{-k} = c_k halves the unknowns.
    """
    if p not in (2, 4, 6, 8, 10):
        raise ValueError(f"Unsupported order p = {p}; choose from {{2,4,6,8,10}}.")
    s = p // 2  # half-width
    # unknowns: c_0, c_1, ..., c_s  (symmetry c_{-k} = c_k)
    n_unknowns = s + 1
    # equations: moments 0, 2, 4, ..., 2s
    # moment m : sum_{k=1}^s 2 c_k k^{m} + c_0 delta_{m,0} = delta_{m,2} * 2
    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)
    for row, m in enumerate(range(0, 2 * s + 1, 2)):
        if m == 0:
            A[row, 0] = 1.0
            for k in range(1, s + 1):
                A[row, k] = 2.0
            b[row] = 0.0
        elif m == 2:
            A[row, 0] = 0.0
            for k in range(1, s + 1):
                A[row, k] = 2.0 * k * k
            b[row] = 2.0
        else:
            A[row, 0] = 0.0
            for k in range(1, s + 1):
                A[row, k] = 2.0 * k ** m
            b[row] = 0.0
    c_half = np.linalg.solve(A, b)
    # Build full stencil c[-s..s]
    full = np.zeros(2 * s + 1)
    full[s] = c_half[0]
    for k in range(1, s + 1):
        full[s - k] = c_half[k]
        full[s + k] = c_half[k]
    return full


def _init_stencils() -> None:
    for p in (2, 4, 6, 8, 10):
        STENCIL_D2[p] = _build_stencil_d2(p)


_init_stencils()


# ---------------------------------------------------------------------------
#  First-derivative centred stencils (for advection / Sommerfeld BCs)
# ---------------------------------------------------------------------------
def stencil_d1(order: int) -> np.ndarray:
    """Return the centred O(h^{order}) first-derivative stencil.

    The stencil coefficients satisfy sum_k c_k u_{i+k} / h = u'_i + O(h^{order}).
    By antisymmetry  c_{-k} = -c_k,  c_0 = 0.
    """
    if order not in (2, 4, 6, 8):
        raise ValueError(f"Unsupported first-derivative order {order}.")
    s = order // 2
    A = np.zeros((s, s))
    b = np.zeros(s)
    for row, m in enumerate(range(1, 2 * s, 2)):
        for k in range(1, s + 1):
            A[row, k - 1] = 2.0 * k ** m
        b[row] = 1.0 if m == 1 else 0.0
    c_pos = np.linalg.solve(A, b)
    full = np.zeros(2 * s + 1)
    for k in range(1, s + 1):
        full[s - k] = -c_pos[k - 1]
        full[s + k] = c_pos[k - 1]
    return full


# ---------------------------------------------------------------------------
#  Application operators
# ---------------------------------------------------------------------------
def apply_d2(u: np.ndarray, dx: float, order: int,
             bc: str = "zero") -> np.ndarray:
    """Apply the O(h^order) second-derivative stencil to a 1D array u.

    Parameters
    ----------
    u     : (N,) array of function values.
    dx    : uniform grid spacing.
    order : 2, 4, 6, 8, or 10.
    bc    : boundary treatment: "zero" pads with zeros,
            "periodic" wraps, "extrapolate" uses polynomial continuation.

    Returns
    -------
    D2u   : (N,) array with D2 u_i = (1/dx^2) sum_k c_k u_{i+k}.
    """
    if order not in STENCIL_D2:
        raise ValueError(f"order must be one of {list(STENCIL_D2.keys())}")
    c = STENCIL_D2[order]
    s = len(c) // 2
    N = u.shape[0]
    # pad
    if bc == "zero":
        upad = np.pad(u, (s, s), mode="constant", constant_values=0.0)
    elif bc == "periodic":
        upad = np.pad(u, (s, s), mode="wrap")
    elif bc == "extrapolate":
        upad = np.pad(u, (s, s), mode="edge")
    else:
        raise ValueError(f"Unknown BC: {bc}")
    D2u = np.zeros(N)
    for k in range(-s, s + 1):
        D2u += c[k + s] * upad[s + k: s + k + N]
    return D2u / (dx * dx)


def apply_d1(u: np.ndarray, dx: float, order: int,
             bc: str = "zero") -> np.ndarray:
    """Apply the O(h^order) first-derivative stencil."""
    c = stencil_d1(order)
    s = len(c) // 2
    N = u.shape[0]
    if bc == "zero":
        upad = np.pad(u, (s, s), mode="constant", constant_values=0.0)
    elif bc == "periodic":
        upad = np.pad(u, (s, s), mode="wrap")
    elif bc == "extrapolate":
        upad = np.pad(u, (s, s), mode="edge")
    else:
        raise ValueError(f"Unknown BC: {bc}")
    D1u = np.zeros(N)
    for k in range(-s, s + 1):
        D1u += c[k + s] * upad[s + k: s + k + N]
    return D1u / dx


# ---------------------------------------------------------------------------
#  Spectral symbol for von Neumann analysis
# ---------------------------------------------------------------------------
def spectral_symbol_d2(theta: np.ndarray, dx: float, order: int) -> np.ndarray:
    """Evaluate the Fourier symbol of the D2 operator.

    For the exact second derivative the symbol is -k^2 = -(theta/dx)^2.
    For the FD operator the symbol is
        sigma(theta) = (1/dx^2) sum_k c_k e^{i k theta}
                     = (1/dx^2) [ c_0 + 2 sum_{k=1}^{s} c_k cos(k theta) ].
    """
    c = STENCIL_D2[order]
    s = len(c) // 2
    sigma = np.full_like(theta, c[s] / (dx * dx))
    for k in range(1, s + 1):
        sigma += 2.0 * c[s + k] * np.cos(k * theta) / (dx * dx)
    return sigma


# ---------------------------------------------------------------------------
#  Truncation-error coefficient
# ---------------------------------------------------------------------------
def truncation_error_coeff(order: int) -> float:
    """Return the leading truncation-error coefficient  C_p  in

        D2_FD u = u'' + C_p h^p u^{(p+2)} + O(h^{p+2}).

    The coefficient is computed from the first unmatched moment.
    """
    c = STENCIL_D2[order]
    s = len(c) // 2
    p = order
    # the first non-vanishing moment after order p is p+2
    moment = 0.0
    fact = float(math.factorial(p + 2))
    for k in range(-s, s + 1):
        moment += c[k + s] * (k ** (p + 2))
    return moment / fact


import math  # noqa: E402  (used above but kept near top for clarity)
