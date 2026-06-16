"""
chebyshev_interp.py - Chebyshev interpolation for rate tables.

Adapted from 591_interp_chebyshev.  Builds Chebyshev nodes and interpolates
nuclear rate tables on [a, b] with exponential convergence.

  x_k = 0.5 (a + b) + 0.5 (b - a) cos((2k+1) pi / (2N))   for k = 0..N-1
  f(x) ~ sum_{k=0}^{N-1} c_k T_k(x')   where x' = (2x - a - b)/(b - a)
  c_k = (2/N) sum_{j=0}^{N-1} f(x_j) T_k(x_j)   (with c_0 halved)
"""
from __future__ import annotations
import math
from typing import List, Tuple

def chebyshev_nodes(a: float, b: float, N: int) -> List[float]:
    """Return N Chebyshev nodes on [a, b]."""
    nodes = []
    for k in range(N):
        xk = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * k + 1) * math.pi / (2 * N))
        nodes.append(xk)
    return nodes


def chebyshev_T(n: int, x: float) -> float:
    """Evaluate T_n(x) = cos(n arccos x) via recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return x
    T0, T1 = 1.0, x
    for _ in range(2, n + 1):
        T0, T1 = T1, 2.0 * x * T1 - T0
    return T1


def chebyshev_coefficients(f_values: List[float], N: int) -> List[float]:
    """
    Compute Chebyshev coefficients c_k from function values at Chebyshev nodes.
    c_k = (2/N) sum_{j=0}^{N-1} f(x_j) T_k(x_j)   with c_0 halved.
    """
    coeffs = []
    for k in range(N):
        s = 0.0
        for j in range(N):
            xj = math.cos((2 * j + 1) * math.pi / (2 * N))
            s += f_values[j] * chebyshev_T(k, xj)
        ck = (2.0 / N) * s
        if k == 0:
            ck *= 0.5
        coeffs.append(ck)
    return coeffs


def chebyshev_eval(coeffs: List[float], a: float, b: float, x: float) -> float:
    """
    Evaluate the Chebyshev expansion at x in [a, b].
    Map x -> x' in [-1, 1], then  f(x) ~ sum c_k T_k(x').
    """
    if b == a:
        return coeffs[0] if coeffs else 0.0
    xp = (2.0 * x - a - b) / (b - a)
    xp = max(-1.0, min(1.0, xp))
    result = 0.0
    for k, ck in enumerate(coeffs):
        result += ck * chebyshev_T(k, xp)
    return result


def chebyshev_interpolate(func, a: float, b: float, N: int) -> Tuple[List[float], float, float]:
    """
    Build Chebyshev interpolant of func on [a, b] with N nodes.
    Returns (coeffs, a, b).
    """
    nodes = chebyshev_nodes(a, b, N)
    f_values = [func(x) for x in nodes]
    coeffs = chebyshev_coefficients(f_values, N)
    return coeffs, a, b


def chebyshev_derivative_coeffs(coeffs: List[float]) -> List[float]:
    """
    Derivative coefficients:  c'_k = 2 (k+1) c_{k+1} + c'_{k+2}  (Clenshaw).
    Returns list of same length (padded with 0).
    """
    N = len(coeffs)
    dcoeffs = [0.0] * N
    if N <= 1:
        return dcoeffs
    dcoeffs[N - 1] = 0.0
    if N >= 2:
        dcoeffs[N - 2] = 2.0 * (N - 1) * coeffs[N - 1]
    for k in range(N - 3, -1, -1):
        dcoeffs[k] = dcoeffs[k + 2] + 2.0 * (k + 1) * coeffs[k + 1]
    return dcoeffs


__all__ = [
    "chebyshev_nodes", "chebyshev_T",
    "chebyshev_coefficients", "chebyshev_eval",
    "chebyshev_interpolate", "chebyshev_derivative_coeffs",
]
