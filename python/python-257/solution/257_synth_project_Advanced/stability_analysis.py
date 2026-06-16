"""
stability_analysis.py
=====================
Stability analysis of high-order finite-difference operators
applied to the CMB power spectrum estimation.

Implements:
  1. von Neumann (Fourier) analysis of amplification factor
  2. Spectral radius via power iteration
  3. Smallest eigenvalue via inverse iteration
  4. CFL conditions for explicit time-stepping (Euler / RK4)
  5. Dispersion error analysis
"""

from __future__ import annotations
import math
import cmath
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# von Neumann analysis: amplification factor
# ---------------------------------------------------------------------------
def von_neumann_amplification(weights: List[float],
                                h: float,
                                k_vals: List[float]) -> List[complex]:
    """G(k) = sum_j w_j exp(i k j h)"""
    G = []
    n = len(weights)
    offset = (n - 1) // 2
    for k in k_vals:
        g = complex(0.0, 0.0)
        for j, w in enumerate(weights):
            m = j - offset
            g += w * cmath.exp(1j * k * m * h)
        G.append(g)
    return G


def von_neumann_stability_check(weights: List[float], h: float,
                                  n_k: int = 1000) -> Tuple[bool, float, float]:
    k_max = math.pi / h
    k_vals = [i * k_max / n_k for i in range(n_k + 1)]
    G = von_neumann_amplification(weights, h, k_vals)
    mag = [abs(g) for g in G]
    max_mag = max(mag)
    max_k = k_vals[mag.index(max_mag)]
    return max_mag <= 1.0 + 1e-12, max_mag, max_k


# ---------------------------------------------------------------------------
# Spectral radius via power iteration
# ---------------------------------------------------------------------------
def spectral_radius(A: List[List[float]], n_iter: int = 500,
                     tol: float = 1.0e-12) -> Tuple[float, List[float]]:
    n = len(A)
    x = [1.0 / math.sqrt(n) + 0.01 * ((i * 7 + 3) % 11 - 5) for i in range(n)]
    norm = math.sqrt(sum(v * v for v in x))
    x = [v / norm for v in x]
    lam = 0.0
    for _ in range(n_iter):
        y = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(max(1e-30, sum(v * v for v in y)))
        if norm < 1e-30:
            break
        x_new = [v / norm for v in y]
        lam_new = sum(x_new[i] * y[i] for i in range(n))
        if abs(lam_new - lam) < tol:
            return abs(lam_new), x_new
        x = x_new
        lam = lam_new
    return abs(lam), x


# ---------------------------------------------------------------------------
def smallest_eigenvalue(A: List[List[float]], shift: float = 0.0,
                          n_iter: int = 200, tol: float = 1.0e-12) -> Tuple[float, List[float]]:
    n = len(A)
    B = [[A[i][j] - (shift if i == j else 0.0) for j in range(n)] for i in range(n)]
    x = [1.0 / math.sqrt(n) + 0.01 * ((i * 13 + 7) % 9 - 4) for i in range(n)]
    norm = math.sqrt(sum(v * v for v in x))
    x = [v / norm for v in x]
    lam = 0.0
    for _ in range(n_iter):
        y = _solve_linear(B, x)
        if y is None:
            break
        norm = math.sqrt(max(1e-30, sum(v * v for v in y)))
        if norm < 1e-30:
            break
        x = [v / norm for v in y]
        num = sum(x[i] * sum(A[i][j] * x[j] for j in range(n)) for i in range(n))
        den = sum(x[i] * x[i] for i in range(n))
        lam_new = num / den if den > 1e-30 else 0.0
        if abs(lam_new - lam) < tol:
            return lam_new, x
        lam = lam_new
    return lam, x


# ---------------------------------------------------------------------------
# CFL conditions
# ---------------------------------------------------------------------------
def cfl_explicit_euler(L: List[List[float]], safety: float = 0.9) -> float:
    rho, _ = spectral_radius(L)
    if rho < 1e-30:
        return float("inf")
    return safety * 2.0 / rho


def cfl_rk4(L: List[List[float]], safety: float = 0.9) -> float:
    rho, _ = spectral_radius(L)
    if rho < 1e-30:
        return float("inf")
    return safety * 2.785 / rho


def backward_euler_stable(L: List[List[float]]) -> bool:
    n = len(L)
    sym = [[(L[i][j] + L[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    x = [1.0 / math.sqrt(n)] * n
    rq = sum(x[i] * sum(sym[i][j] * x[j] for j in range(n)) for i in range(n))
    return rq <= 1e-10


def eigenvalue_spread(L: List[List[float]]) -> Tuple[float, float, float]:
    lam_max, _ = spectral_radius(L)
    lam_min, _ = smallest_eigenvalue(L)
    if abs(lam_min) < 1e-30:
        kappa = float("inf")
    else:
        kappa = lam_max / abs(lam_min)
    return lam_min, lam_max, kappa


# ---------------------------------------------------------------------------
def _solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
        if max_val < 1.0e-30:
            return None
        M[k], M[max_row] = M[max_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-30:
            return None
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def dispersion_error(weights: List[float], h: float,
                       deriv_order: int = 2,
                       n_k: int = 100) -> List[Tuple[float, float]]:
    results = []
    for i in range(1, n_k + 1):
        k = i * math.pi / (n_k * h)
        G = von_neumann_amplification(weights, h, [k])[0]
        if deriv_order == 2 and G.real < 0:
            k_num_h = math.sqrt(-G.real)
            k_exact_h = k * h
            err = abs(k_num_h - k_exact_h) / k_exact_h if k_exact_h > 1e-30 else 0.0
            results.append((k_exact_h, err))
    return results


if __name__ == "__main__":
    w = [1.0, -2.0, 1.0]
    h = 0.1
    stable, max_mag, max_k = von_neumann_stability_check(w, h)
    print(f"von Neumann stability: {stable}, max|G| = {max_mag:.4e}")
    n = 5
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = -2.0
        if i > 0:     L[i][i - 1] = 1.0
        if i < n - 1: L[i][i + 1] = 1.0
    rho, _ = spectral_radius(L)
    print(f"Spectral radius = {rho:.6f}")
