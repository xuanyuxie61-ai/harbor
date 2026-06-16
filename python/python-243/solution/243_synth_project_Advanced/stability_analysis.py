"""
stability_analysis.py - Eigenvalue stability and Cholesky factorization.

Adapted from 025_asa006 (Cholesky decomposition).  Analyzes the Jacobian
eigenvalues to determine stiffness and stability of the r-process network.

Eigenvalue criterion:
  Stiff if  Re(lambda_max) / Re(lambda_min) > 10^3
  Unstable if  any Re(lambda) > 0

Cholesky factorization for positive-definite covariance matrices:
  A = L L^T   where L is lower triangular.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Cholesky factorization  A = L L^T  for symmetric positive-definite A.
    Returns L (lower triangular).  Adapted from 025_asa006/cholesky.m.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Matrix must be square")
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - s
                if val <= 0.0:
                    raise ValueError(f"Matrix not positive-definite at ({i},{i})")
                L[i, j] = np.sqrt(val)
            else:
                if abs(L[j, j]) < 1e-300:
                    L[i, j] = 0.0
                else:
                    L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve  A x = b  given  A = L L^T  via forward/back substitution."""
    n = len(b)
    # Forward: L y = b
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = (b[i] - s) / L[i, i] if abs(L[i, i]) > 1e-300 else 0.0
    # Backward: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k, i] * x[k]
        x[i] = (y[i] - s) / L[i, i] if abs(L[i, i]) > 1e-300 else 0.0
    return x


def eigenvalue_stiffness(J: np.ndarray) -> Tuple[float, float, bool, bool]:
    """
    Compute eigenvalue ratio and stiffness indicators.
    Returns (lambda_max_real, lambda_min_real, is_stiff, is_unstable).
    """
    try:
        eigvals = np.linalg.eigvals(J)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, False, True
    re_eig = np.real(eigvals)
    if len(re_eig) == 0:
        return 0.0, 0.0, False, False
    lam_max = np.max(re_eig)
    lam_min = np.min(np.abs(re_eig[re_eig != 0])) if np.any(re_eig != 0) else 1e-300
    ratio = abs(lam_max) / max(abs(lam_min), 1e-300)
    is_stiff = ratio > 1e3
    is_unstable = lam_max > 0.0
    return float(lam_max), float(lam_min), bool(is_stiff), bool(is_unstable)


def von_neumann_stability(dx: float, dt: float, u_adv: float, nu: float) -> bool:
    """
    Von Neumann stability analysis for advection-diffusion:
      C = u dt / dx <= 1  (advection)
      D = nu dt / dx^2 <= 0.5  (diffusion)
    Returns True if stable.
    """
    C = abs(u_adv) * dt / dx if dx > 0 else 0.0
    D = nu * dt / (dx * dx) if dx > 0 else 0.0
    return C <= 1.0 and D <= 0.5


def cfl_condition(dx: float, u_max: float) -> float:
    """Return maximum stable dt for CFL condition  C <= 1."""
    if abs(u_max) < 1e-300:
        return 1.0e10
    return dx / abs(u_max)


def build_jacobian_network(dYdt_func, Y0: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Numerically approximate Jacobian  J_ij = dF_i / dY_j  via finite differences.
    """
    n = len(Y0)
    J = np.zeros((n, n))
    F0 = dYdt_func(Y0)
    for j in range(n):
        Y_pert = Y0.copy()
        Y_pert[j] += eps
        F_pert = dYdt_func(Y_pert)
        J[:, j] = (F_pert - F0) / eps
    return J


__all__ = [
    "cholesky_decomposition", "cholesky_solve",
    "eigenvalue_stiffness", "von_neumann_stability", "cfl_condition",
    "build_jacobian_network",
]
