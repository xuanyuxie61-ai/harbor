# -*- coding: utf-8 -*-
"""
linear_solver.py
================
Robust linear system solvers adapted from three seed projects:
  - 989_r8po:  Cholesky factorization for symmetric positive-definite matrices
  - 1001_r8ut: Upper-triangular back-substitution
  - 153_cg_squared:  CGS (Conjugate Gradient Squared) iterative solver

These solvers support the time-integration loop where each step requires
solving a linear system with the effective stiffness matrix:
  K_eff * Delta_u = R_eff

The effective stiffness matrix in Newmark-β is:
  K_eff = K + (gamma / (beta * dt)) * C + (1 / (beta * dt^2)) * M
which is symmetric positive definite for stable Newmark parameters.
"""

import numpy as np
from typing import Tuple, Optional


# ====================================================================== #
# Cholesky factorization (from 989_r8po seed)
# ====================================================================== #
def cholesky_factorize(A: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute the Cholesky factorization  A = R^T * R  where R is upper triangular.
    
    Returns
    -------
    R : np.ndarray
        Upper triangular Cholesky factor.
    info : int
        0 if successful, otherwise the index of the first non-positive
        principal minor.
    """
    n = A.shape[0]
    R = A.copy().astype(float)
    info = 0

    for j in range(n):
        # Update column j
        for k in range(j):
            t = np.dot(R[:k, k], R[:k, j])
            R[k, j] = (R[k, j] - t) / R[k, k]

        t = np.dot(R[:j, j], R[:j, j])
        s = R[j, j] - t

        if s <= 0.0:
            info = j + 1
            return R, info

        R[j, j] = np.sqrt(s)

    # Zero out strictly lower triangle (ensure R8UT / upper-triangular form)
    for i in range(n):
        for j in range(i):
            R[i, j] = 0.0

    return R, info


# ====================================================================== #
# Upper-triangular solve (from 1001_r8ut seed)
# ====================================================================== #
def solve_upper_triangular(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve  U * x = b  for x, where U is upper triangular.
    
    Algorithm (back-substitution):
      x_j = (b_j - sum_{i=1}^{j-1} U_{i,j} * x_i) / U_{j,j}
    processed j = n, n-1, ..., 1.
    """
    n = U.shape[0]
    x = b.copy().astype(float)

    for j in range(n - 1, -1, -1):
        if abs(U[j, j]) < 1e-15:
            raise ValueError(f"Zero diagonal entry in upper triangular system at index {j}")
        x[j] = x[j] / U[j, j]
        for i in range(j):
            x[i] = x[i] - U[i, j] * x[j]

    return x


def solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve  L * x = b  for x, where L is lower triangular.
    """
    n = L.shape[0]
    x = b.copy().astype(float)

    for j in range(n):
        if abs(L[j, j]) < 1e-15:
            raise ValueError(f"Zero diagonal entry in lower triangular system at index {j}")
        x[j] = x[j] / L[j, j]
        for i in range(j + 1, n):
            x[i] = x[i] - L[i, j] * x[j]

    return x


# ====================================================================== #
# Cholesky solve: A * x = b  via  L * L^T  (or  R^T * R)
# ====================================================================== #
def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A * x = b using Cholesky factorization.
    Steps:
      1. Factor A = R^T * R
      2. Solve R^T * y = b   (forward substitution)
      3. Solve R   * x = y   (back substitution)
    """
    R, info = cholesky_factorize(A)
    if info != 0:
        # Fallback: add small regularization
        eps = 1e-10 * np.max(np.diag(A))
        R, info2 = cholesky_factorize(A + eps * np.eye(A.shape[0]))
        if info2 != 0:
            raise np.linalg.LinAlgError(f"Matrix is not positive definite (info={info})")

    # A = R^T * R, where R is upper triangular
    # Solve R^T * y = b
    y = solve_lower_triangular(R.T, b)
    # Solve R * x = y
    x = solve_upper_triangular(R, y)
    return x


# ====================================================================== #
# CGS iterative solver (from 153_cg_squared seed)
# ====================================================================== #
def cgs_squared(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """
    Conjugate Gradient Squared (CGS) method for solving A * x = b.
    
    CGS avoids the multiplication by A^T required by BiCG and converges
    roughly twice as fast as standard CG for nonsymmetric systems.
    For symmetric positive-definite A, CGS is robust and efficient.
    
    Reference:
      Sonneveld, P. (1989). CGS: A fast Lanczos-type solver for nonsymmetric
      linear systems. SIAM J. Sci. Stat. Comput., 10(1), 36-52.
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy().astype(float)

    normb = float(np.linalg.norm(b))
    if normb == 0.0:
        normb = 1.0

    r = b - A @ x
    resid = float(np.linalg.norm(r)) / normb
    if resid <= tol:
        return x

    rtilde = r.copy()
    rho = 0.0
    u = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    q = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        rho_old = rho
        rho = float(rtilde @ r)

        if abs(rho) < 1e-30:
            break

        if _ == 0:
            u = r.copy()
            p = u.copy()
        else:
            beta = rho / rho_old
            u = r + beta * q
            p = u + beta * (q + beta * p)

        phat = p.copy()
        vhat = A @ phat
        denom = float(rtilde @ vhat)
        if abs(denom) < 1e-30:
            break

        alpha = rho / denom
        q = u - alpha * vhat
        uhat = u + q
        x = x + alpha * uhat
        qhat = A @ uhat
        r = r - alpha * qhat

        resid = float(np.linalg.norm(r)) / normb
        if resid < tol:
            break

    return x


# ====================================================================== #
# Unified solver wrapper with automatic method selection
# ====================================================================== #
def solve_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    method: str = "auto",
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Solve A * x = b with automatic method selection.
    
    Methods:
      - "cholesky": Direct Cholesky factorization (best for SPD, small/medium)
      - "cgs":      CGS iterative solver (best for large sparse)
      - "auto":     Choose based on matrix size and properties
    """
    n = A.shape[0]

    if method == "auto":
        if n <= 100:
            method = "cholesky"
        else:
            method = "cgs"

    if method == "cholesky":
        try:
            return cholesky_solve(A, b)
        except np.linalg.LinAlgError:
            return cgs_squared(A, b, tol=tol)
    elif method == "cgs":
        return cgs_squared(A, b, tol=tol)
    else:
        raise ValueError(f"Unknown solver method: {method}")
