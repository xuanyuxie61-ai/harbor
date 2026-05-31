# -*- coding: utf-8 -*-
"""
Tridiagonal and Banded Linear Solvers
======================================
Specialized solvers for tridiagonal systems arising from spectral
and finite-difference discretizations. Adapted from the R83T library.

For a tridiagonal matrix A with sub-diagonal a, diagonal b, super-diagonal c:
    A u = d
The R83T format stores the three diagonals as columns of an N x 3 matrix:
    R83T[i,0] = a_i  (sub-diagonal, a_0 unused)
    R83T[i,1] = b_i  (main diagonal)
    R83T[i,2] = c_i  (super-diagonal, c_{N-1} unused)

Methods implemented:
- Direct Thomas algorithm (O(N))
- Jacobi iteration
- Gauss-Seidel iteration
- Conjugate Gradient (for SPD systems)
"""

import numpy as np


def r83t_to_dense(r83t):
    """
    Convert R83T format to dense N x N matrix for verification.

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
        Tridiagonal matrix in R83T format.

    Returns
    -------
    A : ndarray, shape (N, N)
        Dense matrix.
    """
    N = r83t.shape[0]
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        A[i, i] = r83t[i, 1]
        if i > 0:
            A[i, i - 1] = r83t[i, 0]
        if i < N - 1:
            A[i, i + 1] = r83t[i, 2]
    return A


def r83t_mv(r83t, x):
    """
    Matrix-vector product y = A x for R83T tridiagonal matrix.

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
    x : ndarray, shape (N,)

    Returns
    -------
    y : ndarray, shape (N,)
    """
    r83t = np.asarray(r83t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    N = r83t.shape[0]
    y = np.zeros(N, dtype=np.float64)
    for i in range(N):
        y[i] = r83t[i, 1] * x[i]
        if i > 0:
            y[i] += r83t[i, 0] * x[i - 1]
        if i < N - 1:
            y[i] += r83t[i, 2] * x[i + 1]
    return y


def r83t_dif2(N):
    """
    Construct the classic DIF2 tridiagonal matrix (1D Laplacian with Dirichlet BC):
        A = tridiag(-1, 2, -1) of size N x N.
    Eigenvalues: lambda_k = 4 sin^2(k pi / (2(N+1))), k=1,...,N.

    Parameters
    ----------
    N : int
        Matrix dimension.

    Returns
    -------
    r83t : ndarray, shape (N, 3)
    """
    r83t = np.zeros((N, 3), dtype=np.float64)
    r83t[:, 0] = -1.0
    r83t[:, 1] = 2.0
    r83t[:, 2] = -1.0
    r83t[0, 0] = 0.0
    r83t[N - 1, 2] = 0.0
    return r83t


def thomas_solve(r83t, d):
    """
    Solve a tridiagonal system using the Thomas algorithm (direct O(N)).

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
        Tridiagonal matrix in R83T format.
    d : ndarray, shape (N,)
        Right-hand side.

    Returns
    -------
    x : ndarray, shape (N,)
        Solution vector.
    """
    r83t = np.asarray(r83t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).copy()
    N = r83t.shape[0]
    a = r83t[:, 0].copy()
    b = r83t[:, 1].copy()
    c = r83t[:, 2].copy()

    # Forward elimination
    for i in range(1, N):
        if abs(b[i - 1]) < 1e-15:
            raise ValueError("Zero pivot encountered in Thomas algorithm.")
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    # Back substitution
    if abs(b[N - 1]) < 1e-15:
        raise ValueError("Zero pivot at last row.")
    x = np.zeros(N, dtype=np.float64)
    x[N - 1] = d[N - 1] / b[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def jacobi_solve(r83t, d, x0=None, tol=1e-10, max_iter=10000):
    """
    Solve tridiagonal system using Jacobi iteration.

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
    d : ndarray, shape (N,)
    x0 : ndarray, optional
        Initial guess.
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    x : ndarray
        Solution.
    info : dict
        Iteration count and residual history.
    """
    r83t = np.asarray(r83t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    N = r83t.shape[0]
    x = np.zeros(N, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    x_new = np.zeros(N, dtype=np.float64)
    diag = r83t[:, 1]
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal entries: Jacobi may diverge.")

    residuals = []
    for it in range(max_iter):
        for i in range(N):
            sigma = 0.0
            if i > 0:
                sigma += r83t[i, 0] * x[i - 1]
            if i < N - 1:
                sigma += r83t[i, 2] * x[i + 1]
            x_new[i] = (d[i] - sigma) / diag[i]
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x[:] = x_new
        residuals.append(diff)
        if diff < tol:
            return x, {"iterations": it + 1, "residual": diff, "history": residuals}
    return x, {"iterations": max_iter, "residual": diff, "history": residuals, "converged": False}


def gauss_seidel_solve(r83t, d, x0=None, tol=1e-10, max_iter=10000):
    """
    Solve tridiagonal system using Gauss-Seidel iteration.

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
    d : ndarray, shape (N,)
    x0 : ndarray, optional
    tol : float
    max_iter : int

    Returns
    -------
    x : ndarray
    info : dict
    """
    r83t = np.asarray(r83t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    N = r83t.shape[0]
    x = np.zeros(N, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    diag = r83t[:, 1]
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal entries: GS may diverge.")

    residuals = []
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(N):
            sigma = 0.0
            if i > 0:
                sigma += r83t[i, 0] * x[i - 1]
            if i < N - 1:
                sigma += r83t[i, 2] * x[i + 1]
            x[i] = (d[i] - sigma) / diag[i]
        diff = np.linalg.norm(x - x_old, ord=np.inf)
        residuals.append(diff)
        if diff < tol:
            return x, {"iterations": it + 1, "residual": diff, "history": residuals}
    return x, {"iterations": max_iter, "residual": diff, "history": residuals, "converged": False}


def conjugate_gradient_solve(r83t, d, x0=None, tol=1e-10, max_iter=None):
    """
    Solve symmetric positive-definite tridiagonal system using Conjugate Gradient.

    Parameters
    ----------
    r83t : ndarray, shape (N, 3)
    d : ndarray, shape (N,)
    x0 : ndarray, optional
    tol : float
    max_iter : int, optional
        Defaults to N.

    Returns
    -------
    x : ndarray
    info : dict
    """
    r83t = np.asarray(r83t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    N = r83t.shape[0]
    if max_iter is None:
        max_iter = N
    x = np.zeros(N, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    r = d - r83t_mv(r83t, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    residuals = []

    for it in range(max_iter):
        Ap = r83t_mv(r83t, p)
        alpha = rs_old / (np.dot(p, Ap) + 1e-15)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        residuals.append(np.sqrt(rs_new))
        if np.sqrt(rs_new) < tol:
            return x, {"iterations": it + 1, "residual": np.sqrt(rs_new), "history": residuals}
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, {"iterations": max_iter, "residual": np.sqrt(rs_new), "history": residuals, "converged": False}


def build_spectral_tridiagonal(D2, bc_type="dirichlet"):
    """
    Extract the tridiagonal approximation of a spectral second-derivative
    operator for use with iterative solvers. In practice, for Chebyshev
    methods the full matrix is dense; here we approximate the Laplacian
    by the standard DIF2 matrix for demonstration of the tridiagonal solvers.

    Parameters
    ----------
    D2 : ndarray
        Spectral second-derivative matrix (dense).
    bc_type : str
        Boundary condition type.

    Returns
    -------
    r83t : ndarray
        Tridiagonal approximation.
    """
    N = D2.shape[0]
    r83t = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        r83t[i, 1] = D2[i, i]
        if i > 0:
            r83t[i, 0] = D2[i, i - 1]
        if i < N - 1:
            r83t[i, 2] = D2[i, i + 1]
    # For Dirichlet, boundary rows should enforce BCs
    if bc_type == "dirichlet":
        r83t[0, :] = 0.0
        r83t[0, 1] = 1.0
        r83t[-1, :] = 0.0
        r83t[-1, 1] = 1.0
    return r83t
