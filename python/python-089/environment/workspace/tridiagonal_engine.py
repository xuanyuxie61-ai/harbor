"""
Tridiagonal Matrix Engine
==========================
Based on project 967_r83v.

Provides efficient solvers for tridiagonal (R83V format) systems arising
from 1D finite difference discretization of beam/plate bending equations
and modal analysis iterations in structural dynamics.

R83V storage format:
  - Subdiagonal: a[0:n-1]
  - Diagonal:    b[0:n]
  - Superdiagonal: c[0:n-1]

For a tridiagonal system A*x = rhs:
  A = [ b0  c0   0   ...   0  ]
      [ a0  b1  c1   ...   0  ]
      [  0  a1  b2   ...   0  ]
      [ ...     ...  ...  ... ]
      [  0   ... a_{n-2} b_{n-1} ]

Key applications:
- 1D beam bending: EI * d^4w/dx^4 = q  -> tridiagonal (with modifications)
- Modal iteration: inverse iteration for eigenvalues involves tridiagonal solves
- Thomas algorithm: O(n) direct solve
"""

import numpy as np


def r83v_mv(n, a, b, c, x):
    """
    Multiply an R83V tridiagonal matrix by a vector.
    
    (Ax)_i = a_{i-1}*x_{i-1} + b_i*x_i + c_i*x_{i+1}
    with a_{-1} = c_n = 0.
    
    Parameters
    ----------
    n : int
        Matrix dimension.
    a : ndarray, shape (n-1,)
        Subdiagonal.
    b : ndarray, shape (n,)
        Diagonal.
    c : ndarray, shape (n-1,)
        Superdiagonal.
    x : ndarray, shape (n,)
    
    Returns
    -------
    ax : ndarray, shape (n,)
    """
    x = np.asarray(x).flatten()
    ax = np.zeros(n)
    if n > 1:
        ax[1:] += a * x[:-1]
    ax += b * x
    if n > 1:
        ax[:-1] += c * x[1:]
    return ax


def r83v_fs(n, a, b, c, rhs):
    """
    Direct solve of tridiagonal system using modified Thomas algorithm
    with partial pivoting (based on LINPACK SGTSL).
    
    Factorization:
    c'(1) = 0
    c'(i) = a_{i-1} for i = 2..n
    d'(1) = e'(n) = 0
    d'(i) = c_{i-1} for i = 1..n-1
    
    Forward elimination with row interchanges for stability.
    
    Parameters
    ----------
    n : int
    a : ndarray, shape (n-1,)
    b : ndarray, shape (n,)
    c : ndarray, shape (n-1,)
    rhs : ndarray, shape (n,)
    
    Returns
    -------
    x : ndarray, shape (n,)
    """
    # Copy to avoid modifying inputs
    cp = np.zeros(n)
    cp[1:] = a.copy()
    dp = np.zeros(n)
    if n > 1:
        dp[:-1] = c.copy()
    dp[-1] = 0.0
    bp = b.copy()
    x = rhs.copy()
    
    # Forward elimination
    bp[0] = b[0]
    if n >= 2:
        dp[0] = c[0] if n > 1 else 0.0
        ep = np.zeros(n)
        ep[0] = 0.0
        ep[-1] = 0.0
        
        for k in range(1, n):
            # Partial pivoting between rows k-1 and k
            if abs(bp[k - 1]) <= abs(cp[k]):
                # Swap rows k-1 and k
                bp[k - 1], cp[k] = cp[k], bp[k - 1]
                dp[k - 1], ep[k] = ep[k], dp[k - 1]
                if k < n - 1:
                    ep[k - 1], dp[k] = dp[k], ep[k - 1]
                x[k - 1], x[k] = x[k], x[k - 1]
            
            if abs(bp[k - 1]) < 1e-30:
                raise ValueError(f"Zero pivot at step k={k}")
            
            t = -cp[k] / bp[k - 1]
            bp[k] = bp[k] + t * dp[k - 1]
            if k < n - 1:
                cp[k + 1] = cp[k + 1] + t * ep[k - 1]
            x[k] = x[k] + t * x[k - 1]
    
    if abs(bp[-1]) < 1e-30:
        raise ValueError("Zero pivot at final step")
    
    # Back substitution
    x[-1] = x[-1] / bp[-1]
    if n > 1:
        x[-2] = (x[-2] - dp[-2] * x[-1]) / bp[-2]
        for k in range(n - 3, -1, -1):
            x[k] = (x[k] - dp[k] * x[k + 1] - (ep[k] if k < n - 2 else 0.0) * x[k + 2]) / bp[k]
    
    return x


def r83v_cg(n, a, b, c, rhs, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Gradient solver for symmetric positive definite tridiagonal system.
    
    Algorithm:
    r_0 = b - A*x_0
    p_0 = r_0
    for k = 0, 1, 2, ...
        alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + alpha_k * p_k
        r_{k+1} = r_k - alpha_k * A * p_k
        beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + beta_k * p_k
    
    Parameters
    ----------
    n : int
    a, b, c : ndarray
        Tridiagonal entries.
    rhs : ndarray
    x0 : ndarray, optional
        Initial guess.
    tol : float
        Residual tolerance.
    max_iter : int, optional
    
    Returns
    -------
    x : ndarray
    iterations : int
    residual : float
    """
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = rhs - r83v_mv(n, a, b, c, x)
    p = r.copy()
    rsold = np.dot(r, r)
    
    for it in range(max_iter):
        Ap = r83v_mv(n, a, b, c, p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            return x, it + 1, np.sqrt(rsnew)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, max_iter, np.sqrt(rsold)


def r83v_jac_sl(n, a, b, c, rhs, x0=None, it_max=100, tol=1e-10):
    """
    Jacobi iteration for tridiagonal system.
    
    x_i^{new} = (rhs_i - a_{i-1}*x_{i-1} - c_i*x_{i+1}) / b_i
    
    Parameters
    ----------
    n : int
    a, b, c : ndarray
    rhs : ndarray
    x0 : ndarray, optional
    it_max : int
    tol : float
    
    Returns
    -------
    x : ndarray
    iterations : int
    residual : float
    """
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if np.any(np.abs(b) < 1e-30):
        raise ValueError("Zero diagonal entries detected")
    
    x_new = np.zeros(n)
    for it in range(it_max):
        x_new[0] = (rhs[0] - c[0] * x[1]) / b[0]
        if n > 2:
            x_new[1:-1] = (rhs[1:-1] - a[:-1] * x[:-2] - c[1:] * x[2:]) / b[1:-1]
        if n > 1:
            x_new[-1] = (rhs[-1] - a[-1] * x[-2]) / b[-1]
        
        diff = np.linalg.norm(x_new - x)
        x[:] = x_new
        if diff < tol:
            residual = np.linalg.norm(rhs - r83v_mv(n, a, b, c, x))
            return x, it + 1, residual
    
    residual = np.linalg.norm(rhs - r83v_mv(n, a, b, c, x))
    return x, it_max, residual


def build_beam_tridiagonal(n, EI, L, load_type='uniform'):
    """
    Build tridiagonal system for Euler-Bernoulli beam finite difference.
    
    For a simply supported beam of length L with n interior points:
    h = L / (n+1)
    (EI/h^4) * (w_{i-2} - 4w_{i-1} + 6w_i - 4w_{i+1} + w_{i+2}) = q_i
    
    This is banded-5, but for iterative modal analysis we can use
    a reduced tridiagonal approximation or preconditioner.
    
    Here we construct a simplified tridiagonal model representing
    the effective stiffness after boundary condition application:
    A = tridiag(-1, 2, -1) * (EI/h^3)  [simplified beam model]
    
    Parameters
    ----------
    n : int
        Number of interior points.
    EI : float
        Flexural rigidity.
    L : float
        Beam length.
    load_type : str
        Type of load for right-hand side.
    
    Returns
    -------
    a, b, c : ndarray
        Tridiagonal entries.
    rhs : ndarray
        Load vector.
    h : float
        Grid spacing.
    """
    h = L / (n + 1)
    
    # Simplified tridiagonal beam stiffness (represents effective bending)
    # This is a physically-motivated approximation used in modal analysis
    factor = EI / (h ** 3)
    a = -factor * np.ones(n - 1)
    b = 2.0 * factor * np.ones(n)
    c = -factor * np.ones(n - 1)
    
    if load_type == 'uniform':
        q0 = 1.0
        rhs = q0 * np.ones(n) * h
    elif load_type == 'point_center':
        rhs = np.zeros(n)
        rhs[n // 2] = 1.0
    else:
        rhs = np.zeros(n)
    
    return a, b, c, rhs, h


def modal_analysis_tridiagonal(n, a, b, c, n_modes=3, max_iter=100, tol=1e-8):
    """
    Compute lowest eigenvalues and eigenvectors of tridiagonal matrix
    using inverse iteration with tridiagonal solves.
    
    For structural dynamics: K * phi = omega^2 * M * phi
    Here using identity mass for simplicity in the 1D beam context.
    
    Parameters
    ----------
    n : int
    a, b, c : ndarray
        Tridiagonal stiffness entries.
    n_modes : int
        Number of modes to compute.
    max_iter : int
    tol : float
    
    Returns
    -------
    eigenvalues : ndarray
    eigenvectors : ndarray, shape (n, n_modes)
    """
    eigenvalues = np.zeros(n_modes)
    eigenvectors = np.zeros((n, n_modes))
    
    # Shift for inverse iteration
    sigma = 0.0
    
    for mode in range(n_modes):
        # Initial guess: sinusoidal shape
        phi = np.sin(np.linspace(0, np.pi, n) * (mode + 1))
        phi = phi / np.linalg.norm(phi)
        
        for it in range(max_iter):
            # Inverse iteration: (A - sigma*I)^{-1} * phi
            b_shifted = b.copy() - sigma
            try:
                phi_new = r83v_fs(n, a.copy(), b_shifted, c.copy(), phi)
            except ValueError:
                # Fallback to CG if direct solve fails
                phi_new, _, _ = r83v_cg(n, a, b_shifted, c, phi, tol=tol * 0.1)
            
            # Rayleigh quotient
            Aphi = r83v_mv(n, a, b, c, phi_new)
            rq = np.dot(phi_new, Aphi) / np.dot(phi_new, phi_new)
            
            phi_new = phi_new / np.linalg.norm(phi_new)
            
            # Convergence check
            diff = np.linalg.norm(phi_new - phi)
            phi = phi_new
            if diff < tol:
                break
        
        eigenvalues[mode] = rq
        eigenvectors[:, mode] = phi
        
        # Deflation: shift for next mode
        sigma = rq * 1.1
    
    return eigenvalues, eigenvectors
