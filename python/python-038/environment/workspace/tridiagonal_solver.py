"""
Tridiagonal Linear System Solvers (R83 format)
===============================================
Derived from 962_r83 (tridiagonal matrix linear algebra).

Provides compact-storage solvers for the symmetric tridiagonal systems
that arise from discretized DGLAP evolution and 1D diffusion equations
in parton shower medium effects.

Storage format (R83): A is stored as a 3×N array where
    A[0,:] = upper diagonal (length N-1, A[0,-1] unused)
    A[1,:] = main diagonal  (length N)
    A[2,:] = lower diagonal (length N-1, A[2,0]  unused)
"""

import numpy as np


def r83_cg_solve(A_r83, b, x0=None, tol=1e-12, max_iter=None):
    """
    Conjugate Gradient solver for symmetric positive-definite tridiagonal systems.
    
    Solves  A x = b  where A is symmetric tridiagonal stored in R83 format.
    
    Parameters
    ----------
    A_r83 : ndarray, shape (3, N)
        Tridiagonal matrix in R83 format.
    b : ndarray, shape (N,)
        Right-hand side vector.
    x0 : ndarray, shape (N,), optional
        Initial guess. Defaults to zero vector.
    tol : float
        Relative residual tolerance.
    max_iter : int, optional
        Maximum iterations. Defaults to N.
    
    Returns
    -------
    x : ndarray, shape (N,)
        Solution vector.
    info : dict
        Contains 'iterations', 'residual', 'converged'.
    """
    b = np.asarray(b, dtype=float)
    N = b.size
    if A_r83.shape != (3, N):
        raise ValueError(f"A_r83 shape {A_r83.shape} incompatible with b size {N}")
    
    if max_iter is None:
        max_iter = N
    
    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    
    # Matrix-vector product for R83 symmetric tridiagonal
    def matvec(v):
        v = np.asarray(v, dtype=float)
        out = A_r83[1, :] * v
        if N > 1:
            out[:-1] += A_r83[0, :-1] * v[1:]
            out[1:] += A_r83[2, 1:] * v[:-1]
        return out
    
    r = b - matvec(x)
    p = r.copy()
    rs_old = float(np.dot(r, r))
    rs0 = rs_old if rs_old > 0 else 1.0
    
    converged = False
    k = 0
    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new / rs0) < tol:
            converged = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    info = {
        'iterations': k + 1,
        'residual': np.sqrt(rs_old / rs0) if rs0 > 0 else 0.0,
        'converged': converged
    }
    return x, info


def r83_cyclic_reduction(A_r83, b):
    """
    Cyclic reduction (odd-even elimination) for symmetric tridiagonal systems.
    Efficient O(N) direct solver, numerically stable for diagonally dominant systems.
    
    Parameters
    ----------
    A_r83 : ndarray, shape (3, N)
    b : ndarray, shape (N,)
    
    Returns
    -------
    x : ndarray, shape (N,)
        Solution vector.
    """
    b = np.asarray(b, dtype=float)
    N = b.size
    if A_r83.shape != (3, N):
        raise ValueError("Shape mismatch in r83_cyclic_reduction")
    
    if N == 0:
        return np.array([], dtype=float)
    if N == 1:
        return np.array([b[0] / A_r83[1, 0]])
    
    # Extract diagonals
    c = A_r83[2, 1:].copy()   # lower: sub-diagonal (length N-1)
    d = A_r83[1, :].copy()    # main diagonal (length N)
    e = A_r83[0, :-1].copy()  # upper: super-diagonal (length N-1)
    rhs = b.copy()
    
    # Forward elimination (Thomas algorithm, standard for tridiagonal)
    for i in range(1, N):
        if abs(d[i - 1]) < 1e-30:
            raise RuntimeError("Zero pivot in cyclic reduction")
        w = c[i - 1] / d[i - 1]
        d[i] -= w * e[i - 1]
        rhs[i] -= w * rhs[i - 1]
    
    # Back substitution
    x = np.zeros(N, dtype=float)
    x[-1] = rhs[-1] / d[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (rhs[i] - e[i] * x[i + 1]) / d[i]
    
    return x


def build_dif2_r83(N):
    """
    Build the classic DIF2 test matrix (discretized 1D Laplacian) in R83 format:
        A_{ii} = 2, A_{i,i+1} = A_{i+1,i} = -1.
    
    This matrix appears in the spatial discretization of diffusion equations
    describing parton cascade thermalization in QGP.
    """
    A = np.zeros((3, N), dtype=float)
    A[1, :] = 2.0           # main diagonal
    if N > 1:
        A[0, :-1] = -1.0    # upper
        A[2, 1:] = -1.0     # lower
    return A


def solve_diffusion_1d(u0, D, dt, dx, n_steps, solver='cyclic'):
    """
    Solve 1D diffusion equation  ∂u/∂t = D ∂²u/∂x²  with zero Dirichlet BCs
    using implicit Euler + tridiagonal solve.
    
    Discretization:
        (u^{n+1}_i - u^n_i)/dt = D (u^{n+1}_{i+1} - 2u^{n+1}_i + u^{n+1}_{i-1})/dx^2
    
    Parameters
    ----------
    u0 : ndarray, shape (N,)
        Initial condition.
    D : float
        Diffusion coefficient.
    dt, dx : float
        Time and space steps.
    n_steps : int
        Number of time steps.
    solver : str
        'cyclic' or 'cg'.
    
    Returns
    -------
    u : ndarray, shape (N,)
        Final solution profile.
    """
    N = u0.size
    r = D * dt / (dx * dx)
    if r > 10.0:
        # Warn but do not abort; implicit method is unconditionally stable
        pass
    
    # Build implicit system: (I + r * L) u^{n+1} = u^n
    A = np.zeros((3, N), dtype=float)
    A[1, :] = 1.0 + 2.0 * r   # main diagonal
    if N > 1:
        A[0, :-1] = -r        # upper
        A[2, 1:] = -r         # lower
    
    # Zero Dirichlet boundaries: override rows
    A[1, 0] = 1.0
    A[0, 0] = 0.0
    A[2, 0] = 0.0
    A[1, -1] = 1.0
    A[0, -1] = 0.0
    A[2, -1] = 0.0
    
    u = np.asarray(u0, dtype=float).copy()
    
    for _ in range(n_steps):
        rhs = u.copy()
        rhs[0] = 0.0
        rhs[-1] = 0.0
        
        if solver == 'cyclic':
            u = r83_cyclic_reduction(A, rhs)
        elif solver == 'cg':
            u, _ = r83_cg_solve(A, rhs)
        else:
            raise ValueError("solver must be 'cyclic' or 'cg'")
    
    return u


def test_tridiagonal_solvers():
    """
    Validation test: solve DIF2 system of size 100 and verify residual.
    """
    N = 100
    A = build_dif2_r83(N)
    x_exact = np.sin(np.linspace(0, np.pi, N))
    # Compute RHS
    b = np.zeros(N)
    b[0] = A[1, 0] * x_exact[0] + A[0, 0] * x_exact[1]
    for i in range(1, N - 1):
        b[i] = A[2, i] * x_exact[i - 1] + A[1, i] * x_exact[i] + A[0, i] * x_exact[i + 1]
    b[-1] = A[2, -1] * x_exact[-2] + A[1, -1] * x_exact[-1]
    
    x_cg, info_cg = r83_cg_solve(A, b)
    x_cr = r83_cyclic_reduction(A, b)
    
    err_cg = np.max(np.abs(x_cg - x_exact))
    err_cr = np.max(np.abs(x_cr - x_exact))
    
    if err_cg > 1e-8 or err_cr > 1e-8:
        raise RuntimeError(f"Tridiagonal solver test failed: CG err={err_cg}, CR err={err_cr}")
    
    return {'cg_error': err_cg, 'cr_error': err_cr, 'cg_info': info_cg}


if __name__ == "__main__":
    results = test_tridiagonal_solvers()
    print("Tridiagonal solver tests passed.")
    print(f"  CG max error: {results['cg_error']:.2e}")
    print(f"  CR max error: {results['cr_error']:.2e}")
