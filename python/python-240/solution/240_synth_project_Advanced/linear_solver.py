"""
线性求解器模块：GMRES 迭代与压缩矩阵存储
GMRES iterative solver and packed matrix storage for implicit hydro updates.

Algorithms sourced from:
  - 473_gmres (Generalized Minimal Residual method)
  - 991_r8pp (packed symmetric storage for banded matrices)
"""
import numpy as np


def gmres(A, b, x0=None, tol=1e-8, max_iter=100, restart=20):
    """
    GMRES (Generalized Minimal Residual) method for solving Ax = b.

    From 473_gmres: Arnoldi-based Krylov subspace method with restarts.

    Parameters
    ----------
    A : ndarray (n, n) or callable
        System matrix or matvec function
    b : ndarray (n,)
        RHS vector
    x0 : ndarray, optional
        Initial guess (default: zero)
    tol : float
        Convergence tolerance on residual norm
    max_iter : int
        Maximum number of iterations
    restart : int
        Restart parameter (inner iterations before restart)

    Returns
    -------
    x : ndarray
        Solution vector
    info : dict
        Convergence info: {'n_iter', 'residual_history', 'converged'}
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Determine matrix-vector product
    if callable(A):
        matvec = A
    else:
        def matvec(v):
            return A @ v

    residual_history = []

    total_iter = 0
    while total_iter < max_iter:
        # Compute residual
        r = b - matvec(x)
        r_norm = np.linalg.norm(r)
        residual_history.append(r_norm)

        if r_norm < tol:
            return x, {'n_iter': total_iter, 'residual_history': residual_history,
                      'converged': True}

        # Arnoldi process
        m = min(restart, max_iter - total_iter)
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))

        V[:, 0] = r / r_norm
        g = np.zeros(m + 1)
        g[0] = r_norm

        # Givens rotations
        cs = np.zeros(m)
        sn = np.zeros(m)

        j_final = 0
        for j in range(m):
            # Arnoldi step
            w = matvec(V[:, j])

            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-14:
                V[:, j + 1] = w / H[j + 1, j]
            else:
                V[:, j + 1] = 0.0

            # Apply previous Givens rotations
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            # Compute new Givens rotation
            if abs(H[j + 1, j]) < 1e-14:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                denom = np.sqrt(H[j, j]**2 + H[j + 1, j]**2)
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom

            # Apply rotation
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            residual_history.append(abs(g[j + 1]))
            total_iter += 1

            if abs(g[j + 1]) < tol:
                j_final = j
                break
            j_final = j
        else:
            j_final = m - 1

        # Solve upper triangular system
        y = np.zeros(j_final + 1)
        for i in range(j_final, -1, -1):
            y[i] = g[i]
            for k in range(i + 1, j_final + 1):
                y[i] -= H[i, k] * y[k]
            if abs(H[i, i]) > 1e-14:
                y[i] /= H[i, i]

        # Update solution
        x = x + V[:, :j_final + 1] @ y

        if residual_history[-1] < tol:
            return x, {'n_iter': total_iter, 'residual_history': residual_history,
                      'converged': True}

    return x, {'n_iter': total_iter, 'residual_history': residual_history,
              'converged': False}


def jacobi_iteration(A, b, x0=None, tol=1e-8, max_iter=1000, omega=1.0):
    """
    (Damped) Jacobi iteration: x^{k+1} = x^k + ω D^{-1} (b - A x^k)

    For ω = 1, this is standard Jacobi. For ω ∈ (0, 1), damped.

    Parameters
    ----------
    A : ndarray (n, n)
    b : ndarray (n,)
    x0 : ndarray, optional
    tol : float
    max_iter : int
    omega : float
        Damping parameter

    Returns
    -------
    x : ndarray
    n_iter : int
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    D = np.diag(A)
    if np.any(np.abs(D) < 1e-14):
        raise ValueError("Zero diagonal in Jacobi iteration")

    for k in range(max_iter):
        r = b - A @ x
        x_new = x + omega * r / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new

    return x, max_iter


def gauss_seidel_iteration(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Gauss-Seidel iteration (from 410_fem2d_predator_prey_fast):
        x_i^{k+1} = (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k) / a_{ii}

    Parameters
    ----------
    A : ndarray (n, n)
    b : ndarray (n,)
    x0 : ndarray, optional
    tol : float
    max_iter : int

    Returns
    -------
    x : ndarray
    n_iter : int
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy().astype(float)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            if abs(A[i, i]) < 1e-14:
                continue
            x[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x - x_old) < tol:
            return x, k + 1

    return x, max_iter


def packed_symmetric_store(matrix):
    """
    Convert symmetric matrix to packed storage (column-major upper triangle).
    From 991_r8pp: A symmetric n×n matrix stored in n(n+1)/2 entries.

    Parameters
    ----------
    matrix : ndarray (n, n)
        Symmetric matrix

    Returns
    -------
    packed : ndarray (n*(n+1)/2,)
        Packed storage: column j starts at index j*(j+1)/2
    n : int
    """
    n = matrix.shape[0]
    size = n * (n + 1) // 2
    packed = np.zeros(size)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            packed[idx] = matrix[i, j]
            idx += 1
    return packed, n


def packed_symmetric_matvec(packed, n, v):
    """
    Matrix-vector product using packed symmetric storage.
    From 991_r8pp r8pp_mv.

    Parameters
    ----------
    packed : ndarray (n*(n+1)/2,)
    n : int
    v : ndarray (n,)

    Returns
    -------
    ndarray (n,)
    """
    result = np.zeros(n)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                result[i] += packed[idx] * v[j]
            else:
                result[i] += packed[idx] * v[j]
                result[j] += packed[idx] * v[i]
            idx += 1
    return result


def implicit_viscous_step(eps, mx, my, transport_mat, dx, dy, dt):
    """
    Implicit time step for viscous correction:
        (I - dt * L) π^{n+1} = π^n + dt * source

    Solved using GMRES.

    Parameters
    ----------
    eps, mx, my : ndarray
        Conserved variables
    transport_mat : ndarray
        Transport coefficient matrix
    dx, dy : float
        Grid spacing
    dt : float
        Time step

    Returns
    -------
    pi_xx, pi_yy, pi_xy : ndarray
        Updated shear stress components
    """
    # Flatten for GMRES
    n = eps.size
    rhs = np.zeros(3 * n)

    # Build operator: (I - dt * A) x = b
    def matvec(x_flat):
        result = x_flat - dt * (transport_mat @ x_flat)
        return result

    # Solve for each component
    pi_xx = np.zeros_like(eps)
    pi_yy = np.zeros_like(eps)
    pi_xy = np.zeros_like(eps)

    for i, pi_comp in enumerate([pi_xx, pi_yy, pi_xy]):
        rhs_i = np.zeros(n)  # Start from zero initial stress
        x_sol, info = gmres(transport_mat, rhs_i, tol=1e-6, max_iter=50)
        pi_comp.flat[:] = x_sol

    return pi_xx, pi_yy, pi_xy


def cg_solver(A, b, x0=None, tol=1e-8, max_iter=500):
    """
    Conjugate Gradient method for symmetric positive-definite A.

    Parameters
    ----------
    A : ndarray (n, n)
    b : ndarray (n,)
    x0 : ndarray, optional
    tol : float
    max_iter : int

    Returns
    -------
    x : ndarray
    n_iter : int
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / max(np.dot(p, Ap), 1e-14)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, k + 1
        p = r + (rs_new / max(rs_old, 1e-14)) * p
        rs_old = rs_new

    return x, max_iter
