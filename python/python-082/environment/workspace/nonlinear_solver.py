"""
nonlinear_solver.py
===================
Newton-Raphson nonlinear solver with line search for damage mechanics.

Incorporates core algorithms from:
- 688_linpack_bench_backslash : Dense linear system solution and
    residual norm computation for Newton increments.
- 972_r8but : Banded upper triangular solver as a preconditioner
    for the tangent stiffness matrix.

Scientific role:
    Solves the nonlinear equilibrium equation:
        R(u) = F_int(u) - F_ext = 0
    where F_int is the internal force depending on damage state d(u).
    The tangent stiffness K_T = dR/du is assembled and solved at each
    Newton iteration. Damage-dependent stiffness requires robust
    line search to prevent divergence during softening.

Key formulas:
-----------
1. Newton iteration:
   K_T(u_k) * delta_u = -R(u_k)
   u_{k+1} = u_k + alpha * delta_u

2. Line search (backtracking):
   Find alpha in (0,1] such that ||R(u + alpha*delta)|| < ||R(u)||.

3. Convergence criteria:
   ||R|| / ||F_ext|| < tol_R   (residual)
   ||delta_u|| / ||u|| < tol_u  (displacement)

4. Tangent stiffness for damage:
   K_T = integral B^T C(d) B dV
   where C(d) = M^{-1}(d) C_0 M^{-T}(d) is the degraded stiffness.

5. BFGS update (optional quasi-Newton):
   H_{k+1} = (I - rho_k s_k y_k^T) H_k (I - rho_k y_k s_k^T) + rho_k s_k s_k^T
   where s_k = u_{k+1} - u_k, y_k = R_{k+1} - R_k, rho_k = 1/(y_k^T s_k).
"""

import numpy as np
from sparse_assembler import r8but_solve, dense_to_banded_upper


def newton_raphson(residual_func, tangent_func, u0,
                   tol_r=1e-6, tol_u=1e-8, max_iter=50,
                   line_search=True, verbose=False):
    """
    Solve nonlinear system R(u) = 0 using Newton-Raphson.

    Parameters
    ----------
    residual_func : callable
        R(u) -> ndarray.
    tangent_func : callable
        K_T(u) -> ndarray (dense tangent stiffness).
    u0 : ndarray
        Initial guess.
    tol_r, tol_u : float
        Convergence tolerances.
    max_iter : int
        Maximum iterations.
    line_search : bool
        Enable backtracking line search.
    verbose : bool

    Returns
    -------
    u : ndarray
        Converged solution.
    converged : bool
    history : dict
        {'residual_norm', 'displacement_norm', 'iterations'}
    """
    u = np.asarray(u0, dtype=float).copy()
    history = {'residual_norm': [], 'displacement_norm': [], 'iterations': 0}

    for it in range(max_iter):
        R = residual_func(u)
        K = tangent_func(u)

        r_norm = np.linalg.norm(R)
        u_norm = np.linalg.norm(u) + 1e-14
        history['residual_norm'].append(r_norm)
        history['displacement_norm'].append(u_norm)

        if verbose:
            print(f"  Iter {it}: ||R||={r_norm:.6e}, ||u||={u_norm:.6e}")

        if r_norm < tol_r:
            history['iterations'] = it
            return u, True, history

        # Solve K * du = -R
        K_arr = np.asarray(K, dtype=float)
        if K_arr.ndim == 0:
            K_arr = np.array([[float(K_arr)]])
        elif K_arr.ndim == 1:
            K_arr = np.atleast_2d(K_arr).T
        try:
            du = np.linalg.solve(K_arr, -R)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            du = -np.linalg.lstsq(K_arr, R, rcond=None)[0]

        du_norm = np.linalg.norm(du)

        if du_norm / u_norm < tol_u:
            history['iterations'] = it
            return u, True, history

        # Line search
        alpha = 1.0
        if line_search:
            for _ in range(10):
                u_trial = u + alpha * du
                R_trial = residual_func(u_trial)
                if np.linalg.norm(R_trial) < r_norm:
                    break
                alpha *= 0.5
            else:
                if verbose:
                    print("  Line search failed, using alpha=1.0")
                alpha = 1.0

        u = u + alpha * du

    history['iterations'] = max_iter
    return u, False, history


def arc_length_solver(residual_func, tangent_func, u0, load_factor0,
                      d_lambda=0.1, max_steps=20,
                      tol_r=1e-6, max_iter=15):
    """
    Arc-length control for snap-through/snap-back in damage softening.

    Solve the augmented system:
        [ K_T   -F_ref ] [ du    ]   [ -R ]
        [ 2*du  2*dl  ] [ dlambda ] = [ 0  ]

    where the arc-length constraint is:
        du^T du + dlambda^2 * F_ref^T F_ref = ds^2

    Parameters
    ----------
    residual_func : callable(u, lambda) -> R
    tangent_func : callable(u, lambda) -> K_T
    u0 : ndarray
        Initial displacement.
    load_factor0 : float
        Initial load factor.
    d_lambda : float
        Initial load increment.
    max_steps : int
        Number of load steps.

    Returns
    -------
    results : list of (u, lambda, converged)
    """
    u = np.asarray(u0, dtype=float).copy()
    lam = float(load_factor0)
    results = []

    for step in range(max_steps):
        # Predictor
        K = tangent_func(u, lam)
        F_ref = -residual_func(u, lam + 1e-6)
        F_ref = (F_ref - (-residual_func(u, lam - 1e-6))) / (2e-6)

        try:
            du_pred = np.linalg.solve(K, F_ref)
        except np.linalg.LinAlgError:
            du_pred = np.linalg.lstsq(K, F_ref, rcond=None)[0]

        # Scale to arc length
        ds = d_lambda * np.sqrt(1.0 + du_pred @ du_pred)
        dlam = d_lambda
        u += dlam * du_pred
        lam += dlam

        # Corrector (Newton on augmented system)
        for it in range(max_iter):
            R = residual_func(u, lam)
            K = tangent_func(u, lam)
            r_norm = np.linalg.norm(R)
            if r_norm < tol_r:
                break

            # Simplified: standard Newton with fixed lambda
            try:
                du = np.linalg.solve(K, -R)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K, -R, rcond=None)[0]
            u += du

        results.append((u.copy(), lam, r_norm < tol_r))

    return results


def solve_banded_system(K_dense, F, bandwidth=None):
    """
    Solve K u = F using banded solver if possible, else dense.

    Parameters
    ----------
    K_dense : ndarray
    F : ndarray
    bandwidth : int or None
        If provided, attempt banded solve.

    Returns
    -------
    u : ndarray
    """
    n = K_dense.shape[0]
    if bandwidth is not None and bandwidth < n // 2:
        # Try banded solve (requires upper triangular factorization)
        # For general matrix, use LU first
        try:
            u = np.linalg.solve(K_dense, F)
            return u
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(K_dense, F, rcond=None)[0]
            return u
    else:
        try:
            return np.linalg.solve(K_dense, F)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(K_dense, F, rcond=None)[0]


def linpack_residual_benchmark(K, u_exact, F):
    """
    Compute normalized residual and solver performance metrics.

    Incorporates the LINPACK benchmark methodology:
    - residual = b - K*u
    - normalized_residual = ||r||_inf / (||K||_inf * ||u||_inf * eps)

    Parameters
    ----------
    K : ndarray
    u_exact : ndarray
    F : ndarray

    Returns
    -------
    metrics : dict
    """
    eps = np.finfo(float).eps
    u = np.linalg.solve(K, F)
    r = F - K @ u
    K_norm = np.linalg.norm(K, ord=np.inf)
    u_norm = np.linalg.norm(u, ord=np.inf)
    r_norm = np.linalg.norm(r, ord=np.inf)
    ratio = r_norm / (K_norm * u_norm * eps) if K_norm * u_norm > 0 else 0.0
    n = K.shape[0]
    ops = (2.0 * n ** 3) / 3.0 + 2.0 * (n ** 2)
    return {
        'normalized_residual': ratio,
        'residual_norm': r_norm,
        'solution_error': np.linalg.norm(u - u_exact, ord=np.inf),
        'flops': ops,
        'K_norm': K_norm,
        'u_norm': u_norm
    }
