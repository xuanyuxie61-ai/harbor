
import numpy as np
from sparse_assembler import r8but_solve, dense_to_banded_upper


def newton_raphson(residual_func, tangent_func, u0,
                   tol_r=1e-6, tol_u=1e-8, max_iter=50,
                   line_search=True, verbose=False):
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


        K_arr = np.asarray(K, dtype=float)
        if K_arr.ndim == 0:
            K_arr = np.array([[float(K_arr)]])
        elif K_arr.ndim == 1:
            K_arr = np.atleast_2d(K_arr).T
        try:
            du = np.linalg.solve(K_arr, -R)
        except np.linalg.LinAlgError:

            du = -np.linalg.lstsq(K_arr, R, rcond=None)[0]

        du_norm = np.linalg.norm(du)

        if du_norm / u_norm < tol_u:
            history['iterations'] = it
            return u, True, history


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
    u = np.asarray(u0, dtype=float).copy()
    lam = float(load_factor0)
    results = []

    for step in range(max_steps):

        K = tangent_func(u, lam)
        F_ref = -residual_func(u, lam + 1e-6)
        F_ref = (F_ref - (-residual_func(u, lam - 1e-6))) / (2e-6)

        try:
            du_pred = np.linalg.solve(K, F_ref)
        except np.linalg.LinAlgError:
            du_pred = np.linalg.lstsq(K, F_ref, rcond=None)[0]


        ds = d_lambda * np.sqrt(1.0 + du_pred @ du_pred)
        dlam = d_lambda
        u += dlam * du_pred
        lam += dlam


        for it in range(max_iter):
            R = residual_func(u, lam)
            K = tangent_func(u, lam)
            r_norm = np.linalg.norm(R)
            if r_norm < tol_r:
                break


            try:
                du = np.linalg.solve(K, -R)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K, -R, rcond=None)[0]
            u += du

        results.append((u.copy(), lam, r_norm < tol_r))

    return results


def solve_banded_system(K_dense, F, bandwidth=None):
    n = K_dense.shape[0]
    if bandwidth is not None and bandwidth < n // 2:


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
