import numpy as np
from constants import TINY, MAX_ITER




def build_fd_laplacian_1d(n, h):
    A = np.zeros((n, n))
    inv_h2 = 1.0 / (h * h)
    for i in range(n):
        A[i, i] = 2.0 * inv_h2
        if i > 0:
            A[i, i - 1] = -inv_h2
        if i < n - 1:
            A[i, i + 1] = -inv_h2
    return A





def jacobi_solve(A, f, u0=None, max_iter=MAX_ITER, tol=1.0e-10):
    n = A.shape[0]
    f = np.asarray(f, dtype=float)
    
    if u0 is None:
        u = np.zeros(n)
    else:
        u = np.asarray(u0, dtype=float).copy()
    
    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < TINY):
        raise ValueError("Zero diagonal element in Jacobi iteration")
    
    for it in range(max_iter):
        u_new = np.zeros(n)
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * u[j]
            u_new[i] = (f[i] - sigma) / diag[i]
        

        residual = f - A @ u_new
        res_norm = np.linalg.norm(residual) / np.sqrt(n)
        
        u = u_new
        
        if res_norm < tol:
            return u, res_norm, True, it + 1
    
    residual = f - A @ u
    res_norm = np.linalg.norm(residual) / np.sqrt(n)
    return u, res_norm, False, max_iter





def sor_solve(A, f, omega=1.5, u0=None, max_iter=MAX_ITER, tol=1.0e-10):
    n = A.shape[0]
    f = np.asarray(f, dtype=float)
    
    if u0 is None:
        u = np.zeros(n)
    else:
        u = np.asarray(u0, dtype=float).copy()
    
    diag = np.diag(A)
    if np.any(np.abs(diag) < TINY):
        raise ValueError("Zero diagonal element in SOR")
    
    for it in range(max_iter):
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * u[j]
            u_gs = (f[i] - sigma) / diag[i]
            u[i] = (1.0 - omega) * u[i] + omega * u_gs
        
        residual = f - A @ u
        res_norm = np.linalg.norm(residual) / np.sqrt(n)
        if res_norm < tol:
            return u, res_norm, True, it + 1
    
    residual = f - A @ u
    res_norm = np.linalg.norm(residual) / np.sqrt(n)
    return u, res_norm, False, max_iter





def direct_solve(A, f):
    return np.linalg.solve(A, f)





def smooth_background_poisson(raw_counts, smoothing_strength=1.0, n_inner=50):
    n = len(raw_counts)
    if n < 3:
        return raw_counts.copy()
    

    h = 1.0 / (n_inner + 1)
    A = build_fd_laplacian_1d(n_inner, h)
    A = smoothing_strength * A + np.eye(n_inner)
    

    x_inner = np.linspace(h, 1.0 - h, n_inner)
    x_raw = np.linspace(0.0, 1.0, n)
    f_interp = np.interp(x_inner, x_raw, raw_counts)
    

    f_interp[0] += smoothing_strength * raw_counts[0] / (h * h)
    f_interp[-1] += smoothing_strength * raw_counts[-1] / (h * h)
    

    try:
        u_inner = direct_solve(A, f_interp)
    except np.linalg.LinAlgError:
        u_inner, _, _, _ = sor_solve(A, f_interp, omega=1.2)
    

    smoothed = np.zeros(n)
    smoothed[0] = raw_counts[0]
    smoothed[-1] = raw_counts[-1]
    smoothed[1:-1] = np.interp(x_raw[1:-1], x_inner, u_inner)
    
    return smoothed





def compare_solvers(f_func, exact_func, n=31):
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1.0 - h, n)
    A = build_fd_laplacian_1d(n, h)
    f = np.array([f_func(xi) for xi in x])
    u_exact = np.array([exact_func(xi) for xi in x])
    

    u_direct = direct_solve(A, f)
    err_direct = np.linalg.norm(u_direct - u_exact) / np.linalg.norm(u_exact)
    

    u_jacobi, _, conv_j, it_j = jacobi_solve(A, f, max_iter=20000, tol=1.0e-12)
    err_jacobi = np.linalg.norm(u_jacobi - u_exact) / np.linalg.norm(u_exact)
    

    omega_opt = 2.0 / (1.0 + np.sin(np.pi / (n + 1)))
    u_sor, _, conv_s, it_s = sor_solve(A, f, omega=omega_opt, max_iter=10000, tol=1.0e-12)
    err_sor = np.linalg.norm(u_sor - u_exact) / np.linalg.norm(u_exact)
    
    return {
        "direct_error": err_direct,
        "jacobi_error": err_jacobi,
        "jacobi_converged": conv_j,
        "jacobi_iters": it_j,
        "sor_error": err_sor,
        "sor_converged": conv_s,
        "sor_iters": it_s,
        "omega_opt": omega_opt,
    }
