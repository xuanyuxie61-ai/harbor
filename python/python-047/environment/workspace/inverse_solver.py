
import numpy as np
from matrix_kernels import r8utt_solve, r8utt_to_dense, r8utt_inverse, \
    toeplitz_matvec, tikhonov_preconditioner_toeplitz, football_combination_count, \
    sparse_approximate_inverse_mod


def build_sensitivity_matrix(obs_points, grid_centers, grid_volumes, use_toeplitz=False):
    G_CONST = 6.67430e-11
    
    obs = np.asarray(obs_points, dtype=float)
    centers = np.asarray(grid_centers, dtype=float)
    vols = np.asarray(grid_volumes, dtype=float)
    
    N_obs = obs.shape[0]
    N_param = centers.shape[0]
    
    if use_toeplitz:

        first_row = np.zeros(N_param)
        for j in range(N_param):
            dx = obs[0, 0] - centers[j, 0]
            dy = obs[0, 1] - centers[j, 1]
            dz = obs[0, 2] - centers[j, 2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r = max(r, 1e-6)
            first_row[j] = G_CONST * vols[j] * dz / (r**3) * 1e5
        return first_row
    













    raise NotImplementedError("Hole_2: build_sensitivity_matrix 格林函数核待实现")


def tikhonov_solve_dense(G, d, alpha, order=1):
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    N_obs, N_param = G.shape
    

    if order == 1:
        L = np.zeros((N_param - 1, N_param))
        for i in range(N_param - 1):
            L[i, i] = 1.0
            L[i, i + 1] = -1.0
    elif order == 2:
        L = np.zeros((N_param - 2, N_param))
        for i in range(N_param - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
    else:
        raise ValueError("order must be 1 or 2")
    

    A = G.T @ G + alpha**2 * (L.T @ L)
    b = G.T @ d
    

    try:

        m = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        m = np.linalg.lstsq(A, b, rcond=1e-10)[0]
    
    residual = np.linalg.norm(G @ m - d)
    reg_term = np.linalg.norm(L @ m)
    
    return m, residual, reg_term


def tikhonov_solve_toeplitz(green_row, d, alpha, grid_shape, dx, dy, dz):
    N_param = len(green_row)
    


    a = np.zeros(N_param)
    for lag in range(N_param):
        s = 0.0
        for k in range(N_param - lag):
            s += green_row[k] * green_row[k + lag]
        a[lag] = s
    a[0] += alpha**2
    


    b = np.zeros(N_param)
    for j in range(N_param):
        b[j] = green_row[j] * np.sum(d)
    


    try:
        m = r8utt_solve(N_param, a, b)
    except ValueError:
        a[0] += 1e-12
        m = r8utt_solve(N_param, a, b)
    
    return m


def l_curve_criterion(G, d, alphas, order=1):
    residuals = []
    reg_terms = []
    
    for alpha in alphas:
        m, res, reg = tikhonov_solve_dense(G, d, alpha, order)
        residuals.append(res)
        reg_terms.append(reg)
    
    residuals = np.array(residuals)
    reg_terms = np.array(reg_terms)
    

    log_res = np.log10(residuals + 1e-15)
    log_reg = np.log10(reg_terms + 1e-15)
    

    n = len(alphas)
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        dx1 = log_res[i] - log_res[i - 1]
        dy1 = log_reg[i] - log_reg[i - 1]
        dx2 = log_res[i + 1] - log_res[i]
        dy2 = log_reg[i + 1] - log_reg[i]
        
        ds1 = np.sqrt(dx1**2 + dy1**2)
        ds2 = np.sqrt(dx2**2 + dy2**2)
        
        if ds1 > 1e-15 and ds2 > 1e-15:
            ddx = dx2 / ds2 - dx1 / ds1
            ddy = dy2 / ds2 - dy1 / ds1
            curvatures[i] = abs(ddx * dy1 / ds1 - ddy * dx1 / ds1)
    
    best_idx = np.argmax(curvatures)
    best_alpha = alphas[best_idx]
    
    return best_alpha, residuals, reg_terms, curvatures


def gcv_criterion(G, d, alphas, order=1):
    N_obs = G.shape[0]
    gcv_values = []
    residuals = []
    
    for alpha in alphas:
        m, res, reg = tikhonov_solve_dense(G, d, alpha, order)
        residuals.append(res)
        



        try:
            s = np.linalg.svd(G, compute_uv=False)
            s = s[s > 1e-12]
            trace_h = np.sum(s**2 / (s**2 + alpha**2))
        except:
            trace_h = N_obs * 0.5
        
        denom = max(N_obs - trace_h, 1e-3)
        gcv = res**2 / (denom**2)
        gcv_values.append(gcv)
    
    best_idx = np.argmin(gcv_values)
    best_alpha = alphas[best_idx]
    
    return best_alpha, np.array(gcv_values), np.array(residuals)


def iterative_tikhonov_cg(G, d, alpha, order=1, max_iter=500, tol=1e-6):
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    N_obs, N_param = G.shape
    

    def apply_L(v):
        if order == 1:
            Lv = np.zeros(N_param - 1)
            for i in range(N_param - 1):
                Lv[i] = v[i] - v[i + 1]
            return Lv
        else:
            Lv = np.zeros(N_param - 2)
            for i in range(N_param - 2):
                Lv[i] = v[i] - 2.0 * v[i + 1] + v[i + 2]
            return Lv
    
    def apply_LT(w):
        if order == 1:
            LTw = np.zeros(N_param)
            LTw[0] = w[0]
            for i in range(1, N_param - 1):
                LTw[i] = w[i] - w[i - 1]
            LTw[-1] = -w[-1]
            return LTw
        else:
            LTw = np.zeros(N_param)
            LTw[0] = w[0]
            LTw[1] = -2.0 * w[0] + w[1]
            for i in range(2, N_param - 2):
                LTw[i] = w[i - 2] - 2.0 * w[i - 1] + w[i]
            LTw[-2] = w[-3] - 2.0 * w[-2]
            LTw[-1] = w[-2]
            return LTw
    
    def matvec(v):
        return G.T @ (G @ v) + alpha**2 * apply_LT(apply_L(v))
    
    b = G.T @ d
    m = np.zeros(N_param)
    r = b - matvec(m)
    p = r.copy()
    
    residual_norms = [np.linalg.norm(r)]
    
    for k in range(max_iter):
        Ap = matvec(p)
        rTr = np.dot(r, r)
        pAp = np.dot(p, Ap)
        
        if abs(pAp) < 1e-15:
            break
        
        alpha_cg = rTr / pAp
        m = m + alpha_cg * p
        r_new = r - alpha_cg * Ap
        rTr_new = np.dot(r_new, r_new)
        
        residual_norms.append(np.sqrt(rTr_new))
        
        if np.sqrt(rTr_new) < tol * np.linalg.norm(b):
            break
        
        beta_cg = rTr_new / rTr
        p = r_new + beta_cg * p
        r = r_new
    
    return m, len(residual_norms), np.array(residual_norms)


def resolution_matrix_analysis(G, alpha, order=1):
    N_obs, N_param = G.shape
    
    if order == 1:
        L = np.zeros((N_param - 1, N_param))
        for i in range(N_param - 1):
            L[i, i] = 1.0
            L[i, i + 1] = -1.0
    else:
        L = np.zeros((N_param - 2, N_param))
        for i in range(N_param - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
    
    A = G.T @ G + alpha**2 * (L.T @ L)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    
    R_m = A_inv @ (G.T @ G)
    R_d = G @ A_inv @ G.T
    

    spread_m = np.sum((R_m - np.eye(N_param))**2)
    trace_d = np.trace(R_d)
    
    return R_m, R_d, spread_m, trace_d
