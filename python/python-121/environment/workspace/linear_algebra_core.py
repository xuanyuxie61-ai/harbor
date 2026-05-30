
import numpy as np
from math import sqrt






def r8pbu_mv(n, mu, a, x):
    if n <= 0 or mu < 0 or mu > n - 1:
        return np.zeros(n)
    y = np.zeros(n)
    for i in range(n):

        y[i] += a[mu, i] * x[i]

        for j in range(i + 1, min(i + mu + 1, n)):
            a_val = a[mu + i - j, j]
            y[i] += a_val * x[j]
            y[j] += a_val * x[i]
    return y


def r8pbu_cg(n, mu, a, b, x0, tol=1e-12, max_iter=None):
    if max_iter is None:
        max_iter = n
    
    b = np.asarray(b, dtype=float).flatten()
    x = np.asarray(x0, dtype=float).flatten().copy()
    
    if n <= 0 or mu < 0:
        return x, float('inf'), 0
    

    ap = r8pbu_mv(n, mu, a, x)
    r = b - ap
    p = r.copy()
    
    rs_old = np.dot(r, r)
    rs0 = rs_old
    
    if rs0 == 0.0:
        return x, 0.0, 0
    
    for it in range(1, max_iter + 1):
        ap = r8pbu_mv(n, mu, a, p)
        pap = np.dot(p, ap)
        
        if pap == 0.0:
            break
        
        alpha = rs_old / pap
        x += alpha * p
        r -= alpha * ap
        
        rs_new = np.dot(r, r)
        residual_norm = sqrt(rs_new)
        
        if residual_norm / sqrt(rs0) < tol:
            return x, residual_norm, it
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    return x, sqrt(rs_old), max_iter


def build_laplacian_banded(nx, ny, dx, dy):
    n = nx * ny
    mu = nx
    a = np.zeros((mu + 1, n))
    
    h2 = dx * dy
    if h2 == 0:
        h2 = 1.0
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i

            diag_val = 0.0
            if i > 0:
                diag_val += 1.0 / dx ** 2
            if i < nx - 1:
                diag_val += 1.0 / dx ** 2
            if j > 0:
                diag_val += 1.0 / dy ** 2
            if j < ny - 1:
                diag_val += 1.0 / dy ** 2
            a[mu, idx] = diag_val
            

            if i < nx - 1:
                a[mu - 1, idx + 1] = -1.0 / dx ** 2
            

            if j < ny - 1:
                a[mu - nx, idx + nx] = -1.0 / dy ** 2
    
    return n, mu, a






def power_method(A, y0, it_max=1000, tol=1e-10):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y0, dtype=float).flatten().copy()
    n = A.shape[0]
    
    if n == 0:
        return y, 0.0, 0
    
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        y = np.ones(n)
        norm_y = sqrt(n)
    y = y / norm_y
    
    ay = A.dot(y)
    lambda_val = np.dot(y, ay)
    y = ay / np.linalg.norm(ay)
    if lambda_val < 0:
        y = -y
    
    for it_num in range(1, it_max + 1):
        lambda_old = lambda_val
        y_old = y.copy()
        
        ay = A.dot(y)
        lambda_val = np.dot(y, ay)
        norm_ay = np.linalg.norm(ay)
        if norm_ay == 0:
            break
        y = ay / norm_ay
        if lambda_val < 0:
            y = -y
        
        val_dif = abs(lambda_val - lambda_old)
        

        cos_yy = np.dot(y, y_old)
        sin_yy = sqrt(max(0.0, (1.0 - cos_yy) * (1.0 + cos_yy)))
        
        if val_dif <= tol and sin_yy <= tol:

            y = ay / lambda_val if lambda_val != 0 else y
            return y, lambda_val, it_num
    
    if lambda_val != 0:
        y = ay / lambda_val
    return y, lambda_val, it_max


def stability_eigenvalue_analysis(diffusion_matrix, reaction_jacobian):
    A = diffusion_matrix + reaction_jacobian

    n = A.shape[0]
    y0 = np.random.randn(n)
    _, lambda_max, _ = power_method(A, y0, it_max=min(n, 500), tol=1e-8)
    
    is_stable = lambda_max < 0
    return lambda_max, is_stable






def solve_poisson_2d_cg(f, nx, ny, dx, dy, boundary_value=0.0, max_iter=None):
    n = nx * ny
    n_total, mu, a = build_laplacian_banded(nx, ny, dx, dy)
    
    b = np.asarray(f, dtype=float).flatten()
    x0 = np.zeros(n)
    


    
    phi_flat, res, iters = r8pbu_cg(n_total, mu, a, b, x0, tol=1e-10, max_iter=max_iter)
    phi = phi_flat.reshape((nx, ny))
    return phi, res, iters
