
import numpy as np
from math import sqrt






def build_diffusion_tensor(D_f, D_t, fiber_angle):
    c = np.cos(fiber_angle)
    s = np.sin(fiber_angle)
    
    Dxx = D_f * c ** 2 + D_t * s ** 2
    Dxy = (D_f - D_t) * s * c
    Dyy = D_f * s ** 2 + D_t * c ** 2
    
    return Dxx, Dxy, Dyy






def anisotropic_laplacian_5point(u, Dxx, Dxy, Dyy, dx, dy):








    raise NotImplementedError("Hole 2: anisotropic_laplacian_5point 待实现")


def isotropic_laplacian_5point(u, dx, dy):
    nx, ny = u.shape
    lap = np.zeros_like(u)
    
    lap[1:nx - 1, 1:ny - 1] = (
        (u[2:nx, 1:ny - 1] - 2.0 * u[1:nx - 1, 1:ny - 1] + u[0:nx - 2, 1:ny - 1]) / (dx ** 2) +
        (u[1:nx - 1, 2:ny] - 2.0 * u[1:nx - 1, 1:ny - 1] + u[1:nx - 1, 0:ny - 2]) / (dy ** 2)
    )
    

    lap[0, :] = lap[1, :]
    lap[nx - 1, :] = lap[nx - 2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, ny - 1] = lap[:, ny - 2]
    
    return lap






def forward_euler_step(u, v, D, dx, dy, dt, reaction_func, params):

    if isinstance(D, tuple) and len(D) == 3:
        Dxx, Dxy, Dyy = D
        lap_u = anisotropic_laplacian_5point(u, Dxx, Dxy, Dyy, dx, dy)
    else:
        lap_u = isotropic_laplacian_5point(u, dx, dy) * D
    

    f, g = reaction_func(u, v, **params)
    

    u_new = u + dt * (lap_u + f)
    v_new = v + dt * g
    

    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def crank_nicolson_step(u, v, D, dx, dy, dt, reaction_func, params,
                        cg_tol=1e-10, max_cg_iter=500):
    nx, ny = u.shape
    n = nx * ny
    

    mu = nx
    a = np.zeros((mu + 1, n))
    
    coeff = 0.5 * dt * D
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i

            diag_val = 1.0
            if i > 0:
                diag_val += coeff / (dx ** 2)
                a[mu - 1, idx + 1] = -coeff / (dx ** 2)
            if i < nx - 1:
                diag_val += coeff / (dx ** 2)
            if j > 0:
                diag_val += coeff / (dy ** 2)
                a[mu - nx, idx + nx] = -coeff / (dy ** 2)
            if j < ny - 1:
                diag_val += coeff / (dy ** 2)
            a[mu, idx] = diag_val
    

    lap_u = isotropic_laplacian_5point(u, dx, dy)
    f, g = reaction_func(u, v, **params)
    
    rhs = u + 0.5 * dt * D * lap_u + dt * f
    rhs_flat = rhs.flatten()
    

    from linear_algebra_core import r8pbu_cg
    u_new_flat, res, iters = r8pbu_cg(n, mu, a, rhs_flat, np.zeros(n),
                                      tol=cg_tol, max_iter=max_cg_iter)
    
    u_new = u_new_flat.reshape((nx, ny))
    v_new = v + dt * g
    
    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def adi_step(u, v, D, dx, dy, dt, reaction_func, params):
    nx, ny = u.shape
    f, g = reaction_func(u, v, **params)
    

    rx = 0.5 * dt * D / (dx ** 2)
    u_star = np.zeros_like(u)
    
    for j in range(ny):

        a_tri = np.full(nx, -rx)
        b_tri = np.full(nx, 1.0 + 2.0 * rx)
        c_tri = np.full(nx, -rx)
        

        d_tri = np.zeros(nx)
        for i in range(nx):
            d_tri[i] = u[i, j]
            if j > 0:
                d_tri[i] += 0.5 * dt * D * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1 if j < ny - 1 else j]) / (dy ** 2)
            d_tri[i] += 0.5 * dt * f[i, j]
        

        b_tri[0] = 1.0
        c_tri[0] = 0.0
        d_tri[0] = u[0, j]
        a_tri[nx - 1] = 0.0
        b_tri[nx - 1] = 1.0
        d_tri[nx - 1] = u[nx - 1, j]
        
        u_star[:, j] = _solve_tridiagonal(a_tri, b_tri, c_tri, d_tri)
    

    ry = 0.5 * dt * D / (dy ** 2)
    u_new = np.zeros_like(u)
    
    f_star, g_star = reaction_func(u_star, v, **params)
    
    for i in range(nx):
        a_tri = np.full(ny, -ry)
        b_tri = np.full(ny, 1.0 + 2.0 * ry)
        c_tri = np.full(ny, -ry)
        
        d_tri = np.zeros(ny)
        for j in range(ny):
            d_tri[j] = u_star[i, j]
            if i > 0:
                d_tri[j] += 0.5 * dt * D * (u_star[i - 1, j] - 2 * u_star[i, j] +
                                               u_star[i + 1 if i < nx - 1 else i, j]) / (dx ** 2)
            d_tri[j] += 0.5 * dt * f_star[i, j]
        

        b_tri[0] = 1.0
        c_tri[0] = 0.0
        d_tri[0] = u_star[i, 0]
        a_tri[ny - 1] = 0.0
        b_tri[ny - 1] = 1.0
        d_tri[ny - 1] = u_star[i, ny - 1]
        
        u_new[i, :] = _solve_tridiagonal(a_tri, b_tri, c_tri, d_tri)
    
    v_new = v + dt * g
    
    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def _solve_tridiagonal(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def apply_boundary_conditions(field):
    nx, ny = field.shape
    field[0, :] = field[1, :]
    field[nx - 1, :] = field[nx - 2, :]
    field[:, 0] = field[:, 1]
    field[:, ny - 1] = field[:, ny - 2]
    return field






def solve_reaction_diffusion_2d(u0, v0, D, dx, dy, dt, T,
                                 reaction_func, reaction_params,
                                 solver='forward_euler',
                                 stimulus_func=None,
                                 stimulus_region=None):
    n_steps = int(T / dt)
    nx, ny = u0.shape
    
    u = u0.copy()
    v = v0.copy()
    

    save_interval = max(1, n_steps // 100)
    n_saved = n_steps // save_interval + 1
    
    u_history = np.zeros((n_saved, nx, ny))
    v_history = np.zeros((n_saved, nx, ny))
    t_history = np.zeros(n_saved)
    
    u_history[0] = u
    v_history[0] = v
    t_history[0] = 0.0
    
    save_idx = 1
    
    for step in range(1, n_steps + 1):

        if stimulus_func is not None and stimulus_region is not None:
            stim = stimulus_func(step * dt)
            u += stimulus_region * stim * dt
        

        if solver == 'forward_euler':
            u, v = forward_euler_step(u, v, D, dx, dy, dt,
                                       reaction_func, reaction_params)
        elif solver == 'crank_nicolson':
            u, v = crank_nicolson_step(u, v, D, dx, dy, dt,
                                        reaction_func, reaction_params)
        elif solver == 'adi':
            u, v = adi_step(u, v, D, dx, dy, dt,
                            reaction_func, reaction_params)
        else:
            u, v = forward_euler_step(u, v, D, dx, dy, dt,
                                       reaction_func, reaction_params)
        

        if step % save_interval == 0 and save_idx < n_saved:
            u_history[save_idx] = u
            v_history[save_idx] = v
            t_history[save_idx] = step * dt
            save_idx += 1
    

    u_history = u_history[:save_idx]
    v_history = v_history[:save_idx]
    t_history = t_history[:save_idx]
    
    return u_history, v_history, t_history






def generate_fiber_angle_field(nx, ny, model='parallel'):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if model == 'parallel':
        angle = np.zeros((nx, ny))
    elif model == 'rotational':

        angle = -np.pi / 3.0 + (2.0 * np.pi / 3.0) * Y
    elif model == 'radial':

        angle = np.arctan2(Y - 0.5, X - 0.5)
    else:
        angle = np.zeros((nx, ny))
    
    return angle
