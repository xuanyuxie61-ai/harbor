
import numpy as np


def backward_euler_step(y_n, f_func, dt, max_picard_iter=10, tol=1e-10):
    y_np1 = np.copy(y_n)

    for _ in range(max_picard_iter):
        y_new = y_n + dt * f_func(y_np1)
        diff = np.linalg.norm(y_new - y_np1)
        y_np1 = y_new
        if diff < tol:
            break

    return y_np1


def backward_euler_linear(A_mat, y_n, dt, b_vec=None):
    n = len(y_n)
    I = np.eye(n)

    lhs = I - dt * A_mat
    rhs = np.copy(y_n)
    if b_vec is not None:
        rhs = rhs + dt * b_vec


    cond = np.linalg.cond(lhs)
    if cond > 1e15:

        lhs = lhs + 1e-12 * I

    y_np1 = np.linalg.solve(lhs, rhs)
    return y_np1


def exp_exact_solution(t, alpha, t0, y0):
    return y0 * np.exp(alpha * (t - t0))


def exp_deriv(t, y, alpha):
    return alpha * y


def compute_cfl_limit(u, v, w, dx, dy, dz, cfl_number=0.5):
    u_max = np.max(np.abs(u)) + 1e-12
    v_max = np.max(np.abs(v)) + 1e-12
    w_max = np.max(np.abs(w)) + 1e-12

    dt_cfl = cfl_number / (u_max / dx + v_max / dy + w_max / dz)
    return dt_cfl


def compute_diffusion_limit(nu, dx, dy, dz, safety=0.5):
    dt_diff = safety / (nu * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
    return dt_diff


def adaptive_timestep(u, v, w, dx, dy, dz, nu_eff, cfl=0.5):
    dt_cfl = compute_cfl_limit(u, v, w, dx, dy, dz, cfl)
    dt_diff = compute_diffusion_limit(nu_eff, dx, dy, dz)
    dt = min(dt_cfl, dt_diff)

    dt = max(1e-8, min(dt, 10.0))
    return dt
