
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def midpoint_implicit_step(K_uu, C, M_p, K_p, F_u, F_p, u_n, p_n, dt,
                           solver_type="dense"):
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n_u = K_uu.shape[0]
    n_p = M_p.shape[0]


    rhs_u = F_u - C @ p_n
    try:
        u_new = np.linalg.solve(K_uu, rhs_u)
    except (np.linalg.LinAlgError, ValueError):

        reg = 1e-8 * np.eye(n_u)
        u_new = np.linalg.lstsq(K_uu + reg, rhs_u, rcond=None)[0]


    A_p = M_p + 0.5 * dt * K_p
    rhs_p = (M_p - 0.5 * dt * K_p) @ p_n - C.T @ (u_new - u_n)
    rhs_p += 0.5 * dt * F_p

    try:
        p_new = np.linalg.solve(A_p, rhs_p)
    except (np.linalg.LinAlgError, ValueError):
        reg = 1e-8 * np.eye(n_p)
        p_new = np.linalg.lstsq(A_p + reg, rhs_p, rcond=None)[0]

    return u_new, p_new


def imex_splitting_step(K_uu, C, M_p, K_p, F_u, F_p,
                        u_n, p_n, dt, explicit_ratio=0.5):
    if not (0.0 <= explicit_ratio <= 1.0):
        raise ValueError("explicit_ratio must be in [0, 1].")

    n_u = K_uu.shape[0]
    n_p = M_p.shape[0]


    if explicit_ratio > 1e-14:
        try:
            p_star = p_n - explicit_ratio * dt * (
                np.linalg.solve(M_p, K_p @ p_n - F_p)
            )
        except np.linalg.LinAlgError:
            p_star = p_n.copy()
    else:
        p_star = p_n.copy()


    rhs_u = F_u - C @ p_star
    try:
        u_new = np.linalg.solve(K_uu, rhs_u)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(K_uu, rhs_u, rcond=None)[0]


    A_p = M_p + (1.0 - explicit_ratio) * dt * K_p
    rhs_p = M_p @ p_n - C.T @ (u_new - u_n) + (1.0 - explicit_ratio) * dt * F_p
    try:
        p_new = np.linalg.solve(A_p, rhs_p)
    except np.linalg.LinAlgError:
        p_new = np.linalg.lstsq(A_p, rhs_p, rcond=None)[0]

    return u_new, p_new


def exponential_integrator_exact(alpha, t0, y0, tstop, n_steps):
    t = np.linspace(t0, tstop, n_steps + 1)
    y = y0 * np.exp(alpha * (t - t0))
    return t, y


def dynamic_time_stepping(M_uu, C, M_p, K_uu, K_p, F_u_func, F_p_func,
                          u0, v0, p0, tspan, n_steps):
    dt = (tspan[1] - tspan[0]) / n_steps
    beta = 0.25
    gamma = 0.5

    n_u = M_uu.shape[0]
    n_p = M_p.shape[0]

    u = u0.copy()
    v = v0.copy()
    a = np.zeros_like(u)
    p = p0.copy()


    rhs = F_u_func(tspan[0]) - K_uu @ u - C @ p
    try:
        a = np.linalg.solve(M_uu, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(M_uu, rhs, rcond=None)[0]

    u_hist = np.zeros((n_steps + 1, n_u))
    p_hist = np.zeros((n_steps + 1, n_p))
    u_hist[0, :] = u
    p_hist[0, :] = p


    K_eff = K_uu + (1.0 / (beta * dt ** 2)) * M_uu

    for n in range(n_steps):
        t = tspan[0] + n * dt
        t_next = t + dt


        u_pred = u + dt * v + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * a
        v_pred = v + dt * (1.0 - gamma) * a


        F_u = F_u_func(t_next)
        F_p = F_p_func(t_next)



        A_p = M_p + dt * K_p
        rhs_p = M_p @ p - C.T @ (v_pred + dt * a) + dt * F_p
        try:
            p_new = np.linalg.solve(A_p, rhs_p)
        except np.linalg.LinAlgError:
            p_new = np.linalg.lstsq(A_p, rhs_p, rcond=None)[0]


        rhs_u = F_u - C @ p_new + (1.0 / (beta * dt ** 2)) * M_uu @ u_pred
        try:
            u_new = np.linalg.solve(K_eff, rhs_u)
        except np.linalg.LinAlgError:
            u_new = np.linalg.lstsq(K_eff, rhs_u, rcond=None)[0]


        a_new = (u_new - u_pred) / (beta * dt ** 2)
        v_new = v_pred + gamma * dt * a_new

        u = u_new
        v = v_new
        a = a_new
        p = p_new

        u_hist[n + 1, :] = u
        p_hist[n + 1, :] = p

    return u_hist, p_hist


def compute_cfl_condition(Vmax, hmin, safety_factor=0.5):
    if Vmax <= 0.0 or hmin <= 0.0:
        raise ValueError("Vmax and hmin must be positive.")
    return safety_factor * hmin / Vmax
