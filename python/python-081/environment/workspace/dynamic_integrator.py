
import numpy as np


class DynamicIntegratorError(Exception):
    pass


def sawtooth_wave(t, period, amplitude):
    t_mod = t % period
    return (2.0 * amplitude / period) * t_mod - amplitude


def compute_rayleigh_damping(M, K_T, alpha_m, beta_k):
    return alpha_m * M + beta_k * K_T


def newmark_predictor(u_n, v_n, a_n, dt, beta=0.25, gamma=0.5):
    u_pred = u_n + dt * v_n + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * a_n
    v_pred = v_n + dt * (1.0 - gamma) * a_n
    return u_pred, v_pred


def newmark_corrector(u_pred, v_pred, du, dt, beta=0.25, gamma=0.5):
    u_new = u_pred + du
    a_new = du / (beta * dt ** 2)
    v_new = v_pred + gamma * dt * a_new
    return u_new, v_new, a_new


def trapezoidal_step(u_n, v_n, a_n, dt, M, C, compute_internal_force,
                     compute_tangent_stiffness, F_ext,
                     tol=1e-8, max_iter=20):
    n_dof = len(u_n)
    coeff_a = 4.0 / (dt ** 2)
    coeff_v = 2.0 / dt


    u = u_n + dt * v_n + 0.5 * dt ** 2 * a_n

    for it in range(max_iter):
        R_int, K_T = compute_internal_force(u), compute_tangent_stiffness(u)


        a = coeff_a * (u - u_n) - coeff_v * v_n - a_n
        v = coeff_v * (u - u_n) - v_n


        residual = M @ a + C @ v + R_int - F_ext
        res_norm = np.linalg.norm(residual)

        if res_norm < tol:
            return u, v, a, True


        K_eff = coeff_a * M + coeff_v * C + K_T


        try:
            du = np.linalg.solve(K_eff, -residual)
        except np.linalg.LinAlgError:

            K_eff += 1e-6 * np.eye(n_dof) * np.max(np.abs(K_eff))
            du = np.linalg.solve(K_eff, -residual)

        u = u + du


    a = coeff_a * (u - u_n) - coeff_v * v_n - a_n
    v = coeff_v * (u - u_n) - v_n
    return u, v, a, False


def dynamic_analysis_trapezoidal(u0, v0, a0, t_span, n_steps,
                                  M, C_func, compute_internal_force,
                                  compute_tangent_stiffness, F_ext_func,
                                  tol=1e-8, max_iter=20):
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    n_dof = len(u0)

    t_array = np.linspace(t_start, t_end, n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n_dof))
    v_hist = np.zeros((n_steps + 1, n_dof))
    a_hist = np.zeros((n_steps + 1, n_dof))

    u_hist[0] = u0
    v_hist[0] = v0
    a_hist[0] = a0

    u = u0.copy()
    v = v0.copy()
    a = a0.copy()

    for i in range(n_steps):
        t_new = t_array[i + 1]
        F_ext = F_ext_func(t_new)
        K_T = compute_tangent_stiffness(u)
        C = C_func(K_T)

        u, v, a, converged = trapezoidal_step(
            u, v, a, dt, M, C,
            compute_internal_force, compute_tangent_stiffness, F_ext,
            tol=tol, max_iter=max_iter
        )

        u_hist[i + 1] = u
        v_hist[i + 1] = v
        a_hist[i + 1] = a

    return t_array, u_hist, v_hist, a_hist
