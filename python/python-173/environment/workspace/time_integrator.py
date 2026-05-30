
import numpy as np


def backward_euler_step(u_n, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    u_new = u_n.copy()

    for _ in range(max_iter):
        F_val = rhs_func(u_new)
        residual = u_new - u_n - dt * F_val
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True


        u_new = u_n + dt * F_val

    return u_new, False


def bdf2_step(u_n, u_nm1, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    u_new = u_n.copy()
    coeff = 2.0 * dt / 3.0

    for _ in range(max_iter):
        F_val = rhs_func(u_new)
        residual = u_new - (4.0 * u_n - u_nm1) / 3.0 - coeff * F_val
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True

        u_new = (4.0 * u_n - u_nm1) / 3.0 + coeff * F_val

    return u_new, False


def crank_nicolson_step(u_n, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    F_n = rhs_func(u_n)
    u_new = u_n.copy()

    for _ in range(max_iter):
        F_new = rhs_func(u_new)
        residual = u_new - u_n - 0.5 * dt * (F_n + F_new)
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True


        u_new = u_n + 0.5 * dt * (F_n + F_new)

    return u_new, False


def adaptive_time_stepping(
    u0, t_span, dt_init,
    rhs_func,
    scheme='bdf2',
    rtol=1e-4,
    atol=1e-6,
    dt_min=1e-8,
    dt_max=1.0,
    safety_factor=0.9
):
    t_start, t_end = t_span
    t = t_start
    dt = dt_init
    u = u0.copy()

    t_history = [t]
    u_history = [u.copy()]
    dt_history = [dt]

    u_prev = None

    while t < t_end:
        dt = min(dt, t_end - t)
        if dt < dt_min:
            dt = dt_min
            if t_end - t < dt_min:
                break

        tol = atol + rtol * np.linalg.norm(u)


        u_euler = u + dt * rhs_func(u)


        u_impl, _ = backward_euler_step(u, dt, rhs_func)


        err = np.linalg.norm(u_euler - u_impl)

        if err < tol or dt <= dt_min:

            if scheme == 'bdf1' or scheme == 'backward_euler':
                u = u_impl
            elif scheme == 'bdf2' and u_prev is not None:
                u_new, _ = bdf2_step(u, u_prev, dt, rhs_func)
                u_prev = u.copy()
                u = u_new
            elif scheme == 'crank_nicolson':
                u, _ = crank_nicolson_step(u, dt, rhs_func)
            else:

                u = u_impl
                u_prev = u.copy()

            t += dt
            t_history.append(t)
            u_history.append(u.copy())
            dt_history.append(dt)


            if err > 1e-14:
                dt_new = dt * safety_factor * np.sqrt(tol / err)
            else:
                dt_new = dt * 2.0
            dt = np.clip(dt_new, dt_min, dt_max)
        else:

            dt = max(dt * 0.5, dt_min)

    return t_history, u_history, dt_history


def analyze_stiffness_eigenvalues(A_matrix, M_lumped=None, n_eig=10):
    n = A_matrix.shape[0]
    n_eig = min(n_eig, n)

    if M_lumped is not None:
        M_inv_sqrt = 1.0 / np.sqrt(M_lumped)
        A_scaled = M_inv_sqrt[:, None] * A_matrix * M_inv_sqrt[None, :]
    else:
        A_scaled = A_matrix


    try:
        eigenvalues = np.linalg.eigvals(A_scaled)
        eigenvalues = eigenvalues[np.argsort(np.real(eigenvalues))]

        real_parts = np.real(eigenvalues)
        abs_real = np.abs(real_parts)
        nonzero = abs_real > 1e-14
        if np.sum(nonzero) > 1:
            stiffness_ratio = np.max(abs_real[nonzero]) / np.min(abs_real[nonzero])
        else:
            stiffness_ratio = 1.0

        spectral_radius = np.max(np.abs(eigenvalues))

        return eigenvalues[:n_eig], stiffness_ratio, spectral_radius
    except Exception:

        return np.zeros(n_eig), 1.0, np.linalg.norm(A_scaled, 2)
