
import numpy as np
from linear_solvers import solve_sparse_system, solve_tridiagonal


class NonlinearSolverError(Exception):
    pass


def newton_solve_burgers_style(u_init, residual_func, jacobian_func,
                                max_iter=50, tol=1e-10, damping=1.0,
                                min_damping=0.1):
    u = np.asarray(u_init, dtype=float).copy()
    n = u.size

    for it in range(max_iter):
        F = residual_func(u)
        f_norm = np.linalg.norm(F, np.inf)

        if f_norm < tol:
            return u, {"converged": True, "iter": it, "resid": f_norm}

        J = jacobian_func(u)


        try:
            du = solve_sparse_system(J, -F)
        except Exception as exc:

            du = np.linalg.lstsq(J, -F, rcond=None)[0]

        du_norm = np.linalg.norm(du, np.inf)
        u_norm = np.linalg.norm(u, np.inf)
        if du_norm < tol * (u_norm + 1.0):
            return u, {"converged": True, "iter": it, "resid": f_norm}


        omega = damping
        while omega >= min_damping:
            u_new = u + omega * du
            F_new = residual_func(u_new)
            f_new_norm = np.linalg.norm(F_new, np.inf)
            if f_new_norm < f_norm:
                u = u_new
                break
            omega *= 0.5
        else:

            u = u + min_damping * du

    return u, {"converged": False, "iter": max_iter, "resid": f_norm}


def solve_coupled_diffusion_reaction_newton(r_nodes, D_e, lambda_eff,
                                             kinetics_model, particle_model,
                                             max_iter=50, tol=1e-8):
    n = r_nodes.size
    if n < 3:
        raise NonlinearSolverError("节点数至少为 3")

    C_surf = particle_model.C_surface_A
    T_surf = particle_model.T_surface


    C = np.linspace(C_surf * 0.85, C_surf, n)
    T = np.linspace(T_surf + 20.0, T_surf, n)
    C[-1] = C_surf
    T[-1] = T_surf

    for it in range(max_iter):

















        raise NotImplementedError("Hole 2: 请实现耦合 C-T 方程的块迭代离散与求解")


        change_C = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        change_T = np.linalg.norm(T_new - T) / max(np.linalg.norm(T), 1e-12)
        C = 0.6 * C_new + 0.4 * C
        T = 0.6 * T_new + 0.4 * T

        if max(change_C, change_T) < tol:
            return C, T, {"converged": True, "iter": it + 1, "resid": max(change_C, change_T)}

    return C, T, {"converged": False, "iter": max_iter, "resid": max(change_C, change_T)}


def pseudo_transient_continuation(r_nodes, D_e, lambda_eff,
                                   kinetics_model, particle_model,
                                   dt_init=1e-6, dt_max=1.0, t_final=100.0):
    n = r_nodes.size
    C = np.ones(n) * particle_model.C_surface_A * 0.5
    T = np.ones(n) * particle_model.T_surface
    C[-1] = particle_model.C_surface_A
    T[-1] = particle_model.T_surface

    t = 0.0
    dt = dt_init
    step = 0

    while t < t_final and step < 10000:

        a_diag = np.ones(n) / dt
        b_sub = np.zeros(n - 1)
        c_sup = np.zeros(n - 1)
        rhs_C = C / dt
        rhs_T = T / dt

        for i in range(1, n - 1):
            rm = r_nodes[i]
            dr_p = r_nodes[i + 1] - r_nodes[i]
            dr_m = r_nodes[i] - r_nodes[i - 1]
            r_plus = 0.5 * (r_nodes[i] + r_nodes[i + 1])
            r_minus = 0.5 * (r_nodes[i] + r_nodes[i - 1])
            vol = rm ** 2 * 0.5 * (dr_p + dr_m)

            a_diag[i] += (r_plus ** 2 * D_e / dr_p + r_minus ** 2 * D_e / dr_m) / vol / dt * 0.0

            Ci = max(float(C[i]), 0.0)
            Ti = max(float(T[i]), 200.0)
            R_local = kinetics_model.rate(Ci, particle_model.C_surface_B, Ti)
            rhs_C[i] += -R_local
            rhs_T[i] += R_local * (-particle_model.heat_of_reaction) / (1000.0)


        a_diag[0] = 1.0
        rhs_C[0] = C[1]
        rhs_T[0] = T[1]
        a_diag[-1] = 1.0
        rhs_C[-1] = particle_model.C_surface_A
        rhs_T[-1] = particle_model.T_surface


        C_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs_C)
        T_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs_T)

        change = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        C = C_new
        T = T_new
        t += dt
        step += 1

        dt = min(dt * 1.1, dt_max)
        if change < 1e-8:
            break

    return C, T, {"steps": step, "time": t, "change": change}
