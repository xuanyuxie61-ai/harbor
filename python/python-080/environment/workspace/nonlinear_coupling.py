
import numpy as np
from numpy.linalg import solve, lstsq, norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres


def coupled_residual(U, params):
    R = max(float(U[0]), 1e-9)
    dRdt = float(U[1])
    T_g = max(float(U[2]), 1.0)
    n_g = max(float(U[3]), 1e-20)
    a2 = float(U[4])
    da2dt = float(U[5])
    a3 = float(U[6])
    da3dt = float(U[7])

    p_inf = float(params['p_inf'])
    sigma = float(params['sigma'])
    rho = float(params['rho'])
    mu = float(params['mu'])
    R0 = float(params['R0'])
    p_g0 = float(params['p_g0'])
    gamma = float(params.get('gamma', 1.4))
    R_eq = float(params.get('R_eq', R0))


    pass

    F = np.zeros(8, dtype=float)


    c = float(params.get('c_sound', 1482.0))
    term1 = (p_g - p_inf) / rho
    term2 = -2.0 * sigma / (rho * R)
    term3 = -4.0 * mu * dRdt / (rho * R)
    F[0] = dRdt
    rhs_rp = term1 + term2 + term3 - 1.5 * (dRdt ** 2)
    F[1] = rhs_rp / R

    F[1] = np.clip(F[1], -1e15, 1e15)


    thermal_diff = 1.43e-7
    sqrt_term = np.sqrt(thermal_diff / (np.pi * (R ** 2) + 1e-30))
    F[2] = -3.0 * (gamma - 1.0) * dRdt * (T_g - 293.15) * sqrt_term / R
    F[2] = np.clip(F[2], -1e10, 1e10)


    D_gas = 2.0e-9
    diff_term = p_g - p_g0
    F[3] = -4.0 * np.pi * (R ** 2) * D_gas * diff_term / (8.314 * T_g * R0 + 1e-30)
    F[3] = np.clip(F[3], -1e10, 1e10)


    n = 2
    R3_safe = max(rho * (R ** 3), 1e-30)
    R2_safe = max(rho * (R ** 2), 1e-30)
    restoring2 = -(n - 1) * sigma / R3_safe * (n + 2) * (n - 1) * a2
    damping2 = -2.0 * mu * (n - 1) * (n + 2) / R2_safe * da2dt
    F1_safe = F[1]
    coupling2 = (n - 1) * F1_safe / R * a2
    expansion2 = -3.0 * dRdt / R * da2dt
    F[4] = da2dt
    F[5] = restoring2 + damping2 + coupling2 + expansion2
    F[5] = np.clip(F[5], -1e12, 1e12)


    n = 3
    restoring3 = -(n - 1) * sigma / R3_safe * (n + 2) * (n - 1) * a3
    damping3 = -2.0 * mu * (n - 1) * (n + 2) / R2_safe * da3dt
    coupling3 = (n - 1) * F1_safe / R * a3
    expansion3 = -3.0 * dRdt / R * da3dt
    F[6] = da3dt
    F[7] = restoring3 + damping3 + coupling3 + expansion3
    F[7] = np.clip(F[7], -1e12, 1e12)

    return F


def coupled_jacobian(U, params, h=1e-8):
    n = len(U)
    J = np.zeros((n, n), dtype=float)
    F0 = coupled_residual(U, params)
    for j in range(n):
        Up = U.copy()
        Up[j] += h
        Fp = coupled_residual(Up, params)
        J[:, j] = (Fp - F0) / h
    return J


def solve_coupled_newton(U0, params, max_iter=30, tol=1e-10, damping=0.7):
    U = np.array(U0, dtype=float)
    history = []

    for k in range(max_iter):
        F = coupled_residual(U, params)
        res_norm = norm(F)
        history.append(res_norm)

        if res_norm < tol:
            return U, True, history

        J = coupled_jacobian(U, params)

        try:
            dU = solve(J, -F)
        except np.linalg.LinAlgError:
            dU = -lstsq(J, F, rcond=None)[0]


        alpha = 1.0
        for _ in range(10):
            U_trial = U + alpha * dU
            F_trial = coupled_residual(U_trial, params)
            if norm(F_trial) < res_norm:
                break
            alpha *= damping

        U = U + alpha * dU


        U[0] = max(U[0], 1e-9)
        U[2] = max(U[2], 1.0)
        U[3] = max(U[3], 1e-20)

    return U, False, history


def solve_coupled_picard(U0, params, max_iter=50, tol=1e-10):
    U = np.array(U0, dtype=float)
    history = []
    dt_relax = 0.1

    for k in range(max_iter):
        F = coupled_residual(U, params)
        res_norm = norm(F)
        history.append(res_norm)

        if res_norm < tol:
            return U, True, history


        U_new = U.copy()
        U_new[0] = max(U[0] + dt_relax * F[0], 1e-9)
        U_new[1] += dt_relax * F[1]
        U_new[2] = max(U[2] + dt_relax * F[2], 1.0)
        U_new[3] = max(U[3] + dt_relax * F[3], 1e-20)
        U_new[4] += dt_relax * F[4]
        U_new[5] += dt_relax * F[5]
        U_new[6] += dt_relax * F[6]
        U_new[7] += dt_relax * F[7]

        U = 0.5 * U + 0.5 * U_new

    return U, False, history


def bifurcation_analysis(params_base, p_inf_range):
    results = []
    for p_inf in p_inf_range:
        params = params_base.copy()
        params['p_inf'] = p_inf
        U0 = np.array([1e-5, 0.0, 293.15, 1e-12, 0.0, 0.0, 0.0, 0.0], dtype=float)
        U, converged, _ = solve_coupled_picard(U0, params, max_iter=100, tol=1e-8)
        if converged:
            results.append((p_inf, U[0], U[4], U[6]))
        else:
            results.append((p_inf, np.nan, np.nan, np.nan))
    return np.array(results)
