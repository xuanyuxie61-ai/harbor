
import numpy as np


def rk4_stability_function(z):
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def low_storage_rk54_stability_function(z):
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0 + 0.02 * z ** 5


def evaluate_stability_region(R_func, x_range, y_range, npts=401):
    x = np.linspace(x_range[0], x_range[1], npts)
    y = np.linspace(y_range[0], y_range[1], npts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    Rval = R_func(Z)
    Rabs = np.abs(Rval)
    return X, Y, Rabs


def check_eigenvalue_in_stability_region(eigenvalues, dt, R_func):
    z = dt * np.asarray(eigenvalues)
    R_vals = R_func(z)
    stable = np.abs(R_vals) <= 1.0 + 1e-10
    return stable


def compute_max_stable_timestep(eigenvalues, R_func):
    ev = np.asarray(eigenvalues)
    real_neg = ev[np.real(ev) < 0]
    if len(real_neg) == 0:
        return np.inf


    lambda_max = np.max(np.real(real_neg))
    lambda_min = np.min(np.real(real_neg))

    def is_stable(dt):
        z = dt * real_neg
        return np.all(np.abs(R_func(z)) <= 1.0 + 1e-10)

    dt_low = 0.0
    dt_high = 10.0 / abs(lambda_min)


    while is_stable(dt_high) and dt_high < 1e6:
        dt_high *= 2.0

    if is_stable(dt_high):
        return dt_high

    for _ in range(50):
        dt_mid = 0.5 * (dt_low + dt_high)
        if is_stable(dt_mid):
            dt_low = dt_mid
        else:
            dt_high = dt_mid

    return dt_low


def analyze_damage_jacobian_eigenvalues(damage_state, stress, params):
    if hasattr(damage_state, 'to_array'):
        d_arr = damage_state.to_array()
    else:
        d_arr = np.asarray(damage_state)
    d_f, d_m, d_s, d_i = d_arr
    sigma1, sigma2, tau12 = stress


    J = np.zeros((4, 4))


    if d_f < 0.99:
        J[0, 0] = params.k_f * ((abs(sigma1) / params.sigma_f0) ** params.m_f) / (
            (1.0 - d_f) ** (params.k_f + 1.0) + 1e-12)


    if d_m < 0.99:
        J[1, 1] = params.k_m * ((abs(sigma2) / params.sigma_m0) ** params.m_m) / (
            (1.0 - d_m) ** (params.k_m + 1.0) + 1e-12)


    if d_s < 0.99:
        J[2, 2] = params.k_s * ((abs(tau12) / params.tau_s0) ** params.m_s) / (
            (1.0 - d_s) ** (params.k_s + 1.0) + 1e-12)


    a_debond = 0.81
    epsilon = params.epsilon_debond
    J[3, 3] = -1.0 / epsilon * (3.0 * d_i ** 2 - a_debond)

    eigvals = np.linalg.eigvals(J)
    return eigvals


def recommend_time_integrator(damage_state, stress, params):
    eigvals = analyze_damage_jacobian_eigenvalues(damage_state, stress, params)
    max_dt_rk4 = compute_max_stable_timestep(eigvals, rk4_stability_function)
    max_dt_rk54 = compute_max_stable_timestep(eigvals, low_storage_rk54_stability_function)

    ratio = np.max(np.abs(np.real(eigvals))) / (np.min(np.abs(np.real(eigvals))) + 1e-12)

    if ratio > 100.0:
        method = "Implicit BDF2 (stiff ratio > 100)"
    elif ratio > 10.0:
        method = "Implicit Midpoint or RK4 with very small dt"
    else:
        method = "Explicit RK4 or Low-Storage RK54"

    return {
        'method': method,
        'max_dt_rk4': max_dt_rk4,
        'max_dt_rk54': max_dt_rk54,
        'stiffness_ratio': ratio,
        'eigenvalues': eigvals
    }
