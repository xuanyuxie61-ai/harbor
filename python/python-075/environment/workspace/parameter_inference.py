
import numpy as np


def svd_linear_least_squares(A, b):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    tol = 1e-12 * s[0] if len(s) > 0 else 1e-12
    s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s])
    x = np.dot(Vt.T, s_inv * np.dot(U.T, b))
    residual = np.linalg.norm(np.dot(A, x) - b)
    return x, residual, s


def fit_turbulent_burning_velocity(u_prime_over_sl, st_over_sl):
    u = np.asarray(u_prime_over_sl)
    s = np.asarray(st_over_sl)


    valid = (u > 0) & (s > 1.0)
    u = u[valid]
    s = s[valid]

    if len(u) < 2:
        return 1.0, 1.0, 0.0


    X = np.log(u)
    Y = np.log(s - 1.0)

    A = np.vstack([np.ones_like(X), X]).T
    coeffs, residual, _ = svd_linear_least_squares(A, Y)
    ln_C, n = coeffs
    C = np.exp(ln_C)


    ss_res = residual**2
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return C, n, r2


def fit_multi_parameter_correlation(Re_t, Da, st_over_sl):
    Re = np.asarray(Re_t)
    Da_arr = np.asarray(Da)
    s = np.asarray(st_over_sl)

    valid = (Re > 0) & (Da_arr > 0) & (s > 0)
    Re = Re[valid]
    Da_arr = Da_arr[valid]
    s = s[valid]

    if len(Re) < 3:
        return np.zeros(3), 0.0

    X1 = np.log(Re)
    X2 = np.log(Da_arr)
    Y = np.log(s)

    A = np.vstack([np.ones_like(X1), X1, X2]).T
    coeffs, residual, _ = svd_linear_least_squares(A, Y)
    a0, a1, a2 = coeffs

    ss_res = residual**2
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return np.array([a0, a1, a2]), r2


def predict_turbulent_flame_speed(u_prime, S_L, l_t, nu, C=1.5, n=0.7):
    ratio = u_prime / max(S_L, 1e-12)
    st = S_L * (1.0 + C * ratio**n)
    Re_t = u_prime * l_t / max(nu, 1e-12)
    delta_L = nu / max(S_L, 1e-12)
    Ka = ratio * np.sqrt(delta_L / max(l_t, 1e-12))
    return st, Re_t, Ka


def turbulent_flame_regime_diagram(u_prime, S_L, l_t, delta_L):
    ratio = u_prime / max(S_L, 1e-12)
    Re_L = S_L * l_t / max(delta_L * u_prime, 1e-12)
    Da = l_t * S_L / max(u_prime * delta_L, 1e-12)
    Ka = ratio * np.sqrt(delta_L / max(l_t, 1e-12))

    if ratio < 1.0:
        regime = "laminar"
    elif ratio < np.sqrt(Re_L):
        regime = "wrinkled_flamelets"
    elif ratio < Da:
        regime = "corrugated_flamelets"
    elif ratio < Ka:
        regime = "thin_reaction_zones"
    else:
        regime = "broken_reaction_zones"

    return regime, Re_L, Da, Ka


def compute_dns_turbulent_burning_velocity(c_field, u_field, v_field, dx, dy, dt, S_L):
    burned_volume = np.sum(c_field > 0.5) * dx * dy


    front_mask = ((c_field > 0.4) & (c_field < 0.6))
    front_length = np.sum(front_mask) * np.sqrt(dx * dy)

    if front_length < 1e-12:
        return S_L




    delta_L = 3.0 * dx
    omega = S_L / delta_L * c_field * (1.0 - c_field)
    consumption = np.sum(omega) * dx * dy

    st = consumption / front_length
    return st
