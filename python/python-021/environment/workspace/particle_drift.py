
import numpy as np
from parameters import get_drift_params


def drift_derivative(t, y, i1, i2, i3, E_field=None, collision_freq=0.0):
    y = np.asarray(y, dtype=float)
    if y.shape != (3,):
        raise ValueError("状态向量必须为 3 维")

    vR, vZ, vpar = y


    dvR_dt = (1.0 / i3 - 1.0 / i2) * vZ * vpar
    dvZ_dt = (1.0 / i1 - 1.0 / i3) * vR * vpar
    dvpar_dt = (1.0 / i2 - 1.0 / i1) * vR * vZ


    if E_field is not None:
        E = np.asarray(E_field(t), dtype=float)
        if E.shape != (3,):
            raise ValueError("电场向量必须为 3 维")
        q_over_m = 1.0e8
        dvR_dt += q_over_m * E[0]
        dvZ_dt += q_over_m * E[1]
        dvpar_dt += q_over_m * E[2]


    dvR_dt -= collision_freq * vR
    dvZ_dt -= collision_freq * vZ
    dvpar_dt -= collision_freq * vpar

    return np.array([dvR_dt, dvZ_dt, dvpar_dt], dtype=float)


def rk2_integrate(dydt_func, t_span, y0, n_steps=10000):
    t0, tstop = t_span
    h = (tstop - t0) / n_steps
    if h <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.asarray(y0))
    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, dim))
    t_arr[0] = t0
    y_arr[0, :] = np.asarray(y0, dtype=float)

    for n in range(n_steps):
        tn = t_arr[n]
        yn = y_arr[n, :]
        k1 = h * dydt_func(tn, yn)
        k2 = h * dydt_func(tn + h, yn + k1)
        y_arr[n + 1, :] = yn + 0.5 * (k1 + k2)
        t_arr[n + 1] = tn + h

    return t_arr, y_arr


def simulate_guiding_center(drift_params=None, n_steps=5000):
    if drift_params is None:
        drift_params = get_drift_params()

    i1 = drift_params["i1"]
    i2 = drift_params["i2"]
    i3 = drift_params["i3"]
    t0 = drift_params["t0"]
    y0 = drift_params["y0"]
    tstop = drift_params["tstop"]


    omega = 2.0 * np.pi * 1.0e3
    E0 = np.array([1.0e-4, 0.5e-4, 0.2e-4])

    def E_field(t):
        return E0 * np.cos(omega * t)


    nu_ei = 1.0e2

    def deriv(t, y):
        return drift_derivative(t, y, i1, i2, i3, E_field=E_field, collision_freq=nu_ei)

    t_arr, y_arr = rk2_integrate(deriv, (t0, tstop), y0, n_steps=n_steps)


    energy = 0.5 * (i1 * y_arr[:, 0] ** 2 +
                    i2 * y_arr[:, 1] ** 2 +
                    i3 * y_arr[:, 2] ** 2)

    return t_arr, y_arr, energy


def compute_magnetic_moment(v_perp, B):
    m_eff = 3.34e-27
    B = np.asarray(B)
    B_safe = np.where(np.abs(B) < 1e-15, 1e-15, B)
    return 0.5 * m_eff * np.asarray(v_perp) ** 2 / B_safe


def compute_adiabatic_invariant(y_arr, B_arr):
    v_perp_sq = y_arr[:, 0] ** 2 + y_arr[:, 1] ** 2
    mu_arr = compute_magnetic_moment(np.sqrt(v_perp_sq), B_arr)
    if np.mean(mu_arr) > 1e-30:
        mu_relative_std = np.std(mu_arr) / np.mean(mu_arr)
    else:
        mu_relative_std = 0.0
    return mu_arr, mu_relative_std
