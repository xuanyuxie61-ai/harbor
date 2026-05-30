
import numpy as np


def reaction_deriv(t, y, k, K_co2, K_ch4, P_total):
    A = max(y[0], 0.0)
    B = max(y[1], 0.0)
    denom = 1.0 + K_co2 * A + K_ch4 * B
    if denom <= 0.0:
        denom = 1e-30
    theta_co2 = K_co2 * A / denom
    theta_ch4 = K_ch4 * B / denom


    r_co2 = k * theta_co2
    r_ch4 = k * theta_ch4

    dAdt = -r_co2
    dBdt = -r_ch4
    dCdt = r_co2 + r_ch4
    return np.array([dAdt, dBdt, dCdt], dtype=float)


def reaction_parameters():
    return {
        "k": 4.2e-3,
        "K_co2": 1.2e-3,
        "K_ch4": 2.5e-4,
        "P_total": 5.0e6,
        "y0": np.array([150.0, 800.0, 0.0], dtype=float),
        "tspan": (0.0, 3600.0),
    }


def kepler_like_trajectory_deriv(t, y, mu=1.0):
    eps = 1e-12
    q1 = y[0]
    q2 = y[1]
    p1 = y[2]
    p2 = y[3]
    r2 = q1 * q1 + q2 * q2
    r_eff = np.sqrt(r2 + eps * eps)
    r_eff3 = r_eff ** 3

    dq1dt = p1
    dq2dt = p2
    dp1dt = -mu * q1 / r_eff3
    dp2dt = -mu * q2 / r_eff3
    return np.array([dq1dt, dq2dt, dp1dt, dp2dt], dtype=float)


def kepler_parameters():
    return {
        "mu": 1.0e-20,
        "y0": np.array([1e-9, 0.0, 0.0, 1e-4], dtype=float),
        "tspan": (0.0, 1e-6),
    }


def quasiperiodic_forcing_deriv(t, y, omega1=np.pi, omega2=1.0):
    dydt = np.zeros(4, dtype=float)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = y[3]
    dydt[3] = -(omega1 ** 2 + 1.0) * y[2] - (omega1 ** 2) * y[0]
    return dydt


def quasiperiodic_parameters():
    return {
        "omega1": np.pi,
        "omega2": 1.0,
        "y0": np.array([0.01, 0.0, -0.01 * np.pi ** 2, 0.0], dtype=float),
        "tspan": (0.0, 10.0),
    }


def runge_function(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_derivative(x):
    x = np.asarray(x, dtype=float)
    denom = (1.0 + 25.0 * x * x) ** 2
    return -50.0 * x / denom


def runge_second_derivative(x):
    x = np.asarray(x, dtype=float)
    x2 = x * x
    num = 50.0 * (75.0 * x2 - 1.0)
    denom = (1.0 + 25.0 * x2) ** 3
    return num / denom


def power_series_runge(x, n_terms=10):
    x = np.asarray(x, dtype=float)
    val = np.zeros_like(x, dtype=float)
    for k in range(n_terms):
        coeff = (-1.0) ** k * (5.0 ** (2 * k))
        val += coeff * (x ** (2 * k))
    return val


def coupled_membrane_reaction_ode(t, y, params):
    k = params["k_reaction"]
    K_co2 = params["K_ads_co2"]
    K_ch4 = params["K_ads_ch4"]
    omega1 = params.get("omega1", np.pi)
    h_mt = params.get("h_mt", 1e-4)


    surf = y[:3]
    dsurf = reaction_deriv(t, surf, k, K_co2, K_ch4, 1.0)


    c_bulk_co2 = max(y[3], 0.0)
    c_bulk_ch4 = max(y[4], 0.0)
    dc_bulk_co2 = -h_mt * (c_bulk_co2 - max(surf[0], 0.0))
    dc_bulk_ch4 = -h_mt * (c_bulk_ch4 - max(surf[1], 0.0))


    qp = y[5:9]
    dqp = quasiperiodic_forcing_deriv(t, qp, omega1)

    return np.concatenate([dsurf, [dc_bulk_co2, dc_bulk_ch4], dqp])
