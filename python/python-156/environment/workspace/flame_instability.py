
import numpy as np


def flame_instability_deriv(t, y, S_L=0.4, rho_u=1.2, rho_b=0.2,
                            gamma_damp=50.0, omega_0=200.0,
                            lam_thermal=100.0, mu_acoustic=500.0):
    xi = y[0]
    v = y[1]


    a_thermal = rho_u * S_L ** 2
    b_thermal = rho_b * S_L ** 2

    F_thermal = (lam_thermal * np.sin(mu_acoustic * t)
                 - a_thermal * max(xi, 0.0)
                 + b_thermal * max(-xi, 0.0))



    phase_lag = 0.1
    F_acoustic = 50.0 * np.cos(mu_acoustic * t + phase_lag) * max(xi, 0.0)

    dxi_dt = v
    dv_dt = -2.0 * gamma_damp * v - omega_0 ** 2 * xi + F_thermal + F_acoustic

    return np.array([dxi_dt, dv_dt])


def integrate_flame_instability(t_span, y0, dt=1.0e-5, **kwargs):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_arr = np.linspace(t_start, t_end, n_steps)

    y_arr = np.zeros((n_steps, 2))
    y_arr[0] = y0

    for i in range(n_steps - 1):
        t = t_arr[i]
        y = y_arr[i]

        k1 = flame_instability_deriv(t, y, **kwargs)
        k2 = flame_instability_deriv(t + dt / 2.0, y + dt / 2.0 * k1, **kwargs)
        k3 = flame_instability_deriv(t + dt / 2.0, y + dt / 2.0 * k2, **kwargs)
        k4 = flame_instability_deriv(t + dt, y + dt * k3, **kwargs)

        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


        y_new[0] = np.clip(y_new[0], -0.1, 0.1)
        y_new[1] = np.clip(y_new[1], -10.0, 10.0)

        y_arr[i + 1] = y_new

    return t_arr, y_arr


def darrieus_landau_growth_rate(k, S_L=0.4, rho_u=1.2, rho_b=0.2):
    alpha = rho_u / rho_b
    term = (alpha / (1.0 + alpha)) * np.sqrt((1.0 + alpha) + (alpha - 1.0) ** 2)
    sigma = S_L * k * (term - alpha)


    k_stabilize = 2.0 * np.pi / 1.0e-3
    stabilization = np.exp(-(k / k_stabilize) ** 2)
    sigma = sigma * stabilization

    return sigma
