
import numpy as np


def heaviside(x):
    return np.where(x >= 0, 1.0, 0.0)


def anishchenko_deriv(t, xyz, mu=1.2, eta=0.5):
    x, y, z = xyz
    dxdt = mu * x + y - x * z
    dydt = -x
    dzdt = -eta * z + eta * heaviside(x) * x ** 2
    return np.array([dxdt, dydt, dzdt])


def ice_mass_balance(T_surface, orbital_forcing, P_base=0.3, mu_melt=0.008, T_melt=273.15):
    beta_p = 0.05
    T_ref = 273.15
    P = P_base * (1.0 + beta_p * (T_surface - T_ref))
    P = max(P, 0.05)

    melt = mu_melt * max(0.0, T_surface - T_melt)
    melt = min(melt, P * 2.0)


    accumulation_boost = 1.0 + 0.3 * orbital_forcing
    B_m = P * accumulation_boost - melt
    return B_m


def ice_discharge(volume, V_max=50e6, k_discharge=0.001):
    D = k_discharge * max(0.0, volume - 0.7 * V_max)
    return D


def sea_level(volume, V_full_glacial=50e6, dSL_dV=7.4e-6):
    return -dSL_dV * volume


def bedrock_depression(volume, tau_relax=5000.0, rho_ice=917.0, rho_mantle=3300.0):

    A_ice = 14e6
    h_ice = volume / A_ice * 1e3
    depression = -(rho_ice / rho_mantle) * h_ice
    return depression


def coupled_climate_ice_deriv(t, state, orbital_forcing_func, mu=1.2, eta=0.5):
    x, y, z, V_ice, T_global, h_bedrock = state


    F_orb = orbital_forcing_func(t)
    F_orb = np.clip(F_orb, 0.0, 1.0)


    climate_deriv = anishchenko_deriv(t, [x, y, z], mu, eta)



    T_base = 288.0
    delta_T_orbital = 5.0 * (F_orb - 0.5)
    delta_T_chaotic = 1.0 * x
    T_target = T_base + delta_T_orbital + delta_T_chaotic


    tau_temp = 10.0
    dTdt = (T_target - T_global) / tau_temp
    T_global = max(200.0, min(350.0, T_global))


    B_m = ice_mass_balance(T_global, F_orb)
    D = ice_discharge(V_ice)


    A_ice_sheet = 14e6
    dVdt = (B_m * 1e-3 * A_ice_sheet) - D


    tau_bed = 5000.0
    h_target = bedrock_depression(V_ice)
    dhdt = (h_target - h_bedrock) / tau_bed

    return np.array([
        climate_deriv[0],
        climate_deriv[1],
        climate_deriv[2],
        dVdt,
        dTdt,
        dhdt
    ])


def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ice_climate(t_span, y0, orbital_forcing_func, dt=1.0, mu=1.2, eta=0.5):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)
    sol = np.zeros((n_steps, len(y0)))
    sol[0] = y0

    y = np.array(y0, dtype=float)
    for i in range(1, n_steps):
        t = t_array[i - 1]
        y = rk4_step(coupled_climate_ice_deriv, t, y, dt, orbital_forcing_func, mu, eta)

        y[3] = max(0.0, y[3])
        y[4] = np.clip(y[4], 200.0, 350.0)
        sol[i] = y

    return t_array, sol


def compute_ice_line_latitude(T_profile, latitudes_deg):
    T_freeze = 263.15

    for i in range(len(latitudes_deg) - 1):
        if (T_profile[i] - T_freeze) * (T_profile[i + 1] - T_freeze) < 0:

            frac = (T_freeze - T_profile[i]) / (T_profile[i + 1] - T_profile[i])
            return latitudes_deg[i] + frac * (latitudes_deg[i + 1] - latitudes_deg[i])

    below = latitudes_deg[T_profile < T_freeze]
    if len(below) > 0:
        return float(np.max(np.abs(below)))
    return 0.0
