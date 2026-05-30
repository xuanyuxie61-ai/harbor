
import numpy as np
import math
from typing import Callable, Tuple





def neutrino_parameters() -> dict:
    return {
        'diffusion_coeff': 1.0e4,
        'convection_velocity': 1.0e7,
        'thermal_conductivity': 1.0e23,
        'heat_capacity': 1.0e20,
        'weak_rate': 1.0e-2,
        'neutrino_luminosity_coeff': 1.0e25,
        't0': 0.0,
        'tstop': 10.0,
        'xmin': 0.0,
        'xmax': 1.0e5,
    }


def neutrino_coefficients(x: float, t: float, state: np.ndarray,
                          dstate_dx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = neutrino_parameters()
    D = params['diffusion_coeff']
    v = params['convection_velocity']
    K_th = params['thermal_conductivity']
    c_v = params['heat_capacity']
    weak_rate = params['weak_rate']
    lum_coeff = params['neutrino_luminosity_coeff']

    Y_e, T = state
    dYedx, dTdx = dstate_dx


    Y_e = np.clip(Y_e, 0.0, 1.0)
    T = max(T, 1.0e3)

    c_vec = np.array([1.0, 1.0])


    f_vec = np.array([
        D * dYedx + v * Y_e,
        (K_th / c_v) * dTdx + v * T
    ])



    Y_eq = 0.05 + 0.1 * math.exp(-T / 1.0e9)
    source_Y = -weak_rate * (Y_e - Y_eq)



    Q_nu = -lum_coeff * (T / 1.0e9)**6

    s_vec = np.array([source_Y, Q_nu])

    return c_vec, f_vec, s_vec





def solve_neutrino_diffusion_1d(
    nx: int = 200,
    nt: int = 5000,
    t_final: float = 10.0,
    L: float = 1.0e5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = neutrino_parameters()
    D = params['diffusion_coeff']
    v = params['convection_velocity']
    K_th = params['thermal_conductivity']
    c_v = params['heat_capacity']
    alpha_th = K_th / c_v

    dx = L / nx

    dt_diff = dx**2 / (2.0 * max(D, alpha_th))
    dt_conv = dx / max(abs(v), 1.0e-10)
    dt = min(dt_diff, dt_conv, t_final / nt)
    nt_actual = int(t_final / dt) + 1

    x = np.linspace(0.0, L, nx + 1)
    t = np.linspace(0.0, t_final, nt_actual)


    Y_e = np.ones(nx + 1) * 0.3
    T = np.ones(nx + 1) * 1.0e9

    solution = np.zeros((nt_actual, nx + 1, 2))
    solution[0, :, 0] = Y_e
    solution[0, :, 1] = T

    for n in range(nt_actual - 1):
        Y_new = Y_e.copy()
        T_new = T.copy()

        for i in range(1, nx):

            diff_Y = D * (Y_e[i + 1] - 2.0 * Y_e[i] + Y_e[i - 1]) / dx**2
            diff_T = alpha_th * (T[i + 1] - 2.0 * T[i] + T[i - 1]) / dx**2


            if v > 0.0:
                conv_Y = v * (Y_e[i] - Y_e[i - 1]) / dx
                conv_T = v * (T[i] - T[i - 1]) / dx
            else:
                conv_Y = v * (Y_e[i + 1] - Y_e[i]) / dx
                conv_T = v * (T[i + 1] - T[i]) / dx


            Y_eq = 0.05 + 0.1 * math.exp(-T[i] / 1.0e9)
            source_Y = -params['weak_rate'] * (Y_e[i] - Y_eq)
            source_T = -params['neutrino_luminosity_coeff'] * (T[i] / 1.0e9)**6

            Y_new[i] = Y_e[i] + dt * (diff_Y - conv_Y + source_Y)
            T_new[i] = T[i] + dt * (diff_T - conv_T + source_T)



        Y_new[0] = Y_new[1]
        T_new[0] = T_new[1]

        Y_new[nx] = 0.1
        T_new[nx] = 5.0e8


        Y_new = np.clip(Y_new, 0.0, 1.0)
        T_new = np.clip(T_new, 1.0e3, 1.0e12)

        Y_e = Y_new
        T = T_new
        solution[n + 1, :, 0] = Y_e
        solution[n + 1, :, 1] = T

    return x, t, solution


def compute_neutrino_luminosity(
    T_profile: np.ndarray,
    dx: float,
    R_star: float = 1.0e6
) -> float:
    params = neutrino_parameters()
    lum_coeff = params['neutrino_luminosity_coeff']

    Q_vol = lum_coeff * (T_profile / 1.0e9)**6
    integral = np.trapz(Q_vol, dx=dx)

    L_nu = 4.0 * math.pi * R_star**2 * integral
    return L_nu


def compute_deleptonization_timescale(
    Y_e_initial: float,
    Y_e_final: float,
    weak_rate: float = 1.0e-2
) -> float:
    if Y_e_initial <= 0.0 or weak_rate <= 0.0:
        return float('inf')
    return abs(Y_e_final - Y_e_initial) / (weak_rate * max(Y_e_initial, 1e-10))
