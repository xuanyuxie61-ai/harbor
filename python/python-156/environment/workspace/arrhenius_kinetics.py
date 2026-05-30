
import numpy as np


ARRHENIUS_PREEXP = 2.0e11
ACTIVATION_ENERGY = 1.26e8
TEMPERATURE_EXPONENT = 0.0


CP_MIX = 1200.0
HEAT_RELEASE_CH4 = 5.0e7


def arrhenius_rate_constant(T, A=ARRHENIUS_PREEXP, Ea=ACTIVATION_ENERGY, n=0.0):
    R_u = 8.314462618
    T = np.maximum(T, 100.0)

    exponent = -Ea / (R_u * T)

    exponent = np.clip(exponent, -700.0, 700.0)

    k = A * (T ** n) * np.exp(exponent)
    return k


def reaction_progress_ode(t, c, T, A_f=None, Ea_f=None, A_b=None, Ea_b=None):
    c = np.clip(c, 0.0, 1.0)

    if A_f is None:
        A_f = ARRHENIUS_PREEXP
    if Ea_f is None:
        Ea_f = ACTIVATION_ENERGY
    if A_b is None:
        A_b = ARRHENIUS_PREEXP * 0.01
    if Ea_b is None:
        Ea_b = ACTIVATION_ENERGY * 0.5

    k_f = arrhenius_rate_constant(T, A_f, Ea_f)
    k_b = arrhenius_rate_constant(T, A_b, Ea_b)

    n_f = 1.0
    n_b = 1.0

    dcdt = k_f * ((1.0 - c) ** n_f) - k_b * (c ** n_b)
    return dcdt


def integrate_progress_variable(T, dt=1.0e-6, n_steps=10000,
                                A_f=None, Ea_f=None, A_b=None, Ea_b=None):
    c = 0.0
    c_history = np.zeros(n_steps + 1)
    t_history = np.zeros(n_steps + 1)

    for i in range(n_steps):
        c_history[i] = c
        t_history[i] = i * dt

        dcdt = reaction_progress_ode(t_history[i], c, T, A_f, Ea_f, A_b, Ea_b)
        c_new = c + dt * dcdt
        c = np.clip(c_new, 0.0, 1.0)

    c_history[-1] = c
    t_history[-1] = n_steps * dt

    return c_history, t_history


def adiabatic_flame_temperature(Y_F0, Y_O0, T0, Q=HEAT_RELEASE_CH4, cp=CP_MIX):
    s = 17.16

    Y_F0 = np.clip(Y_F0, 0.0, 1.0)
    Y_O0 = np.clip(Y_O0, 0.0, 1.0)

    phi = (Y_F0 / Y_O0) / (1.0 / s) if Y_O0 > 1.0e-12 else 0.0

    Y_F_burnt = max(0.0, Y_F0 - min(Y_F0, Y_O0 / s))
    heat_released = Q * (Y_F0 - Y_F_burnt)

    T_ad = T0 + heat_released / cp
    T_ad = min(T_ad, 5000.0)

    return T_ad, phi
