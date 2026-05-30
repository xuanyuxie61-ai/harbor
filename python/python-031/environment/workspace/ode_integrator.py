# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp


K_B = 8.617333262e-11
HBARC = 197.3269804


def unstable_exact(t, mu):
    if mu == 0.0:
        raise ValueError("mu不能为零")
    exp_term = np.exp(mu * t)
    cos_term = np.cos(t / mu)
    sin_term = np.sin(t / mu)
    y1 = exp_term * (cos_term - mu**2 * sin_term)
    y2 = exp_term * (sin_term / mu + mu * cos_term)
    return y1, y2


def unstable_deriv(t, y, mu):
    u, v = y
    if mu == 0.0:
        return np.array([0.0, 0.0])
    dudt = mu * u + (1.0 / mu) * v
    dvdt = -(1.0 / mu) * u + mu * v
    return np.array([dudt, dvdt])


def tough_deriv(t, y):
    y1, y2, y3, y4 = y

    if y1 <= 0.0:
        y1 = 1e-15
    if y2 <= 0.0:
        y2 = 1e-15

    dy1dt = 2.0 * t * (y2**0.2) * y4
    dy2dt = 10.0 * t * np.exp(5.0 * (y2 - 1.0)) * y4
    dy3dt = 2.0 * t * y4
    dy4dt = -2.0 * t * np.log(y1)
    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])


def neutrino_luminosity(temperature, rho, proton_fraction, m_star=1.0):
    if temperature <= 0.0 or rho <= 0.0:
        return 0.0

    T_8 = temperature / (K_B * 1e8)
    x_p = proton_fraction


    coeff = 4.13e27

    coeff_MeV = coeff * 6.2415e5 / 1e39

    epsilon = coeff_MeV * (m_star**3) * (x_p**(1.0/3.0)) * (T_8**8)
    return epsilon


def heat_capacity_degenerate(rho, proton_fraction, temperature):
    if temperature <= 0.0 or rho <= 0.0:
        return 0.0

    rho_p = rho * proton_fraction
    m_p = 938.272
    k_fp = (3.0 * np.pi**2 * rho_p)**(1.0/3.0)
    epsilon_F = k_fp**2 / (2.0 * m_p)

    if epsilon_F <= 0.0:
        return 0.0

    N0 = 3.0 * rho_p / (2.0 * epsilon_F)
    C_V = (np.pi**2 / 2.0) * N0 * K_B**2 * temperature
    return max(0.0, C_V)


def crust_cooling_ode(t, y, rho, proton_fraction, m_star=1.0,
                      heating_rate=0.0):
    T = y[0]
    if T <= 0.0:
        return np.array([0.0, 0.0])

    C_V = heat_capacity_degenerate(rho, proton_fraction, T)
    if C_V <= 1e-30:
        return np.array([0.0, heating_rate])

    eps_nu = neutrino_luminosity(T, rho, proton_fraction, m_star)

    dTdt = (-eps_nu + heating_rate) / C_V
    dQdt = heating_rate


    if T + dTdt < 0.0:
        dTdt = -T * 0.1

    return np.array([dTdt, dQdt])


def solve_crust_cooling(t_span, T0, rho, proton_fraction,
                        m_star=1.0, heating_rate=0.0, method='RK45'):
    y0 = [T0, 0.0]

    def deriv(t, y):
        return crust_cooling_ode(t, y, rho, proton_fraction, m_star, heating_rate)

    sol = solve_ivp(deriv, t_span, y0, method=method, max_step=t_span[1]/100.0,
                    dense_output=True, rtol=1e-6, atol=1e-9)
    return sol


def phase_transition_kinetics(t, psi, T, T_c, gamma, a0, b, c):
    if T_c <= 0.0:
        return 0.0
    a_T = a0 * (T - T_c) / T_c
    dF_dpsi = 2.0 * a_T * psi + 4.0 * b * psi**3
    dpsi_dt = -gamma * dF_dpsi
    return dpsi_dt


def solve_unstable_system(t_span, y0, mu, n_points=200):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)


    y1_exact, y2_exact = unstable_exact(t_eval, mu)


    sol = solve_ivp(lambda t, y: unstable_deriv(t, y, mu),
                    t_span, y0, t_eval=t_eval, method='RK45',
                    rtol=1e-9, atol=1e-12)

    return t_eval, np.array([y1_exact, y2_exact]), sol.y


def solve_tough_system(t_span, y0, method='RK45'):
    sol = solve_ivp(tough_deriv, t_span, y0, method=method,
                    dense_output=True, rtol=1e-8, atol=1e-10)
    return sol


if __name__ == '__main__':

    t = np.linspace(0, 1, 10)
    y1e, y2e = unstable_exact(t, mu=0.5)
    print(f"unstable exact: y1[0]={y1e[0]:.4f}, y1[-1]={y1e[-1]:.4f}")

    sol = solve_crust_cooling([0, 1e5], T0=1.0, rho=0.1, proton_fraction=0.3)
    print(f"crust cooling: T_final={sol.y[0,-1]:.6f} MeV")
