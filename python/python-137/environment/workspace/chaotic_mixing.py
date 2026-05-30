# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp


def chen_attractor_rhs(t, state, a=40.0, b=3.0, c=28.0):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt], dtype=float)


def generate_chaotic_mixing_trajectory(t_span, y0=None, params=None,
                                        method='RK45', rtol=1e-8, atol=1e-10):
    if y0 is None:
        y0 = np.array([-0.1, 0.5, -0.6], dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float)

    if params is None:
        params = {'a': 40.0, 'b': 3.0, 'c': 28.0}

    def rhs(t, y):
        return chen_attractor_rhs(t, y, **params)

    sol = solve_ivp(rhs, t_span, y0, method=method,
                    dense_output=True, rtol=rtol, atol=atol,
                    max_step=(t_span[1] - t_span[0]) / 1000)
    return sol


def map_chen_to_supersaturation_fluctuation(t, sol, sigma_base,
                                             scale_T=0.5, scale_c=0.3):
    t = np.asarray(t, dtype=float)

    t0, tf = sol.t[0], sol.t[-1]
    t = np.clip(t, t0, tf)

    states = sol.sol(t)
    x = states[0, :]
    y = states[1, :]





    d_sigma = scale_c * y - scale_T * x
    sigma_local = sigma_base + d_sigma

    sigma_local = np.clip(sigma_local, 0.0, 5.0)
    return sigma_local


def mixing_enhanced_nucleation_rate(sigma_base, t, sol, B0, scale_T=0.5, scale_c=0.3):
    sigma_local = map_chen_to_supersaturation_fluctuation(t, sol, sigma_base,
                                                           scale_T, scale_c)


    S = 1.0 + sigma_local
    S = np.where(S <= 1.0, 1.0 + 1e-6, S)
    lnS = np.log(S)

    lnS = np.where(np.abs(lnS) < 1e-6, 1e-6, lnS)
    A = 16.0
    B_instant = B0 * np.exp(-A / (lnS ** 2))
    B_avg = np.mean(B_instant)
    return B_avg, B_instant
