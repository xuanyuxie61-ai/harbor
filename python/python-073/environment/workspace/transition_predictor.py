# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt


def e_n_method(Re_x_array, alpha_i_array, N_cr=9.0):
    Re = np.asarray(Re_x_array)
    ai = np.asarray(alpha_i_array)
    n = len(Re)
    N = np.zeros(n)

    for i in range(1, n):
        dRe = Re[i] - Re[i - 1]
        if dRe <= 0:
            N[i] = N[i - 1]
            continue

        N[i] = N[i - 1] - 0.5 * dRe * (ai[i] + ai[i - 1])


    Re_xt = None
    for i in range(1, n):
        if N[i - 1] < N_cr <= N[i] or N[i] < N_cr <= N[i - 1]:

            frac = (N_cr - N[i - 1]) / (N[i] - N[i - 1])
            Re_xt = Re[i - 1] + frac * (Re[i] - Re[i - 1])
            break

    if Re_xt is None:
        Re_xt = Re[-1] if N[-1] >= N_cr else Re[0]

    return Re_xt, N


def compute_growth_rate_profile(Re_x, Ma=6.0, Re_unit=1e6, Tw_Te=1.0):
    Re = np.asarray(Re_x)

    C = 0.002 * (Ma / 6.0) ** 1.5 * (Tw_Te ** (-0.3))
    p = 0.5
    q = 2.0
    Re_max = 3e6 * (Ma / 6.0) ** (-0.8) * (Tw_Te ** 0.4)

    alpha_i = -C * (Re / 1e6) ** p * np.exp(-(Re / Re_max) ** q)
    return alpha_i


def transition_front_cost(positions, penalties):
    path_diff = np.sum(np.abs(np.diff(positions)))
    penalty_sum = np.sum(penalties)
    return path_diff + penalty_sum


def optimize_transition_front(spanwise_positions, initial_xt, penalties,
                               max_iter=5000, lambda_penalty=0.5):
    n = len(initial_xt)
    xt = initial_xt.copy()
    cost = transition_front_cost(xt, lambda_penalty * penalties)
    cost_history = [cost]

    for _ in range(max_iter):

        i, j = np.random.randint(0, n, size=2)
        if i == j:
            continue

        xt_new = xt.copy()
        xt_new[i], xt_new[j] = xt_new[j], xt_new[i]
        cost_new = transition_front_cost(xt_new, lambda_penalty * penalties)

        if cost_new < cost:
            xt = xt_new
            cost = cost_new
            cost_history.append(cost)
            continue


        i = np.random.randint(0, n)
        j = np.random.randint(0, n - 1)
        xt_new = np.delete(xt, i)
        xt_new = np.insert(xt_new, j, xt[i])
        cost_new = transition_front_cost(xt_new, lambda_penalty * penalties)

        if cost_new < cost:
            xt = xt_new
            cost = cost_new
            cost_history.append(cost)

    return xt, cost_history


def multi_station_transition_prediction(Ma, Re_unit, Tw_Te, Tu,
                                         z_stations, roughness_array,
                                         N_cr=9.0):
    n_stations = len(z_stations)
    Re_xt = np.zeros(n_stations)
    N_profiles = []

    for i in range(n_stations):

        Re_x = np.linspace(1e5, 1e7, 500)


        ai = compute_growth_rate_profile(Re_x, Ma, Re_unit, Tw_Te)



        ksd = roughness_array[i]
        N_cr_eff = max(2.0, N_cr - 3.0 * (ksd ** 0.5))

        Re_t, N_prof = e_n_method(Re_x, ai, N_cr_eff)
        Re_xt[i] = Re_t
        N_profiles.append(N_prof)


    smoothness = np.sum(np.diff(Re_xt) ** 2)

    return {
        'z_stations': z_stations,
        'Re_xt': Re_xt,
        'N_profiles': N_profiles,
        'smoothness': smoothness,
        'mean_Re_xt': np.mean(Re_xt),
        'std_Re_xt': np.std(Re_xt, ddof=1)
    }


def receptivity_coefficient(Ma, Tw_Te, Tu, F=2.5e-6):
    C_rec = Tu * (Ma ** 2) * (Tw_Te ** (-0.5)) * np.exp(-F ** 2 / 1e-11)
    return max(C_rec, 1e-10)
