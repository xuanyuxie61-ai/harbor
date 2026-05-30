# -*- coding: utf-8 -*-

import numpy as np
from special_functions import lambert_w


def lcg_park_miller(seed):
    modulus = 2147483647
    multiplier = 16807
    seed = int(seed) % modulus
    if seed == 0:
        seed = 1
    k = seed // 127773
    seed = multiplier * (seed - k * 127773) - k * 2836
    if seed < 0:
        seed += modulus
    uniform_val = seed / modulus
    return seed, uniform_val


def classical_nucleation_rate(supersaturation, temperature,
                               gamma0=0.025, vm=1e-28, kb=1.380649e-23,
                               A_prefactor=1e20, k_gamma=0.001, T_ref=298.15):







    raise NotImplementedError("Hole 1: classical_nucleation_rate is not implemented.")


def secondary_nucleation_rate(supersaturation, magma_density,
                               kb_sec=1e8, b_exp=2.0, j_exp=1.0):
    sigma = np.asarray(supersaturation, dtype=float)
    MT = np.asarray(magma_density, dtype=float)

    sigma = np.where(sigma < 0, 0.0, sigma)
    MT = np.where(MT < 0, 0.0, MT)

    B = kb_sec * (sigma ** b_exp) * (MT ** j_exp)
    return B


def total_nucleation_rate(sigma, T, MT, A_prefactor=1e20, kb_sec=1e8,
                          b_exp=2.0, j_exp=1.0, gamma0=0.025):
    B_prim = classical_nucleation_rate(sigma, T, gamma0=gamma0,
                                        A_prefactor=A_prefactor)
    B_sec = secondary_nucleation_rate(sigma, MT, kb_sec, b_exp, j_exp)
    return B_prim + B_sec


def critical_nucleus_radius(sigma, T, gamma0=0.025, vm=1e-28,
                            kb=1.380649e-23, k_gamma=0.001, T_ref=298.15):
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    T = np.where(T <= 0, 1e-6, T)
    S = 1.0 + sigma
    S = np.where(S <= 1.0, 1.0 + 1e-10, S)
    lnS = np.log(S)
    lnS = np.where(np.abs(lnS) < 1e-10, 1e-10, lnS)

    gamma = gamma0 * (1.0 - k_gamma * (T - T_ref))
    gamma = np.where(gamma <= 0, 1e-6, gamma)

    r_star = 2.0 * gamma * vm / (kb * T * lnS)
    return r_star


def stochastic_nucleation_events(sigma, T, dt, volume, seed=12345,
                                  A_prefactor=1e20, gamma0=0.025):
    B = classical_nucleation_rate(sigma, T, gamma0=gamma0,
                                   A_prefactor=A_prefactor)
    N_expected = float(B) * volume * dt

    if N_expected < 1e-12:
        return 0, seed

    new_seed = seed
    if N_expected < 10.0:

        new_seed, u = lcg_park_miller(new_seed)
        prob = 1.0 - np.exp(-N_expected)
        n_events = 1 if u < prob else 0
    else:

        new_seed, u1 = lcg_park_miller(new_seed)
        new_seed, u2 = lcg_park_miller(new_seed)

        z = np.sqrt(-2.0 * np.log(max(u1, 1e-300))) * np.cos(2.0 * np.pi * u2)
        n_events = int(round(N_expected + np.sqrt(N_expected) * z))
        n_events = max(0, n_events)

    return n_events, new_seed


def analytical_size_dependent_growth_law(t, alpha, beta, k_g, sigma):
    t = np.asarray(t, dtype=float)
    G0 = k_g * (sigma ** 2)

    if np.abs(beta + 1.0) < 1e-10:


        val = 1.0 + 2.0 * alpha * G0 * t
        val = np.where(val < 0, 0.0, val)
        L = (-1.0 + np.sqrt(val)) / alpha
        L = np.where(alpha == 0, G0 * t, L)
        return L
    elif np.abs(beta - 1.0) < 1e-10:


        if np.abs(alpha) < 1e-12:
            return G0 * t
        L = (np.exp(alpha * G0 * t) - 1.0) / alpha
        return L
    else:


        L0 = 1e-9
        arg = alpha * L0 * np.exp(alpha * L0 + G0 * t)
        arg = np.where(arg < -1.0 / np.e, -1.0 / np.e, arg)
        w = lambert_w(arg, branch=0)
        L = w / alpha
        L = np.where(alpha == 0, L0 * np.exp(G0 * t), L)
        return L
