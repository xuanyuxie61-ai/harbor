# -*- coding: utf-8 -*-

import numpy as np


def power_law_growth(sigma, T, k_g0, E_g, g_exp, R=8.314):
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    G = k_g0 * np.exp(-E_g / (R * T)) * (sigma ** g_exp)
    return G


def size_dependent_growth(L, sigma, T, k_g0, E_g, g_exp, alpha, beta, R=8.314):
    L = np.asarray(L, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)
    L = np.where(L < 0, 0.0, L)

    base = k_g0 * np.exp(-E_g / (R * T)) * (sigma ** g_exp)
    size_factor = (1.0 + alpha * L) ** beta
    G = base * size_factor
    return G


def two_step_growth(sigma, T, k_d, k_r0, E_r, g_r, R=8.314):
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    k_r = k_r0 * np.exp(-E_r / (R * T))
    term = k_r * (sigma ** g_r)
    denom = k_d + term
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    G = k_d * term / denom
    return G


def bcf_spiral_growth(sigma, T, A_bcf, B_bcf, E_act, R=8.314):
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    prefactor = A_bcf * np.exp(-E_act / (R * T))

    arg = B_bcf / np.where(sigma < 1e-10, 1e-10, sigma)

    tanh_val = np.tanh(arg)
    G = prefactor * (sigma ** 2) * tanh_val
    return G


def growth_rate_dispersion(G_mean, cv=0.1, n_samples=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if G_mean <= 0:
        return np.zeros(n_samples)
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln = np.log(G_mean) - 0.5 * sigma_ln ** 2
    G_samples = rng.lognormal(mu_ln, sigma_ln, n_samples)
    return G_samples
