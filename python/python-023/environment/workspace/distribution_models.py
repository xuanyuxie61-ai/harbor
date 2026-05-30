#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, gammainc


def kappa_distribution_3d(vx, vy, vz, n0, v_th, kappa):

    kappa = max(kappa, 1.6)
    
    v_sq = vx**2 + vy**2 + vz**2
    

    norm = n0 / ((np.pi * kappa * v_th**2)**(1.5))
    norm *= gamma(kappa + 1.0) / (gamma(kappa - 0.5) * kappa**1.5)
    

    f = norm * (1.0 + v_sq / (kappa * v_th**2))**(-(kappa + 1.0))
    
    return f


def incomplete_beta_noncentral(x, a, b, lam, error_max=1e-10):
    x = np.clip(x, 0.0, 1.0)
    a = max(a, 1e-10)
    b = max(b, 1e-10)
    

    pi_val = np.exp(-lam / 2.0)
    

    beta_log = np.log(gamma(a)) + np.log(gamma(b)) - np.log(gamma(a + b))
    

    if lam < 1e-6:
        from scipy.special import betainc
        return betainc(a, b, x)
    

    p_sum = pi_val
    pb_sum = pi_val * gammainc(a, a, x)
    
    i = 0
    bi = gammainc(a, a, x)
    si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))
    
    while p_sum < 1.0 - error_max and i < 1000:
        i += 1
        pi_val = 0.5 * lam * pi_val / i
        bi = bi - si
        si = x * (a + b + i - 1.0) * si / (a + i)
        
        p_sum += pi_val
        pb_sum += pi_val * bi
    
    return pb_sum


def noncentral_beta_tail(v, v_max, a=2.0, b=5.0, lam=1.5):

    x = np.clip(v / v_max, 0.0, 1.0)
    

    from scipy.special import betainc
    

    f_tail = betainc(a, b, x)
    

    f_tail *= (1.0 + 0.1 * lam * x)
    
    return f_tail


def kappa_nonthermal_distribution(n_particles, v_max, v_te, kappa=4.0, params=None):

    v_grid = np.zeros((n_particles, 3))
    

    for i in range(n_particles):

        u = np.random.rand()

        v_mag = v_te * np.sqrt(kappa) * np.sqrt(u / (1.0 - u + 1e-10))
        v_mag = min(v_mag, v_max)
        

        cos_theta = 2.0 * np.random.rand() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * np.random.rand()
        
        v_grid[i, 0] = v_mag * sin_theta * np.cos(phi)
        v_grid[i, 1] = v_mag * sin_theta * np.sin(phi)
        v_grid[i, 2] = v_mag * cos_theta
    

    f_kappa = kappa_distribution_3d(
        v_grid[:, 0], v_grid[:, 1], v_grid[:, 2],
        n0=1.0, v_th=v_te, kappa=kappa
    )
    

    v_mag = np.linalg.norm(v_grid, axis=1)
    f_tail = noncentral_beta_tail(v_mag, v_max)
    f_tail = f_tail / (np.max(f_tail) + 1e-30) * np.max(f_kappa) * 0.1
    

    w_kappa = 0.9
    w_tail = 0.1
    f_dist = w_kappa * f_kappa + w_tail * f_tail
    

    f_dist = np.maximum(f_dist, 1e-30)
    
    return f_dist, v_grid


def escape_probability(t, tau_esc):
    t = np.maximum(t, 0.0)
    tau_esc = max(tau_esc, 1e-30)
    return 1.0 - np.exp(-t / tau_esc)


def survival_probability(t, tau_esc):
    return 1.0 - escape_probability(t, tau_esc)
