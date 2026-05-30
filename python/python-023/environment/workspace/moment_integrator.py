#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def integrate_2d_velocity_space(v_parallel, v_perp, integrand):
    nv_par = len(v_parallel)
    nv_perp = len(v_perp)
    
    dv_par = v_parallel[1] - v_parallel[0] if nv_par > 1 else 1.0
    dv_perp = v_perp[1] - v_perp[0] if nv_perp > 1 else 1.0
    

    w_par = np.ones(nv_par)
    w_par[0] = 0.5
    w_par[-1] = 0.5
    if nv_par > 2:
        w_par[1:-1:2] = 2.0
        w_par[2:-1:2] = 2.0

        w_par = np.ones(nv_par)
        w_par[0] = 1.0 / 3.0
        w_par[-1] = 1.0 / 3.0
        if nv_par % 2 == 0:

            w_par[-2] = 4.0 / 3.0
        else:
            for i in range(1, nv_par - 1):
                if i % 2 == 1:
                    w_par[i] = 4.0 / 3.0
                else:
                    w_par[i] = 2.0 / 3.0
    
    w_perp = np.ones(nv_perp)
    w_perp[0] = 1.0 / 3.0
    w_perp[-1] = 1.0 / 3.0
    if nv_perp > 2:
        for i in range(1, nv_perp - 1):
            if i % 2 == 1:
                w_perp[i] = 4.0 / 3.0
            else:
                w_perp[i] = 2.0 / 3.0
    

    if nv_par % 2 == 0 or nv_par < 3:
        w_par = np.ones(nv_par)
        w_par[0] = 0.5
        w_par[-1] = 0.5
    
    if nv_perp % 2 == 0 or nv_perp < 3:
        w_perp = np.ones(nv_perp)
        w_perp[0] = 0.5
        w_perp[-1] = 0.5
    
    result = 0.0
    for j in range(nv_perp):
        for i in range(nv_par):
            result += w_perp[j] * w_par[i] * integrand[j, i] * dv_perp * dv_par
    

    result *= 2.0 * np.pi
    
    return result


def compute_velocity_space_moments(v_parallel, v_perp, f_grid, params):
    m_e = params['m_e']
    q_e = params['q_e']
    n0 = params['n0']
    v_te = params['v_te']
    
    nv_par = len(v_parallel)
    nv_perp = len(v_perp)
    
    VP, VPL = np.meshgrid(v_perp, v_parallel, indexing='ij')
    

    f_grid = np.maximum(f_grid, 0.0)
    

    v_sq = VPL**2 + VP**2
    f_maxwell = (1.0 / (np.pi * v_te**2))**(1.5) * np.exp(-v_sq / v_te**2)
    n_maxwell = integrate_2d_velocity_space(v_parallel, v_perp, f_maxwell * VP)
    

    f_peak = np.max(f_grid)
    fM_peak = np.max(f_maxwell)
    if f_peak > 1e-30 and fM_peak > 1e-30:
        f_grid = f_grid * (fM_peak / f_peak)
    

    integrand_n = f_grid * VP
    n_density = integrate_2d_velocity_space(v_parallel, v_perp, integrand_n)
    
    density_perturbation = (n_density - n_maxwell) / (n_maxwell + 1e-30)
    

    integrand_gamma = f_grid * VP * VPL
    gamma_parallel = integrate_2d_velocity_space(v_parallel, v_perp, integrand_gamma)
    u_parallel = gamma_parallel / (n_density + 1e-30)
    

    integrand_ppar = f_grid * VP * (VPL - u_parallel)**2
    p_parallel = m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_ppar)
    

    integrand_pperp = f_grid * VP * VP**2 / 2.0
    p_perp = m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_pperp)
    

    T_parallel = p_parallel / (n_density * q_e + 1e-30)
    T_perp = p_perp / (n_density * q_e + 1e-30)
    

    T_parallel = max(T_parallel, 0.01)
    T_perp = max(T_perp, 0.01)
    

    anisotropy = T_perp / T_parallel - 1.0
    

    integrand_q = f_grid * VP * (VPL - u_parallel)**3
    Q_parallel = 0.5 * m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_q)
    

    f_pos = np.maximum(f_grid, 1e-30)
    integrand_s = -f_pos * np.log(f_pos) * VP
    entropy = integrate_2d_velocity_space(v_parallel, v_perp, integrand_s)
    
    moments = {
        'density': n_density,
        'density_perturbation': density_perturbation,
        'parallel_flow': u_parallel,
        'p_parallel': p_parallel,
        'p_perp': p_perp,
        'T_parallel': T_parallel,
        'T_perp': T_perp,
        'anisotropy': anisotropy,
        'Q_parallel': Q_parallel,
        'entropy': entropy
    }
    
    return moments
