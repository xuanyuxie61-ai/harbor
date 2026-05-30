#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import eval_legendre


def legendre_polynomial_normalized(n, x):

    x = np.clip(x, -1.0, 1.0)
    Pn = eval_legendre(n, x)
    norm = np.sqrt((2.0 * n + 1.0) / 2.0)
    return norm * Pn


def multivariate_legendre_basis(alpha, xi):
    N = len(alpha)
    result = 1.0
    for i in range(N):
        result *= legendre_polynomial_normalized(alpha[i], xi[i])
    return result


def enumerate_multi_indices(N, P):
    indices = []
    
    def recurse(current, remaining, dim):
        if dim == N - 1:
            current.append(remaining)
            indices.append(np.array(current, dtype=int))
            current.pop()
            return
        for val in range(remaining + 1):
            current.append(val)
            recurse(current, remaining - val, dim + 1)
            current.pop()
    
    recurse([], P, 0)
    return indices


def compute_mass_matrix(indices):
    M = len(indices)
    C = np.eye(M)
    return C


def polychaos_magnetic_uncertainty(v_parallel, v_perp, params, n_stochastic=2, p_degree=3):
    nv = len(v_parallel)
    

    indices = enumerate_multi_indices(n_stochastic, p_degree)
    M = len(indices)
    

    C = compute_mass_matrix(indices)
    


    f_coeffs = np.zeros((M, nv, nv))
    

    v_te = params['v_te']
    VP, VPL = np.meshgrid(v_perp, v_parallel, indexing='ij')
    v_sq = VPL**2 + VP**2
    

    f_maxwell = (1.0 / (np.pi * v_te**2))**(1.5) * np.exp(-v_sq / v_te**2)
    f_coeffs[0] = f_maxwell
    


    n_samples = 500
    for samp in range(n_samples):

        xi = 2.0 * np.random.rand(n_stochastic) - 1.0
        

        delta_B = 0.1 * params['B0'] * (1.0 + 0.3 * np.sum(xi) / n_stochastic)
        


        Omega_e = params['Omega_e']
        response = 1.0 + 0.1 * delta_B / params['B0'] * np.sin(2 * np.pi * VPL * Omega_e / v_te)
        response = np.clip(response, 0.5, 2.0)
        
        f_sample = f_maxwell * response
        

        for alpha_idx, alpha in enumerate(indices):
            psi_val = multivariate_legendre_basis(alpha, xi)
            f_coeffs[alpha_idx] += f_sample * psi_val / n_samples
    

    f_mean = f_coeffs[0].copy()
    

    f_var = np.zeros((nv, nv))
    for alpha_idx in range(1, M):

        norm_sq = 1.0
        for ai in indices[alpha_idx]:
            norm_sq *= 1.0
        f_var += f_coeffs[alpha_idx]**2 * norm_sq
    

    f_mean = np.maximum(f_mean, 0.0)
    f_var = np.maximum(f_var, 0.0)
    
    return f_mean, f_var
