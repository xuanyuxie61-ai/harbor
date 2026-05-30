#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def barycentric_weights(x_nodes):
    n = len(x_nodes)
    w = np.ones(n)
    
    for j in range(n):
        for m in range(n):
            if m != j:
                diff = x_nodes[j] - x_nodes[m]

                if np.abs(diff) < 1e-30:
                    diff = 1e-30 * np.sign(diff) if diff != 0 else 1e-30
                w[j] /= diff
    
    return w


def barycentric_interpolate(x_nodes, f_values, x_eval, w=None):
    x_nodes = np.asarray(x_nodes)
    f_values = np.asarray(f_values)
    
    if w is None:
        w = barycentric_weights(x_nodes)
    
    scalar_input = np.isscalar(x_eval)
    x_eval = np.atleast_1d(x_eval)
    
    f_eval = np.zeros_like(x_eval, dtype=float)
    
    for idx, x in enumerate(x_eval):

        exact_match = np.abs(x - x_nodes) < 1e-30 * (np.max(np.abs(x_nodes)) + 1.0)
        if np.any(exact_match):
            f_eval[idx] = f_values[np.argmax(exact_match)]
            continue
        

        numer = 0.0
        denom = 0.0
        for j in range(len(x_nodes)):
            term = w[j] / (x - x_nodes[j])
            numer += term * f_values[j]
            denom += term
        

        if np.abs(denom) < 1e-30:
            denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30
        
        f_eval[idx] = numer / denom
    
    return f_eval[0] if scalar_input else f_eval


def chebyshev_nodes(a, b, n):
    j = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return x


def lagrange_phase_space_reconstruction(v_parallel, v_perp, f_grid, params,
                                         n_cheb=16):
    nv = len(v_parallel)
    

    vpar_min, vpar_max = np.min(v_parallel), np.max(v_parallel)
    vperp_min, vperp_max = np.min(v_perp), np.max(v_perp)
    
    n_cheb_eff = min(n_cheb, nv)
    

    if nv <= n_cheb:

        return f_grid.copy()
    

    cheb_idx_par = np.round(np.linspace(0, nv - 1, n_cheb_eff)).astype(int)
    cheb_idx_perp = np.round(np.linspace(0, nv - 1, n_cheb_eff)).astype(int)
    

    cheb_idx_par = np.clip(cheb_idx_par, 0, nv - 1)
    cheb_idx_perp = np.clip(cheb_idx_perp, 0, nv - 1)
    
    x_nodes = v_parallel[cheb_idx_par]
    y_nodes = v_perp[cheb_idx_perp]
    f_nodes = f_grid[np.ix_(cheb_idx_perp, cheb_idx_par)]
    

    wx = barycentric_weights(x_nodes)
    wy = barycentric_weights(y_nodes)
    

    f_reconstructed = np.zeros((nv, nv))
    
    for j in range(nv):
        for i in range(nv):
            x = v_parallel[i]
            y = v_perp[j]
            


            val = 0.0
            for m in range(n_cheb_eff):
                Lx = barycentric_interpolate(x_nodes, np.eye(n_cheb_eff)[m], x, wx)
                for n in range(n_cheb_eff):
                    Ly = barycentric_interpolate(y_nodes, np.eye(n_cheb_eff)[n], y, wy)
                    val += f_nodes[n, m] * Lx * Ly
            
            f_reconstructed[j, i] = val
    

    f_reconstructed = np.maximum(f_reconstructed, 0.0)
    

    f_reconstructed = np.clip(f_reconstructed, 0.0, 10.0 * np.max(f_grid))
    
    return f_reconstructed


def test_interpolation_accuracy(v_parallel, v_perp, f_grid, params):
    f_rec = lagrange_phase_space_reconstruction(v_parallel, v_perp, f_grid, params)
    
    dv_par = v_parallel[1] - v_parallel[0]
    dv_perp = v_perp[1] - v_perp[0]
    
    error = np.sqrt(np.sum((f_rec - f_grid)**2) * dv_par * dv_perp)
    norm = np.sqrt(np.sum(f_grid**2) * dv_par * dv_perp)
    
    if norm > 1e-30:
        rel_error = error / norm
    else:
        rel_error = 0.0
    
    return rel_error
