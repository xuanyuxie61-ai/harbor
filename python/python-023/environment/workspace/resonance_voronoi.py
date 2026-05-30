#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def detect_resonant_particles(v_grid, omega_solutions, params,
                               resonance_width=0.15, p_norm=2):
    N = v_grid.shape[0]
    Omega_e = params['Omega_e']
    v_te = params['v_te']
    c = params['c']
    
    resonant_indices = []
    
    for i in range(N):
        v = v_grid[i]
        v_parallel = v[2]
        v_perp = np.sqrt(v[0]**2 + v[1]**2)
        

        v_sq = np.sum(v**2)
        gamma = 1.0 / np.sqrt(max(1.0 - v_sq / c**2, 1e-10))
        gamma = min(gamma, 100.0)
        
        is_resonant = False
        

        for k_omega in omega_solutions:
            k = k_omega[0]
            omega = complex(k_omega[1])
            omega_r = omega.real
            
            if omega_r <= 0 or np.abs(k) < 1e-20:
                continue
            

            v_res = (omega_r + Omega_e / gamma) / k
            

            delta_v = np.abs(v_parallel - v_res)
            

            if p_norm == 2:
                dist = delta_v
            elif p_norm == 1:
                dist = delta_v + 0.1 * v_perp
            elif p_norm == np.inf:
                dist = max(delta_v, 0.1 * v_perp)
            else:
                dist = (delta_v**p_norm + (0.1 * v_perp)**p_norm)**(1.0 / p_norm)
            
            if dist < resonance_width * v_te:
                is_resonant = True
                break
        
        if is_resonant:
            resonant_indices.append(i)
    
    return resonant_indices


def voronoi_nearest_neighbor(query_points, centers, p_norm=2):
    Nq = query_points.shape[0]
    Nc = centers.shape[0]
    
    nearest_idx = np.zeros(Nq, dtype=int)
    min_dist = np.zeros(Nq)
    
    for i in range(Nq):
        q = query_points[i]
        

        diff = centers - q
        
        if p_norm == 2:
            dists = np.sum(diff**2, axis=1)
        elif p_norm == 1:
            dists = np.sum(np.abs(diff), axis=1)
        elif p_norm == np.inf:
            dists = np.max(np.abs(diff), axis=1)
        else:
            dists = np.sum(np.abs(diff)**p_norm, axis=1)**(1.0 / p_norm)
        
        nearest_idx[i] = np.argmin(dists)
        min_dist[i] = dists[nearest_idx[i]]
    
    return nearest_idx, min_dist


def compute_resonance_region_volume(v_parallel, v_perp, omega_solutions, params):
    n_samples = 2000
    v_max = 3.0 * params['v_te']
    
    samples = np.random.uniform(-v_max, v_max, (n_samples, 3))
    samples[:, 0] = np.abs(samples[:, 0])
    
    resonant = detect_resonant_particles(samples, omega_solutions, params)
    
    fraction = len(resonant) / n_samples
    total_volume = (2 * v_max) * (2 * v_max) * v_max
    
    return fraction * total_volume
