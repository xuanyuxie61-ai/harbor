#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def generate_fractal_flux_tubes(n_points=5000):
    if n_points <= 0:
        raise ValueError("n_points 必须为正整数")
    


    b_vectors = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0/3.0], [0.0, 0.0, 2.0/3.0],
        [0.0, 1.0/3.0, 0.0], [0.0, 1.0/3.0, 2.0/3.0],
        [0.0, 2.0/3.0, 0.0], [0.0, 2.0/3.0, 1.0/3.0], [0.0, 2.0/3.0, 2.0/3.0],
        [1.0/3.0, 0.0, 0.0], [1.0/3.0, 0.0, 2.0/3.0],
        [1.0/3.0, 2.0/3.0, 0.0], [1.0/3.0, 2.0/3.0, 2.0/3.0],
        [2.0/3.0, 0.0, 0.0], [2.0/3.0, 0.0, 1.0/3.0], [2.0/3.0, 0.0, 2.0/3.0],
        [2.0/3.0, 1.0/3.0, 0.0], [2.0/3.0, 1.0/3.0, 2.0/3.0],
        [2.0/3.0, 2.0/3.0, 0.0], [2.0/3.0, 2.0/3.0, 1.0/3.0], [2.0/3.0, 2.0/3.0, 2.0/3.0]
    ])
    

    scale = 1.0 / 3.0
    

    points = np.zeros((n_points, 3))
    x = np.random.rand(3)
    

    burn_in = min(100, n_points // 10)
    for _ in range(burn_in):
        j = np.random.randint(0, 20)
        x = scale * x + b_vectors[j]
    

    for i in range(n_points):
        j = np.random.randint(0, 20)
        x = scale * x + b_vectors[j]
        points[i] = x.copy()
    
    return points


def compute_fractal_dimension(points, r_min=0.01, r_max=0.5, n_r=20):
    N = points.shape[0]
    if N < 100:
        return 0.0
    

    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    extent = np.max(pmax - pmin)
    if extent < 1e-20:
        return 0.0
    
    p_norm = (points - pmin) / extent
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    counts = np.zeros(n_r)
    
    for i, r in enumerate(radii):
        n_boxes = max(1, int(np.ceil(1.0 / r)))

        idx = np.floor(p_norm * n_boxes).astype(int)
        idx = np.clip(idx, 0, n_boxes - 1)

        unique = set(map(tuple, idx))
        counts[i] = len(unique)
    

    valid = counts > 0
    if np.sum(valid) < 3:
        return 0.0
    
    log_inv_r = np.log(1.0 / radii[valid])
    log_N = np.log(counts[valid])
    
    D_box = np.polyfit(log_inv_r, log_N, 1)[0]
    

    D_box = min(max(D_box, 0.0), 3.0)
    
    return D_box


def map_fractal_to_magnetic_field(points, B0, fractal_scale=1e4):
    N = points.shape[0]
    

    centers = points * fractal_scale
    

    wavelengths = fractal_scale / (3.0 ** np.arange(1, 5))
    
    def B_field(x):
        x = np.asarray(x, dtype=float)
        B = np.array([0.0, 0.0, B0])
        

        for j, lam in enumerate(wavelengths):
            k = 2.0 * np.pi / lam
            amplitude = 0.05 * B0 / (j + 1.0)
            

            if np.abs(k * x[2]) > 100:
                continue
                
            B[0] += amplitude * np.sin(k * x[2])
            B[1] += amplitude * np.cos(k * x[2])
        
        return B
    
    return B_field
