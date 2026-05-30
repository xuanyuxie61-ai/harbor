# -*- coding: utf-8 -*-

import numpy as np
import math






def line_grid(n, a, b, c=1):
    if n < 1:
        raise ValueError("n must be >= 1.")
    if c < 1 or c > 5:
        raise ValueError("c must be in [1,5].")
    x = np.zeros(n, dtype=float)
    for j in range(1, n + 1):
        if c == 1:
            if n == 1:
                x[j - 1] = 0.5 * (a + b)
            else:
                x[j - 1] = ((n - j) * a + (j - 1) * b) / (n - 1)
        elif c == 2:
            x[j - 1] = ((n - j + 1) * a + j * b) / (n + 1)
        elif c == 3:
            x[j - 1] = ((n - j + 1) * a + (j - 1) * b) / n
        elif c == 4:
            x[j - 1] = ((n - j) * a + j * b) / n
        elif c == 5:
            x[j - 1] = ((2 * n - 2 * j + 1) * a + (2 * j - 1) * b) / (2 * n)
    return x






def cvt_1d_lloyd(n, it_num, s_num, density_func, init_type=1, seed=None):
    rng = np.random.default_rng(seed)

    if init_type == 1:
        g = rng.random(n) * 2.0 - 1.0
        g.sort()
    elif init_type == 2:
        g = np.cos(math.pi * (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n))
    else:
        g = np.array([((n - i) * (-1.0) + i * 1.0) / (n + 1) for i in range(1, n + 1)], dtype=float)


    eps = 1e-12
    s = np.linspace(-1.0 + eps, 1.0 - eps, s_num)
    mu = density_func(s)
    mu = np.clip(mu, 1e-30, 1e30)
    rho = mu ** 3

    energy_history = np.zeros(it_num, dtype=float)
    motion_history = np.zeros(it_num, dtype=float)

    for it in range(it_num):

        gb = np.zeros(n + 1, dtype=float)
        gb[0] = -1.0
        for j in range(1, n):
            gb[j] = 0.5 * (g[j - 1] + g[j])
        gb[n] = 1.0


        g_new = np.zeros(n, dtype=float)
        energy = 0.0


        region_sums = np.zeros(n, dtype=float)
        region_weights = np.zeros(n, dtype=float)

        for k in range(s_num):
            sk = s[k]


            j = np.searchsorted(gb, sk, side='right') - 1
            j = max(0, min(j, n - 1))
            region_sums[j] += rho[k] * sk
            region_weights[j] += rho[k]
            energy += rho[k] * (sk - g[j]) ** 2

        for j in range(n):
            if region_weights[j] > 1e-30:
                g_new[j] = region_sums[j] / region_weights[j]
            else:
                g_new[j] = g[j]


        if n % 2 == 1:
            g_new[n // 2] = 0.0

        energy_history[it] = energy / s_num
        motion_history[it] = np.mean((g_new - g) ** 2)
        g = g_new.copy()

    return g, energy_history, motion_history


def chebyshev_zero_density(s):
    s = np.asarray(s, dtype=float)
    return 1.0 / np.sqrt(np.maximum(1e-30, 1.0 - s ** 2))


def polynomial_density(s, alpha=2.0):
    s = np.asarray(s, dtype=float)
    return (1.0 + np.abs(s)) ** alpha






def mesh_refinement_1d(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return x.copy()
    x_fine = np.zeros(2 * n - 1, dtype=float)
    x_fine[0::2] = x
    x_fine[1::2] = 0.5 * (x[:-1] + x[1:])
    return x_fine


def multi_level_grid(n_coarse, levels, a, b, c=1):
    grids = []
    n = n_coarse
    for _ in range(levels):
        grids.append(line_grid(n, a, b, c))
        n = 2 * n - 1
    return grids






def tensor_product_grid_1d(n_each, a, b, c=1):
    d = len(n_each)
    grids = []
    for dim in range(d):
        grids.append(line_grid(n_each[dim], a[dim], b[dim], c))
    return grids
