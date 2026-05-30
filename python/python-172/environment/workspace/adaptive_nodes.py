# -*- coding: utf-8 -*-

import numpy as np


def density_function(x, alpha=2.0, center=0.0):
    return np.exp(-alpha * (x - center) ** 2) + 0.01


def cvt_1d_lloyd(n_generators, n_samples, it_num, domain=(-1.0, 1.0),
                 rho_func=None, seed=42):
    rng = np.random.default_rng(seed)
    xmin, xmax = domain

    z = np.linspace(xmin, xmax, n_generators)

    if rho_func is None:
        def rho_func(x):
            return np.ones_like(x)

    for it in range(it_num):

        samples = rng.uniform(xmin, xmax, n_samples)
        weights = rho_func(samples)



        z_sorted = np.sort(z)
        boundaries = np.zeros(n_generators + 1)
        boundaries[0] = xmin - 1e-10
        boundaries[-1] = xmax + 1e-10
        for i in range(1, n_generators):
            boundaries[i] = 0.5 * (z_sorted[i - 1] + z_sorted[i])


        z_new = np.zeros(n_generators)
        for i in range(n_generators):
            mask = (samples >= boundaries[i]) & (samples < boundaries[i + 1])
            if np.any(mask):
                z_new[i] = np.average(samples[mask], weights=weights[mask])
            else:
                z_new[i] = z_sorted[i]
        z = z_new

    return np.sort(z)


def map_cvt_to_cgl(cvt_nodes, n_cgl):
    cvt_nodes = np.asarray(cvt_nodes)
    n = len(cvt_nodes)

    j = np.arange(n_cgl)
    cgl = np.cos(np.pi * j / (n_cgl - 1))


    cvt_sorted = np.sort(cvt_nodes)
    cvt_cdf = np.arange(1, n + 1) / n


    warped = np.interp((j + 0.5) / n_cgl, cvt_cdf, cvt_sorted)

    warped[0] = cvt_sorted[-1]
    warped[-1] = cvt_sorted[0]


    mapping_derivative = np.gradient(warped, cgl)
    mapping_derivative = np.clip(np.abs(mapping_derivative), 0.1, 10.0)

    return warped, mapping_derivative


def quadratic_superconvergent_point(x_vals, y_vals):
    x_vals = np.asarray(x_vals, dtype=np.float64)
    y_vals = np.asarray(y_vals, dtype=np.float64)
    if len(x_vals) != 3 or len(y_vals) != 3:
        raise ValueError("Exactly three points required.")

    V = np.vander(x_vals, 3)
    try:
        coeffs = np.linalg.solve(V, y_vals)
    except np.linalg.LinAlgError:

        return np.mean(x_vals), np.mean(y_vals)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    if abs(a) < 1e-12:
        return np.mean(x_vals), np.mean(y_vals)
    x_star = -b / (2.0 * a)
    y_star = a * x_star ** 2 + b * x_star + c
    return x_star, y_star


def extract_superconvergent_points(x, u):
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    sc_x = []
    sc_u = []
    for i in range(len(x) - 2):
        xs, ys = quadratic_superconvergent_point(x[i:i + 3], u[i:i + 3])
        if x[i + 2] <= xs <= x[i]:
            sc_x.append(xs)
            sc_u.append(ys)
    return np.array(sc_x), np.array(sc_u)
