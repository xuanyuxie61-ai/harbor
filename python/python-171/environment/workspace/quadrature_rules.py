# -*- coding: utf-8 -*-

import numpy as np
import math
from orthogonal_polynomials import gauss_laguerre_rule, gauss_hermite_rule






def spherical_to_cartesian(theta, phi):
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def lebedev_rule_6():
    nodes = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0]
    ], dtype=float)
    w = np.full(6, 4.0 * math.pi / 6.0, dtype=float)
    return nodes, w


def lebedev_rule_14():
    nodes_axis = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    w_axis = np.full(6, 0.6666666666666667 * 4.0 * math.pi / 14.0, dtype=float)

    s = 1.0 / math.sqrt(3.0)
    nodes_diag = np.array([
        [ s,  s,  s], [ s,  s, -s], [ s, -s,  s], [ s, -s, -s],
        [-s,  s,  s], [-s,  s, -s], [-s, -s,  s], [-s, -s, -s]
    ], dtype=float)
    w_diag = np.full(8, 0.75 * 4.0 * math.pi / 14.0, dtype=float)

    nodes = np.vstack([nodes_axis, nodes_diag])
    w = np.concatenate([w_axis, w_diag])

    w = w / np.sum(w) * 4.0 * math.pi
    return nodes, w


def integrate_on_sphere(f_eval, rule='14'):
    if rule == '6':
        nodes, w = lebedev_rule_6()
    elif rule == '14':
        nodes, w = lebedev_rule_14()
    else:
        nodes, w = lebedev_rule_14()
    vals = f_eval(nodes)
    return float(np.dot(w, vals))






def genz_cosine(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = 2.0 * math.pi * w[0] + np.dot(c, x)
    return np.cos(val)


def genz_product_peak(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.ones(x.shape[1], dtype=float)
    for i in range(m):
        val *= 1.0 / (c[i] ** (-2) + (x[i, :] - w[i]) ** 2)
    return val


def genz_corner_peak(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.ones(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * x[i, :]
    return val ** (-(m + 1))


def genz_gaussian(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] ** 2 * (x[i, :] - w[i]) ** 2
    return np.exp(-val)


def genz_c0_function(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * np.abs(x[i, :] - w[i])
    return np.exp(-val)


def genz_discontinuous(m, c, w, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * x[i, :]
    mask = (x[0, :] <= w[0])
    if m > 1:
        mask &= (x[1, :] <= w[1])
    return np.exp(val) * mask.astype(float)



def genz_evaluate(prob, m, c, w, x):
    if prob == 1:
        return genz_cosine(m, c, w, x)
    elif prob == 2:
        return genz_product_peak(m, c, w, x)
    elif prob == 3:
        return genz_corner_peak(m, c, w, x)
    elif prob == 4:
        return genz_gaussian(m, c, w, x)
    elif prob == 5:
        return genz_c0_function(m, c, w, x)
    elif prob == 6:
        return genz_discontinuous(m, c, w, x)
    else:
        raise ValueError("prob must be in 1..6")


def genz_integral_exact(prob, m, c, w):
    if prob == 1:


        from scipy.integrate import nquad
        def f(*x):
            return float(genz_cosine(m, c, w, np.array(x)))
        ranges = [(0.0, 1.0)] * m
        val, _ = nquad(f, ranges)
        return val
    elif prob == 4:

        val = 1.0
        for i in range(m):
            from math import erf, sqrt
            val *= (sqrt(math.pi) / (2.0 * c[i])) * (erf(c[i] * (1.0 - w[i])) + erf(c[i] * w[i]))
        return val
    else:

        rng = np.random.default_rng(42)
        N = 200000
        pts = rng.random((m, N))
        vals = genz_evaluate(prob, m, c, w, pts)
        return float(np.mean(vals))






def tensor_product_quadrature_1d(rule_func, n_points, a, b):
    nodes, weights = rule_func(n_points)










    x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * weights
    return x_mapped, w_mapped


def multidimensional_gauss_legendre_simple(m, n_per_dim, a=0.0, b=1.0):
    N_total = n_per_dim ** m
    if N_total > 1_000_000:
        raise ValueError("Total quadrature points exceed 1e6. Reduce m or n_per_dim.")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n_per_dim)

    nodes_1d = 0.5 * (b - a) * nodes_1d + 0.5 * (a + b)
    weights_1d = 0.5 * (b - a) * weights_1d


    grids = np.meshgrid(*[nodes_1d] * m, indexing='ij')
    nodes = np.vstack([g.ravel() for g in grids])
    weight_grid = np.meshgrid(*[weights_1d] * m, indexing='ij')
    weights = np.prod(np.stack([g.ravel() for g in weight_grid]), axis=0)
    return nodes, weights






def qmc_integral_halton(f, m, n_samples, a=0.0, b=1.0, seed=0):
    from random_tools import halton_sequence
    pts = halton_sequence(0, n_samples - 1, m)

    pts = a + (b - a) * pts
    vals = f(pts)
    volume = (b - a) ** m
    return volume * float(np.mean(vals)), volume * float(np.std(vals) / math.sqrt(n_samples))
