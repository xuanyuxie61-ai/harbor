# -*- coding: utf-8 -*-

import numpy as np


def dirichlet_sample_uniform_simplex(n_samples, dim, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_samples <= 0 or dim <= 0:
        return np.empty((max(0, n_samples), max(0, dim)), dtype=float)

    E = -np.log(rng.random((n_samples, dim + 1)))
    E = np.where(E == 0, 1e-300, E)
    s = E.sum(axis=1, keepdims=True)
    x = E[:, :-1] / s
    return x


def wedge01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if e.size != 3:
        raise ValueError("e must have exactly 3 elements")
    e1, e2, e3 = e[0], e[1], e[2]
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative")


    value_xy = 1.0
    for i in range(1, e2 + 1):
        value_xy *= float(i) / float(e1 + i)
    value_xy /= float((e1 + e2 + 1) * (e1 + e2 + 2))


    if e3 % 2 == 1:
        value_z = 0.0
    else:
        value_z = 2.0 / float(e3 + 1)

    return value_xy * value_z


def tetrahedron01_monomial_integral(e):
    e = np.asarray(e, dtype=int)
    if e.size != 3:
        raise ValueError("e must have exactly 3 elements")
    if np.any(e < 0):
        raise ValueError("Exponents must be non-negative")

    e1, e2, e3 = e[0], e[1], e[2]
    k = e1 + e2 + e3 + 3
    value = 1.0

    for j in range(1, e1 + 1):
        value *= float(j) / float(k)
        k -= 1

    for j in range(1, e2 + 1):
        value *= float(j) / float(k)
        k -= 1

    for j in range(1, e3 + 1):
        value *= float(j) / float(k)
        k -= 1

    for _ in range(k):
        value /= float(k)
        k -= 1
    return value


def tetrahedron01_volume():
    return 1.0 / 6.0


def monomial_value(m, n, e, x):
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    if x.shape[0] != m or x.shape[1] != n:
        raise ValueError(f"x shape must be ({m}, {n}), got {x.shape}")

    v = np.ones(n, dtype=float)
    for i in range(m):
        if e[i] == 0:
            continue
        xi = x[i, :]


        xi = np.where(np.abs(xi) < 1e-300, 1e-300 * np.sign(xi) if np.any(xi < 0) else 1e-300, xi)
        v *= xi ** e[i]
    return v


def composition_space_integral(func, dim, n_samples=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = dirichlet_sample_uniform_simplex(n_samples, dim, rng)
    values = func(samples)
    from math import factorial
    vol = 1.0 / factorial(dim)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    integral = vol * mean_val
    std_err = vol * std_val / np.sqrt(n_samples)
    return integral, std_err
