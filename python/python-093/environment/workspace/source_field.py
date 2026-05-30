#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from special_functions import sincn_fun






def prime_list(n):
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229
    ]
    return primes[:n]


def radical_inverse(i, base):
    i = int(i)
    base = int(base)
    result = 0.0
    f = 1.0 / base
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def hammersley_sequence(i1, i2, m, n=None):
    if n is None:
        n = i2
    primes = prime_list(m)
    seq = np.zeros((i2 - i1, m), dtype=np.float64)
    for idx, i in enumerate(range(i1, i2)):
        if m >= 1:
            seq[idx, 0] = i / n
        for j in range(1, m):
            seq[idx, j] = radical_inverse(i, primes[j - 1])
    return seq






def disk_uniform_sample(n, radius=1.0, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    g = rng.standard_normal((n, 2))
    norms = np.linalg.norm(g, axis=1)
    norms = np.maximum(norms, 1e-15)
    g = g / norms[:, None]
    u = rng.random(n)
    g *= (radius * np.sqrt(u))[:, None]
    return g


def disk_gaussian_sample(n, radius=1.0, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    samples = []
    while len(samples) < n:
        u1, u2 = rng.random(2)
        r = radius * np.sqrt(-2.0 * np.log(u1 + 1e-15))
        theta = 2.0 * np.pi * u2
        if r <= radius:
            samples.append([r * np.cos(theta), r * np.sin(theta)])
    return np.asarray(samples[:n], dtype=np.float64)






def gaussian_starter(z, z_s, w0, k0, R_c=np.inf):
    z = np.asarray(z, dtype=np.float64)
    dz = z - z_s




    raise NotImplementedError("HOLE 1: Gaussian starter formula missing")
    return amplitude * np.exp(1j * phase)


def green_starter(z, z_s, k0):
    z = np.asarray(z, dtype=np.float64)
    rho = np.abs(z - z_s)
    rho = np.maximum(rho, 1e-6)

    amp = np.sqrt(2.0 / (np.pi * k0 * rho))
    phase = k0 * rho - np.pi / 4.0
    return amp * np.exp(1j * phase)


def directional_factor(theta, ka):
    theta = np.asarray(theta, dtype=np.float64)
    x = ka * np.sin(theta)
    x = np.maximum(np.abs(x), 1e-15)


    j1 = _bessel_j1_series(x)
    return 2.0 * j1 / x


def _bessel_j1_series(x, max_terms=30):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    for m in range(max_terms):
        sign = (-1) ** m
        num = (x / 2.0) ** (2 * m + 1)
        den = np.exp(gammaln_approx(m + 1) + gammaln_approx(m + 2))
        term = sign * num / den
        result += term
        if np.all(np.abs(term) < 1e-15):
            break
    return result


def gammaln_approx(n):
    if n <= 1:
        return 0.0
    return (n - 0.5) * np.log(n) - n + 0.5 * np.log(2.0 * np.pi)


def sinc_interpolate(z_query, z_grid, u_grid):
    z_query = np.asarray(z_query, dtype=np.float64)
    z_grid = np.asarray(z_grid, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.complex128)
    dz = z_grid[1] - z_grid[0]
    result = np.zeros_like(z_query, dtype=np.complex128)
    for m, z_m in enumerate(z_grid):
        result += u_grid[m] * sincn_fun((z_query - z_m) / dz)
    return result


def build_initial_field(z_grid, z_s, source_type='gaussian', **kwargs):
    k0 = kwargs.get('k0', 2.0 * np.pi * 100.0 / 1500.0)
    if source_type == 'gaussian':
        w0 = kwargs.get('w0', 5.0)
        R_c = kwargs.get('R_c', np.inf)
        return gaussian_starter(z_grid, z_s, w0, k0, R_c)
    elif source_type == 'green':
        return green_starter(z_grid, z_s, k0)
    elif source_type == 'directional':

        w0 = kwargs.get('w0', 5.0)
        ka = kwargs.get('ka', 10.0)
        u = gaussian_starter(z_grid, z_s, w0, k0)

        theta = np.arctan2(z_grid - z_s, 1.0)
        u *= directional_factor(theta, ka)
        return u
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def source_power_normalization(u, z_grid):
    z_grid = np.asarray(z_grid, dtype=np.float64)
    u = np.asarray(u, dtype=np.complex128)
    intensity = np.abs(u) ** 2
    power = np.trapezoid(intensity, z_grid)
    if power > 1e-15:
        return u / np.sqrt(power)
    return u
