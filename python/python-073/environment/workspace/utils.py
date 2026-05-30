# -*- coding: utf-8 -*-

import numpy as np
from math import gcd
from functools import reduce


def prime_factors(n):
    if not isinstance(n, int):
        raise TypeError("prime_factors: 输入必须为整数")
    if n < 1:
        raise ValueError("prime_factors: 输入整数必须 >= 1")

    factors = []
    i = 2
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def optimal_chebyshev_order(target_n, max_prime=5):
    N = target_n
    while True:
        factors = prime_factors(N + 1)
        if all(p <= max_prime for p in set(factors)):
            return N
        N += 1


def normalize_array(a, method="minmax"):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    a_min = np.min(a)
    a_max = np.max(a)
    if method == "minmax":
        if abs(a_max - a_min) < 1e-15:
            return np.zeros_like(a)
        return (a - a_min) / (a_max - a_min)
    elif method == "zscore":
        mu = np.mean(a)
        sigma = np.std(a)
        if sigma < 1e-15:
            return np.zeros_like(a)
        return (a - mu) / sigma
    else:
        raise ValueError(f"未知归一化方法: {method}")


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a, b = np.broadcast_arrays(a, b)
    result = np.empty_like(a, dtype=np.float64)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def blasius_function(eta, n_terms=50):
    eta = np.asarray(eta, dtype=np.float64)
    f = np.zeros_like(eta)
    fp = np.zeros_like(eta)
    fpp = np.zeros_like(eta)


    alpha = 0.332057336215196



    mask_small = eta < 5.0
    e = eta[mask_small]
    f[mask_small] = (alpha / 2.0) * e**2 - (alpha**2 / 240.0) * e**5 \
                    + (11.0 * alpha**3 / 161280.0) * e**8
    fp[mask_small] = alpha * e - (alpha**2 / 48.0) * e**4 \
                     + (11.0 * alpha**3 / 20160.0) * e**7
    fpp[mask_small] = alpha - (alpha**2 / 12.0) * e**3 \
                      + (11.0 * alpha**3 / 2880.0) * e**6


    mask_large = ~mask_small
    e = eta[mask_large]
    beta = 1.7207876575205
    f[mask_large] = e - beta
    fp[mask_large] = 1.0 - np.exp(-beta * (e - beta))
    fpp[mask_large] = beta * np.exp(-beta * (e - beta))

    return f, fp, fpp


def compressible_blasius_velocity(eta, Ma, gamma=1.4, Pr=0.72, Tw_Te=1.0):
    _, fp, _ = blasius_function(eta)
    u = np.clip(fp, 0.0, 1.0)


    r = Pr ** (1.0 / 3.0)
    Tr_Te = 1.0 + r * (gamma - 1.0) / 2.0 * Ma**2
    T = Tw_Te + (Tr_Te - Tw_Te) * u - (gamma - 1.0) / 2.0 * Ma**2 * u**2
    T = np.clip(T, 0.1, None)


    mu = sutherland_viscosity(T)
    return u, T, mu


def sutherland_viscosity(T, T_ref=300.0, mu_ref=1.7894e-5, S=110.4):
    T = np.asarray(T, dtype=np.float64)
    ratio = T / T_ref
    mu = mu_ref * ratio**1.5 * (T_ref + S) / np.maximum(T + S, 1e-10)
    return mu


def chebyshev_nodes(n, a=-1.0, b=1.0):
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def chebyshev_diff_matrix(n, a=-1.0, b=1.0):




    raise NotImplementedError("chebyshev_diff_matrix: 请根据 Chebyshev 谱微分矩阵的经典公式完成实现")
