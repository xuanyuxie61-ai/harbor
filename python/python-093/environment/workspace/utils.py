#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from special_functions import gammaln_stable





def pascal_to_i4(k):
    k = int(k)
    if k < 1:
        raise ValueError("Pascal index k must be >= 1")
    d = int((np.sqrt(8 * k - 7) - 1) // 2)
    j = k - d * (d + 1) // 2 - 1
    i = d - j
    return i, j


def i4_to_pascal(i, j):
    d = i + j
    return d * (d + 1) // 2 + j + 1


def triangle_number(d):
    return d * (d + 1) // 2






def binomial_coefficient(n, k):
    if k < 0 or k > n or n < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    return float(np.exp(gammaln_stable(n + 1)
                        - gammaln_stable(k + 1)
                        - gammaln_stable(n - k + 1)))


def trinomial_coefficient(i, j, k):
    n = i + j + k
    if n < 0 or i < 0 or j < 0 or k < 0:
        return 0.0
    return float(np.exp(gammaln_stable(n + 1)
                        - gammaln_stable(i + 1)
                        - gammaln_stable(j + 1)
                        - gammaln_stable(k + 1)))






def chebyshev_to_monomial_matrix(n):
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 1.0
    for k in range(2, n + 1):


        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * M[j - 1, k - 1]
            val -= M[j, k - 2]
            M[j, k] = val
    return M


def monomial_to_chebyshev_matrix(n):
    M = chebyshev_to_monomial_matrix(n)



    return np.linalg.inv(M)


def legendre_to_monomial_matrix(n):
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += (2 * k - 1) * M[j - 1, k - 1]
            val -= (k - 1) * M[j, k - 2]
            M[j, k] = val / k
    return M


def monomial_to_legendre_matrix(n):
    M = legendre_to_monomial_matrix(n)
    return np.linalg.inv(M)


def hermite_to_monomial_matrix(n):
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * M[j - 1, k - 1]
            val -= 2.0 * (k - 1) * M[j, k - 2]
            M[j, k] = val
    return M


def gegenbauer_to_monomial_matrix(n, lam=0.5):
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0 * lam
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * (k + lam - 1.0) * M[j - 1, k - 1]
            val -= (k + 2.0 * lam - 2.0) * M[j, k - 2]
            M[j, k] = val / k
    return M


def laguerre_to_monomial_matrix(n):
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[0, 1] = -1.0
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = (2 * k - 1) * M[j, k - 1]
            if j - 1 >= 0:
                val -= M[j - 1, k - 1]
            val -= (k - 1) * M[j, k - 2]
            M[j, k] = val / k
    return M






def triangle01_monomial_integral(i, j):
    if i < 0 or j < 0:
        return 0.0
    return float(np.exp(gammaln_stable(i + 1)
                        + gammaln_stable(j + 1)
                        - gammaln_stable(i + j + 3)))


def triangle_monomial_integral(i, j, t):
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (3, 2):
        raise ValueError("Triangle must be 3x2 array")
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]

    a = x2 - x1
    b = x3 - x1
    c = y2 - y1
    d = y3 - y1
    jac = abs(a * d - b * c)
    if jac < 1e-15:
        return 0.0

    pi = _poly_power_linear_2d(x1, a, b, i)

    pj = _poly_power_linear_2d(y1, c, d, j)

    p = _poly_product_2d(pi, pj)

    total = 0.0
    for (exp_r, exp_s), coeff in p.items():
        total += coeff * triangle01_monomial_integral(exp_r, exp_s)
    return jac * total


def _poly_power_linear_2d(const, coef_r, coef_s, n):
    if n < 0:
        return {}
    result = {}
    for j in range(n + 1):
        for k in range(n - j + 1):
            i = n - j - k
            coeff = trinomial_coefficient(i, j, k)
            coeff *= (const ** i) * (coef_r ** j) * (coef_s ** k)
            result[(j, k)] = result.get((j, k), 0.0) + coeff
    return result


def _poly_product_2d(p1, p2):
    result = {}
    for (i1, j1), c1 in p1.items():
        for (i2, j2), c2 in p2.items():
            key = (i1 + i2, j1 + j2)
            result[key] = result.get(key, 0.0) + c1 * c2
    return result






def log_sum_exp(a, b):
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def log_sum_exp_array(arr):
    arr = np.asarray(arr, dtype=np.float64)
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(a, fill_value, dtype=np.float64)
    mask = np.abs(b) > np.finfo(np.float64).eps * 100
    result[mask] = a[mask] / b[mask]
    return result
