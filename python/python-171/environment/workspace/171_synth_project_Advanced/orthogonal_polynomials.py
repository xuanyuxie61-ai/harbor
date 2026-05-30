# -*- coding: utf-8 -*-

import numpy as np
import math






def laguerre_polynomial(m, n, x):
    x = np.asarray(x, dtype=float).flatten()
    m = x.size
    if n < 0:
        return np.zeros((m, 0), dtype=float)
    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v
    v[:, 1] = 1.0 - x
    for j in range(2, n + 1):
        v[:, j] = (((2 * j - 1) - x) * v[:, j - 1] + (-j + 1) * v[:, j - 2]) / j
    return v


def generalized_laguerre_function(m, n, alpha, x):
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for generalized Laguerre.")
    x = np.asarray(x, dtype=float).flatten()
    m = x.size
    if n < 0:
        return np.zeros((m, 0), dtype=float)
    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v
    v[:, 1] = 1.0 + alpha - x
    for i in range(2, n + 1):
        v[:, i] = (((2 * i - 1 + alpha) - x) * v[:, i - 1] + (-i + 1 - alpha) * v[:, i - 2]) / i
    return v






def hermite_probabilist(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    He_prev2 = np.ones_like(x)
    He_prev1 = x.copy()
    for k in range(2, n + 1):
        He_curr = x * He_prev1 - (k - 1) * He_prev2
        He_prev2, He_prev1 = He_prev1, He_curr
    return He_prev1


def hermite_physicist(n, x):
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    H_prev2 = np.ones_like(x)
    H_prev1 = 2.0 * x
    for k in range(2, n + 1):
        H_curr = 2.0 * x * H_prev1 - 2.0 * (k - 1) * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1






def imtqlx(n, d, e, z):
    d = np.asarray(d, dtype=float).copy().flatten()
    e = np.asarray(e, dtype=float).copy().flatten()
    z = np.asarray(z, dtype=float).copy().flatten()

    if n == 1:
        return d, z

    itn = 30
    prec = np.finfo(float).eps
    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = l
            while m < n - 1:
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1

            p = d[l]
            if m == l:
                break

            if j >= itn:
                raise RuntimeError("IMTQLX: iteration limit exceeded.")

            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            g = d[m] - p + e[l] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p_local = 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(f) >= abs(g):
                    c_val = g / f
                    r = math.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c_val * s
                else:
                    s_val = f / g
                    r = math.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s_val * c

                g = d[i + 1] - p_local
                r = (d[i] - g) * s + 2.0 * c * b
                p_local = s * r
                d[i + 1] = g + p_local
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[l] = d[l] - p_local
            e[l] = g
            e[m] = 0.0


    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z






def gauss_laguerre_rule(n):
    zemu = 1.0
    bj = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        bj[i - 1] = math.sqrt(i)
    x = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        x[i - 1] = 2.0 * i - 1.0
    w = np.zeros(n, dtype=float)
    w[0] = math.sqrt(zemu)

    x, w = imtqlx(n, x, bj, w)
    w = w ** 2
    return x, w


def gauss_generalized_laguerre_rule(n, alpha):
    if alpha < 0:
        raise ValueError("alpha must be >= 0 for generalized Laguerre quadrature.")
    zemu = math.gamma(alpha + 1.0)
    bj = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        bj[i - 1] = math.sqrt(i * (i + alpha))
    x = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        x[i - 1] = 2.0 * i - 1.0 + alpha
    w = np.zeros(n, dtype=float)
    w[0] = math.sqrt(zemu)

    x, w = imtqlx(n, x, bj, w)
    w = w ** 2
    return x, w






def gauss_hermite_rule(n):
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes.astype(float), weights.astype(float)






def build_polynomial_preconditioner_spectrum(n, poly_type='laguerre', param=0.0):
    if poly_type == 'laguerre':
        return gauss_laguerre_rule(n)
    elif poly_type == 'generalized_laguerre':
        return gauss_generalized_laguerre_rule(n, param)
    elif poly_type == 'hermite':
        return gauss_hermite_rule(n)
    else:
        raise ValueError(f"Unknown poly_type: {poly_type}")
