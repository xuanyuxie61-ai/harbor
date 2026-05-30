# -*- coding: utf-8 -*-

import numpy as np


def chebyshev_derivative_series(coef):
    coef = np.asarray(coef, dtype=np.float64)
    npl = len(coef)
    if npl <= 1:
        return np.zeros_like(coef)
    n = npl - 1
    dcoef = np.zeros(npl, dtype=np.float64)
    xxn = coef[n - 1]
    dcoef[n - 1] = 2.0 * coef[n] * n
    dcoef[n] = 0.0
    for k in range(3, npl + 1):
        l = npl - k
        xxl = coef[l]
        dcoef[l] = dcoef[l + 2] + 2.0 * xxn * (l + 1)
        xxn = xxl
    return dcoef


def chebyshev_integral_series(coef):
    coef = np.asarray(coef, dtype=np.float64)
    npl = len(coef)
    if npl == 0:
        return np.array([0.0])
    n = npl - 1
    icoef = np.zeros(npl + 1, dtype=np.float64)

    icoef[1] = coef[0] - 0.5 * coef[1]
    for k in range(2, n):
        icoef[k] = (coef[k - 1] - coef[k + 1]) / (2.0 * k)
    icoef[n] = coef[n - 1] / (2.0 * n)
    icoef[n + 1] = coef[n] / (2.0 * (n + 1))

    alt_sum = np.sum(icoef[1:] * np.power(-1.0, np.arange(1, npl + 1)))
    icoef[0] = -alt_sum
    return icoef


def chebyshev_series_multiply(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    nc = max(na, nb)
    c = np.zeros(nc, dtype=np.float64)
    for j in range(na):
        for k in range(nb):
            if j + k < nc:
                c[j + k] += 0.5 * a[j] * b[k]
            if abs(j - k) < nc:
                c[abs(j - k)] += 0.5 * a[j] * b[k]
    return c


def chebyshev_series_invert(coef, y_target, max_iter=50, tol=1e-12):
    from chebyshev_spectral import clenshaw_evaluate

    x_sol = 0.0
    for _ in range(max_iter):
        y_val = clenshaw_evaluate(coef, x_sol)
        dcoef = chebyshev_derivative_series(coef)
        dy_val = clenshaw_evaluate(dcoef, x_sol)
        if abs(dy_val) < 1e-14:
            break
        dx = (y_val - y_target) / dy_val
        x_sol -= dx
        x_sol = np.clip(x_sol, -1.0, 1.0)
        if abs(dx) < tol:
            break
    return x_sol


def chebyshev_l2_norm(coef):
    coef = np.asarray(coef, dtype=np.float64)
    s = 2.0 * coef[0] * coef[0] + np.sum(coef[1:] ** 2)
    return np.sqrt(0.5 * np.pi * s)
