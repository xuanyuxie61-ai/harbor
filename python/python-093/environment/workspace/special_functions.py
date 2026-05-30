#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sincu_fun(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    result[mask] = np.sin(x_masked) / x_masked
    return result


def sincn_fun(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    result[mask] = np.sin(pix) / pix
    return result


def sincu_deriv(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    result[mask] = (x_masked * np.cos(x_masked) - np.sin(x_masked)) / (x_masked ** 2)
    return result


def sincn_deriv(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    result[mask] = (pix * np.cos(pix) - np.sin(pix)) / (np.pi * x_masked ** 2)
    return result


def sincu_deriv2(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    sx = np.sin(x_masked)
    cx = np.cos(x_masked)
    result[mask] = ((2.0 - x_masked ** 2) * sx - 2.0 * x_masked * cx) / (x_masked ** 3)
    return result


def sincn_deriv2(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    sp = np.sin(pix)
    cp = np.cos(pix)
    result[mask] = np.pi * ((2.0 - pix ** 2) * sp - 2.0 * pix * cp) / (pix ** 3)
    return result


def _cisi_series(x):
    x = float(x)
    gamma = 0.577215664901533
    ci = gamma + np.log(x)
    si = x
    term_ci = 1.0
    term_si = x
    k = 1
    while True:
        term_ci *= -x * x / ((2 * k - 1) * (2 * k))
        term_si *= -x * x / ((2 * k) * (2 * k + 1))
        dci = term_ci / (2 * k)
        dsi = term_si / (2 * k + 1)
        ci += dci
        si += dsi
        if abs(dci) < 1e-15 and abs(dsi) < 1e-15:
            break
        k += 1
        if k > 200:
            break
    return ci, si


def _cisi_bessel(x):
    x = float(x)

    f = (1.0
         + 3.0381634e-2 / x ** 2
         - 3.4686916e-4 / x ** 4
         + 7.2189434e-6 / x ** 6)
    g = (1.0 / x
         - 1.9203743e-2 / x ** 3
         + 3.4108765e-4 / x ** 5
         - 5.2203843e-6 / x ** 7)
    ci = f * np.sin(x) / x - g * np.cos(x) / x
    si = np.pi / 2.0 - f * np.cos(x) / x - g * np.sin(x) / x
    return ci, si


def _cisi_asymptotic(x):
    x = float(x)
    x2 = x * x

    p = 1.0
    q = 1.0 / x
    term_p = 1.0
    term_q = 1.0 / x
    for k in range(1, 9):
        term_p *= -(2 * k - 1) * (2 * k) / x2
        term_q *= -(2 * k) * (2 * k + 1) / x2
        p += term_p
        q += term_q
    sx = np.sin(x)
    cx = np.cos(x)
    ci = sx / x * p - cx / x * q
    si = np.pi / 2.0 - cx / x * p - sx / x * q
    return ci, si


def cisi(x):
    x = np.asarray(x, dtype=np.float64)
    ci = np.full_like(x, np.nan, dtype=np.float64)
    si = np.full_like(x, np.nan, dtype=np.float64)


    mask1 = (x > 0) & (x <= 16)
    if np.any(mask1):
        for idx in np.where(mask1)[0]:
            c, s = _cisi_series(x[idx])
            ci[idx] = c
            si[idx] = s


    mask2 = (x > 16) & (x <= 32)
    if np.any(mask2):
        for idx in np.where(mask2)[0]:
            c, s = _cisi_bessel(x[idx])
            ci[idx] = c
            si[idx] = s


    mask3 = x > 32
    if np.any(mask3):
        for idx in np.where(mask3)[0]:
            c, s = _cisi_asymptotic(x[idx])
            ci[idx] = c
            si[idx] = s

    return ci, si


def sincu_antideriv(x):
    x = np.asarray(x, dtype=np.float64)
    ci, si = cisi(np.abs(x))
    return np.sign(x) * si


def sincn_antideriv(x):
    x = np.asarray(x, dtype=np.float64)
    ci, si = cisi(np.abs(np.pi * x))
    return np.sign(x) * si / np.pi


def alnorm(x, upper=False):
    x = np.asarray(x, dtype=np.float64)
    p = 0.2316419
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429

    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    z = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    lower_tail = 1.0 - z * poly

    lower_tail = np.clip(lower_tail, 0.0, 1.0)
    result = np.where(sign < 0, 1.0 - lower_tail, lower_tail)
    if upper:
        return 1.0 - result
    return result


def gammaln_stable(x):
    from scipy.special import gammaln
    x = np.asarray(x, dtype=np.float64)
    return gammaln(x)
