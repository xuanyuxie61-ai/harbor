# -*- coding: utf-8 -*-

import numpy as np
import math






def lambert_w_high_accuracy(x):
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)


    em1 = -1.0 / math.e
    w = np.zeros_like(x)
    en = np.zeros_like(x)

    for idx in range(x.size):
        xi = x[idx]
        if xi < em1:
            w[idx] = np.nan
            en[idx] = np.nan
            continue
        if abs(xi) < 1e-30:
            w[idx] = 0.0
            en[idx] = 0.0
            continue

        if xi > 0:
            f = math.log(xi)
        else:
            f = -1e300


        c1 = 4.0 / 3.0
        c2 = 7.0 / 3.0
        c3 = 5.0 / 6.0
        c4 = 2.0 / 3.0

        if xi <= 6.46:
            wn = xi * (1.0 + c1 * xi) / (1.0 + xi * (c2 + c3 * xi))
            zn = f - wn - math.log(wn) if wn > 0 else -1e300
        else:
            wn = f
            zn = -math.log(wn) if wn > 0 else 1e300


        temp = 1.0 + wn
        y = 2.0 * temp * (temp + c4 * zn) - zn
        wn = wn * (1.0 + zn * y / (temp * (y - zn)))


        zn = f - wn - math.log(wn) if wn > 0 else -1e300
        temp = 1.0 + wn
        temp2 = temp + c4 * zn
        eni = zn * temp2 / (temp * temp2 - 0.5 * zn)
        wn = wn * (1.0 + eni)

        w[idx] = wn
        en[idx] = eni

    if scalar_input:
        return float(w[0]), float(en[0])
    return w.reshape(x.shape), en.reshape(x.shape)


def lambert_w_fast(x):
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    em1 = -1.0 / math.e
    w = np.zeros_like(x)
    en = np.zeros_like(x)

    for idx in range(x.size):
        xi = x[idx]
        if xi < em1:
            w[idx] = np.nan
            en[idx] = np.nan
            continue
        if abs(xi) < 1e-30:
            w[idx] = 0.0
            en[idx] = 0.0
            continue

        if xi > 0:
            f = math.log(xi)
            c1 = 4.0 / 3.0
            c2 = 7.0 / 3.0
            c3 = 5.0 / 6.0
            c4 = 2.0 / 3.0

            if xi <= 0.7385:
                wn = xi * (1.0 + c1 * xi) / (1.0 + xi * (c2 + c3 * xi))
            else:
                wn = f - 24.0 * ((f + 2.0) * f - 3.0) / ((0.7 * f + 58.0) * f + 127.0)

            zn = f - wn - math.log(wn) if wn > 0 else -1e300
            temp = 1.0 + wn
            y = 2.0 * temp * (temp + c4 * zn) - zn
            den = temp * (y - zn)
            if abs(den) < 1e-30:
                eni = 0.0
            else:
                eni = zn * y / den
            wn = wn * (1.0 + eni)
        else:

            wn = -1.0
            for _ in range(10):
                ew = math.exp(wn)
                num = wn * ew - xi
                den = (wn + 1.0) * ew
                if abs(den) < 1e-30:
                    break
                dw = num / den
                wn = wn - dw
                if abs(dw) < 1e-14:
                    break
            eni = 0.0

        w[idx] = wn
        en[idx] = eni

    if scalar_input:
        return float(w[0]), float(en[0])
    return w.reshape(x.shape), en.reshape(x.shape)


def lambert_w_convergence_rate(kappa):
    if kappa <= 1.0:
        return 0.0, 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)




    arg = -2.0 / math.sqrt(kappa)
    if arg >= -1.0 / math.e:
        w_val, _ = lambert_w_fast(arg)
        refined = -2.0 / math.sqrt(kappa) + w_val / kappa
    else:
        refined = math.log(rho)
    return rho, refined






def steinerberger_function(n, x):
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    val = np.zeros_like(x)
    for k in range(1, n + 1):
        val += np.abs(np.sin(math.pi * k * x)) / k
    if scalar_input:
        return float(val[0])
    return val.reshape(x.shape)


def steinerberger_integral_01(n):
    h = harmonic_number(n)
    return 2.0 * h / math.pi


def harmonic_number(n):
    if n <= 0:
        return 0.0


    if n <= 10000:
        return float(np.sum(1.0 / np.arange(1, n + 1)))
    gamma = 0.5772156649015328606
    return math.log(n) + gamma + 1.0 / (2.0 * n) - 1.0 / (12.0 * n * n)


def steinerberger_rhs(n, x_grid):
    x_grid = np.asarray(x_grid, dtype=float)
    f_vals = steinerberger_function(n, x_grid)
    h = 1.0
    if x_grid.size > 1:
        h = float(x_grid[1] - x_grid[0])
    return f_vals * (h ** 2)






def signed_power(x, p):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sign(x) * np.power(np.abs(x), p)
