
import numpy as np
from math import factorial, sqrt, pi, sin, cos, exp, log


def spherical_bessel_j(l, x):
    if l < 0:
        raise ValueError("角动量量子数 l 必须非负")
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    result = np.zeros_like(x)


    tiny = 1e-8
    small_mask = np.abs(x) < tiny
    if np.any(small_mask):
        xs = x[small_mask]
        double_fact = 1.0
        for k in range(1, 2 * l + 2, 2):
            double_fact *= k
        result[small_mask] = (xs ** l) / double_fact


    med_mask = (~small_mask) & (np.abs(x) <= 30.0)
    if np.any(med_mask):
        xm = x[med_mask]
        j0 = np.sin(xm) / xm
        if l == 0:
            result[med_mask] = j0
        else:
            j1 = (np.sin(xm) / (xm ** 2)) - np.cos(xm) / xm
            if l == 1:
                result[med_mask] = j1
            else:
                j_prev2 = j0
                j_prev1 = j1
                for ll in range(1, l):
                    j_curr = (2 * ll + 1) / xm * j_prev1 - j_prev2
                    j_prev2 = j_prev1
                    j_prev1 = j_curr
                result[med_mask] = j_prev1


    large_mask = np.abs(x) > 30.0
    if np.any(large_mask):
        xl = x[large_mask]
        result[large_mask] = np.sin(xl - l * pi / 2.0) / xl

    return result.item() if scalar_input else result.reshape(np.asarray(x).shape)


def spherical_neumann_n(l, x):
    if l < 0:
        raise ValueError("角动量量子数 l 必须非负")
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    result = np.zeros_like(x)

    tiny = 1e-8
    small_mask = np.abs(x) < tiny
    if np.any(small_mask):
        xs = x[small_mask]
        if l == 0:
            result[small_mask] = -1.0 / xs
        else:
            double_fact = 1.0
            for k in range(1, 2 * l, 2):
                double_fact *= k
            result[small_mask] = -double_fact / (xs ** (l + 1))

    med_mask = (~small_mask) & (np.abs(x) <= 30.0)
    if np.any(med_mask):
        xm = x[med_mask]
        n0 = -np.cos(xm) / xm
        if l == 0:
            result[med_mask] = n0
        else:
            n1 = -np.cos(xm) / (xm ** 2) - np.sin(xm) / xm
            if l == 1:
                result[med_mask] = n1
            else:
                n_prev2 = n0
                n_prev1 = n1
                for ll in range(1, l):
                    n_curr = (2 * ll + 1) / xm * n_prev1 - n_prev2
                    n_prev2 = n_prev1
                    n_prev1 = n_curr
                result[med_mask] = n_prev1

    large_mask = np.abs(x) > 30.0
    if np.any(large_mask):
        xl = x[large_mask]
        result[large_mask] = -np.cos(xl - l * pi / 2.0) / xl

    return result.item() if scalar_input else result.reshape(np.asarray(x).shape)


def sine_integral_si(x):
    x = float(x)
    p2 = pi / 2.0
    el = 0.5772156649015329
    epsilon = 1.0e-15
    x2 = x * x
    xabs = abs(x)
    xsign = -1.0 if x < 0.0 else 1.0

    if xabs == 0.0:
        return 0.0

    elif xabs <= 16.0:

        xr = xabs
        value = xabs
        for k in range(1, 40):
            xr = -0.5 * xr * (2 * k - 1) / k / (4 * k * k + 4 * k + 1) * x2
            value = value + xr
            if abs(xr) < abs(value) * epsilon:
                return xsign * value
        return xsign * value

    elif xabs <= 32.0:

        m = int(47.2 + 0.82 * xabs)
        bj = np.zeros(m)
        xa1 = 0.0
        xa0 = 1.0e-100
        for k in range(m - 1, -1, -1):
            xa = 4.0 * (k + 1) * xa0 / xabs - xa1
            bj[k] = xa
            xa1 = xa0
            xa0 = xa
        xs = bj[0]
        for k in range(2, m, 2):
            xs = xs + 2.0 * bj[k]
        bj[0] = bj[0] / xs
        for k in range(1, m):
            bj[k] = bj[k] / xs

        xr = 1.0
        xg1 = bj[0]
        for k in range(2, m):
            xr = 0.25 * xr * (2.0 * k - 3.0) ** 2 / ((k - 1.0) * (2.0 * k - 1.0) ** 2) * xabs
            xg1 = xg1 + bj[k] * xr

        xr = 1.0
        xg2 = bj[0]
        for k in range(2, m):
            xr = 0.25 * xr * (2.0 * k - 5.0) ** 2 / ((k - 1.0) * (2.0 * k - 3.0) ** 2) * xabs
            xg2 = xg2 + bj[k] * xr

        xcs = cos(xabs / 2.0)
        xss = sin(xabs / 2.0)
        value = xsign * (xabs * xcs * xg1 + 2.0 * xss * xg2 - sin(xabs))
        return value

    else:

        xr = 1.0
        xf = 1.0
        for k in range(1, 10):
            xr = -2.0 * xr * k * (2 * k - 1) / x2
            xf = xf + xr
        xr = 1.0 / xabs
        xg = xr
        for k in range(1, 9):
            xr = -2.0 * xr * (2 * k + 1) * k / x2
            xg = xg + xr
        value = xsign * (p2 - xf * cos(xabs) / xabs - xg * sin(xabs) / xabs)
        return value


def associated_legendre(l, m, x):
    if abs(m) > l:
        return 0.0
    if abs(x) > 1.0:
        raise ValueError("|x| 必须 ≤ 1 以保证勒让德多项式定义")

    m_abs = abs(m)

    pmm = 1.0
    if m_abs > 0:
        somx2 = sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m_abs + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m_abs:
        return pmm

    pmmp1 = x * (2 * m_abs + 1) * pmm
    if l == m_abs + 1:
        return pmmp1

    pll = 0.0
    for ll in range(m_abs + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m_abs - 1) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll

    return pmmp1 if l == m_abs + 1 else pll


def spherical_harmonic_Y(l, m, theta, phi):
    x = cos(theta)
    plm = associated_legendre(l, m, x)
    norm = sqrt((2 * l + 1) * factorial(l - abs(m)) / (4 * pi * factorial(l + abs(m))))
    phase = (-1) ** ((m + abs(m)) // 2)
    return phase * norm * plm * complex(cos(m * phi), sin(m * phi))


def nuclear_form_factor(q, A, Z, R0=1.2):
    R = R0 * (A ** (1.0 / 3.0))
    if abs(q) < 1e-10:
        return 1.0
    qr = q * R
    j0 = spherical_bessel_j(0, qr)

    F = 3.0 * j0 / qr ** 2
    return abs(F) ** 2
