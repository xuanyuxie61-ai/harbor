# -*- coding: utf-8 -*-

import numpy as np
from numpy.polynomial import polynomial as P


def lambert_w(x, branch=0):
    x = np.asarray(x, dtype=float)
    w = np.full_like(x, np.nan, dtype=float)
    em1 = -1.0 / np.e


    if branch == 0:
        valid = x >= em1
    elif branch == -1:
        valid = (x >= em1) & (x < 0.0)
    else:
        return w

    xv = x[valid]
    if xv.size == 0:
        return w



    if branch == 0:

        near_branch = xv < -0.2
        far = ~near_branch

        wv = np.empty_like(xv)
        if np.any(near_branch):
            delta = xv[near_branch] - em1
            p = np.sqrt(2.0 * np.e * delta)
            wv[near_branch] = -1.0 + p - (np.e / 3.0) * delta + \
                              (11.0 * np.sqrt(2.0) / 72.0) * p * delta
        if np.any(far):

            lx = np.log(xv[far])
            llx = np.log(lx)
            wv[far] = lx - llx + llx / lx
    else:


        near_branch = xv < -0.1
        far = ~near_branch

        wv = np.empty_like(xv)
        if np.any(near_branch):
            delta = xv[near_branch] - em1
            p = np.sqrt(2.0 * np.e * delta)
            wv[near_branch] = -1.0 - p - (np.e / 3.0) * delta - \
                              (11.0 * np.sqrt(2.0) / 72.0) * p * delta
        if np.any(far):
            lx = np.log(-xv[far])
            llx = np.log(-lx)
            wv[far] = lx - llx + llx / lx





    for _ in range(8):
        ew = np.exp(wv)
        we = wv * ew - xv
        denom = (wv + 1.0) * ew - (wv + 2.0) * we / (2.0 * wv + 2.0)

        denom = np.where(np.abs(denom) < 1e-300, np.copysign(1e-300, denom), denom)
        wv = wv - we / denom

    w[valid] = wv
    return w


def fresnel_integrals(x):
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    sgn = np.sign(x)
    C = np.zeros_like(x, dtype=float)
    S = np.zeros_like(x, dtype=float)


    region1 = ax < 2.5
    if np.any(region1):
        t = ax[region1]
        t2 = t * t



        c_val = np.zeros_like(t)
        s_val = np.zeros_like(t)

        for i, ti in enumerate(t):
            c_sum = 0.0
            s_sum = 0.0
            term_c = ti
            term_s = (np.pi / 2.0) * ti**3 / 3.0
            c_sum += term_c
            s_sum += term_s
            for n in range(1, 50):

                term_c *= -(np.pi / 2.0)**2 * ti**4 / ((2*n) * (2*n - 1) * (4*n + 1) / (4*n - 3))

                term_s *= -(np.pi / 2.0)**2 * ti**4 / ((2*n + 1) * (2*n) * (4*n + 3) / (4*n - 1))
                if np.abs(term_c) < 1e-15 and np.abs(term_s) < 1e-15:
                    break
                c_sum += term_c
                s_sum += term_s
            c_val[i] = c_sum
            s_val[i] = s_sum
        C[region1] = c_val
        S[region1] = s_val


    region2 = (ax >= 2.5) & (ax < 4.5)
    if np.any(region2):
        t = ax[region2]
        t0 = 0.5 * np.pi * t * t




        f = np.zeros_like(t)
        g = np.zeros_like(t)
        for i, ti in enumerate(t):
            u = 1.0 / ((0.5 * np.pi * ti * ti)**2)

            f_sum = 1.0
            term = 1.0
            for n in range(1, 20):
                term *= -(4*n - 3) * (4*n - 1) * u / ((4*n - 4) * (4*n) if n > 1 else 1)
                if np.abs(term) < 1e-15:
                    break
                f_sum += term

            g_sum = 1.0
            term = 1.0
            for n in range(1, 12):
                term *= -(4*n - 1) * (4*n + 1) * u / ((4*n - 2) * (4*n + 2))
                if np.abs(term) < 1e-15:
                    break
                g_sum += term
            f[i] = f_sum
            g[i] = g_sum
        st0 = np.sin(t0)
        ct0 = np.cos(t0)
        C[region2] = 0.5 + (f * st0 - g * ct0) / (np.pi * t)
        S[region2] = 0.5 - (f * ct0 + g * st0) / (np.pi * t)


    region3 = ax >= 4.5
    if np.any(region3):
        t = ax[region3]
        t0 = 0.5 * np.pi * t * t

        t0_red = t0 % (2.0 * np.pi)
        st0 = np.sin(t0_red)
        ct0 = np.cos(t0_red)

        f = np.ones_like(t)
        g = np.ones_like(t)
        u = 1.0 / t**2
        for n in range(1, 10):
            coeff_f = 1.0
            for k in range(1, n + 1):
                coeff_f *= (4.0 * k - 3.0) * (4.0 * k - 1.0)
            from math import factorial
            coeff_f /= (np.pi * t)**(2 * n) * factorial(2 * n) if 2*n < 20 else 1e300

            cf = 1.0
            cg = 1.0
            for k in range(1, n + 1):
                cf *= (4.0 * k - 3.0) * (4.0 * k - 1.0) / ((2.0 * k - 1.0) * 2.0 * k)
                cg *= (4.0 * k - 1.0) * (4.0 * k + 1.0) / ((2.0 * k) * (2.0 * k + 1.0))
            f += cf * ((-1)**n) * u**n
            g += cg * ((-1)**n) * u**n
        C[region3] = 0.5 + (f * st0 - g * ct0) / (np.pi * t)
        S[region3] = 0.5 - (f * ct0 + g * st0) / (np.pi * t)


    C = sgn * C
    S = sgn * S
    return C, S


def fraunhofer_diffraction_particle_size(radius, wavelength, theta):
    from scipy.special import j1
    k = 2.0 * np.pi / wavelength
    x = k * radius * np.sin(theta)

    x = np.where(np.abs(x) < 1e-10, 1e-10, x)
    intensity = (2.0 * j1(x) / x) ** 2
    return intensity
