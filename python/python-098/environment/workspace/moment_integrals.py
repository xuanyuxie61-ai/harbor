# -*- coding: utf-8 -*-

import numpy as np


def square01_monomial_integral(exponents):
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != 2:
        raise ValueError("exponents 必须为 [e1, e2]")
    if np.any(exponents < 0):
        raise ValueError("指数必须非负")
    integral = 1.0
    for e in exponents:
        integral /= float(e + 1)
    return integral


def squaresym_monomial_integral(exponents):
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != 2:
        raise ValueError("exponents 必须为 [e1, e2]")
    if np.any(exponents < 0):
        raise ValueError("指数必须非负")
    if np.any(exponents % 2 == 1):
        return 0.0
    integral = 4.0
    for e in exponents:
        integral /= float(e + 1)
    return integral


def gauss_legendre_nodes_weights(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_2d_gauss_legendre(func, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), n=8):
    x_nodes, x_weights = gauss_legendre_nodes_weights(n)
    y_nodes, y_weights = gauss_legendre_nodes_weights(n)


    x_mapped = 0.5 * (xlim[1] - xlim[0]) * x_nodes + 0.5 * (xlim[0] + xlim[1])
    y_mapped = 0.5 * (ylim[1] - ylim[0]) * y_nodes + 0.5 * (ylim[0] + ylim[1])
    wx = 0.5 * (xlim[1] - xlim[0]) * x_weights
    wy = 0.5 * (ylim[1] - ylim[0]) * y_weights

    integral = 0.0
    for i in range(n):
        for j in range(n):
            integral += wx[i] * wy[j] * func(x_mapped[i], y_mapped[j])
    return integral


def compute_field_moments(field_func, max_order, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5)):
    moments = np.zeros((max_order + 1, max_order + 1), dtype=float)
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            def integrand(x, y):
                val = field_func(x, y)
                if not np.isfinite(val):
                    return 0.0
                return (x ** p) * (y ** q) * val
            moments[p, q] = integrate_2d_gauss_legendre(
                integrand, xlim=xlim, ylim=ylim, n=max(4, max_order + 2)
            )
    return moments


def verify_monomial_integrals(max_order):
    max_rel_error = 0.0
    for e1 in range(max_order + 1):
        for e2 in range(max_order + 1):
            exponents = [e1, e2]
            analytic = square01_monomial_integral(exponents)

            def f(x, y):
                return (x ** e1) * (y ** e2)

            numerical = integrate_2d_gauss_legendre(f, xlim=(0.0, 1.0), ylim=(0.0, 1.0), n=8)
            if abs(analytic) > 1e-12:
                rel_error = abs(analytic - numerical) / abs(analytic)
            else:
                rel_error = abs(numerical)
            if rel_error > max_rel_error:
                max_rel_error = rel_error
    return max_rel_error


def patch_impedance_moment(patch_size, wavelength, order=2):
    k0 = 2.0 * np.pi / wavelength

    def roof_func(x, y):

        a = patch_size
        fx = max(0.0, 1.0 - 2.0 * abs(x) / a)
        fy = max(0.0, 1.0 - 2.0 * abs(y) / a)
        return fx * fy

    moments = compute_field_moments(roof_func, order,
                                    xlim=(-patch_size / 2.0, patch_size / 2.0),
                                    ylim=(-patch_size / 2.0, patch_size / 2.0))
    return moments, k0
