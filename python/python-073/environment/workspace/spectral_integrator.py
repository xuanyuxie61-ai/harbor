# -*- coding: utf-8 -*-

import numpy as np
from math import pi, sin, cos, sqrt, atan2


def chebyshev1_gauss_weights(n):
    j = np.arange(1, n + 1)
    x = np.cos((2.0 * j - 1.0) * pi / (2.0 * n))
    w = np.full(n, pi / n)
    return x, w


def chebyshev1_exactness_test(n, degree_max):
    x, w = chebyshev1_gauss_weights(n)
    results = []
    for m in range(degree_max + 1):

        if m % 2 == 1:
            exact = 0.0
        else:
            top = 1.0
            bot = 1.0
            for i in range(2, m + 1, 2):
                top *= (i - 1)
                bot *= i
            exact = pi * top / bot


        fvals = x ** m
        quad = np.sum(w * fvals)

        if abs(exact) < 1e-15:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)
        results.append((m, err))
    return results


def integrate_boundary_layer_growth(x_coords, alpha_i_vals):
    x = np.asarray(x_coords)
    a = np.asarray(alpha_i_vals)
    n = len(x)
    N = np.zeros(n)

    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if abs(dx) < 1e-15:
            N[i] = N[i - 1]
            continue

        N[i] = N[i - 1] - 0.5 * dx * (a[i] + a[i - 1])
    return N


def sphere01_triangle_area(a_xyz, b_xyz, c_xyz):
    a_xyz = np.asarray(a_xyz) / np.linalg.norm(a_xyz)
    b_xyz = np.asarray(b_xyz) / np.linalg.norm(b_xyz)
    c_xyz = np.asarray(c_xyz) / np.linalg.norm(c_xyz)


    a_len = atan2(np.linalg.norm(np.cross(b_xyz, c_xyz)), np.dot(b_xyz, c_xyz))
    b_len = atan2(np.linalg.norm(np.cross(c_xyz, a_xyz)), np.dot(c_xyz, a_xyz))
    c_len = atan2(np.linalg.norm(np.cross(a_xyz, b_xyz)), np.dot(a_xyz, b_xyz))

    s = 0.5 * (a_len + b_len + c_len)

    tan_sq = (np.tan(0.5 * max(s - a_len, 0.0)) *
              np.tan(0.5 * max(s - b_len, 0.0)) *
              np.tan(0.5 * max(s - c_len, 0.0)) *
              max(np.tan(0.5 * s), 1e-15))
    E = 4.0 * np.arctan(sqrt(max(tan_sq, 0.0)))
    return E


def sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3):
    a = np.asarray(a_xyz)
    b = np.asarray(b_xyz)
    c = np.asarray(c_xyz)
    p = (f1 * a + f2 * b + f3 * c) / max(f1 + f2 + f3, 1e-15)
    p = p / np.linalg.norm(p)
    return p


def sphere_triangle_quad_icos1c(a_xyz, b_xyz, c_xyz, factor, func):
    result = 0.0
    area_total = 0.0
    node_num = 0


    for f3 in range(1, 3 * factor - 1, 3):
        for f2 in range(1, 3 * factor - f3, 3):
            f1 = 3 * factor - f3 - f2
            node = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3)
            a2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 2, f2 - 1, f3 - 1)
            b2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 1, f2 + 2, f3 - 1)
            c2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 1, f2 - 1, f3 + 2)
            area = sphere01_triangle_area(a2, b2, c2)
            v = func(node)
            node_num += 1
            result += area * v
            area_total += area


    for f3 in range(2, 3 * factor - 3, 3):
        for f2 in range(2, 3 * factor - f3 - 2, 3):
            f1 = 3 * factor - f3 - f2
            node = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3)
            a2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 2, f2 + 1, f3 + 1)
            b2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 1, f2 - 2, f3 + 1)
            c2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 1, f2 + 1, f3 - 2)
            area = sphere01_triangle_area(a2, b2, c2)
            v = func(node)
            node_num += 1
            result += area * v
            area_total += area

    return result, node_num


def integrate_wavevector_direction(k_mag, theta_range, phi_range, integrand, n_theta=32, n_phi=16):
    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range


    theta_nodes = np.linspace(theta_min, theta_max, n_theta)
    phi_nodes = np.linspace(phi_min, phi_max, n_phi)

    d_theta = theta_nodes[1] - theta_nodes[0]
    d_phi = phi_nodes[1] - phi_nodes[0]

    total = 0.0
    for i in range(n_theta):
        for j in range(n_phi):
            th = theta_nodes[i]
            ph = phi_nodes[j]
            kx = k_mag * sin(ph) * cos(th)
            ky = k_mag * cos(ph)
            kz = k_mag * sin(ph) * sin(th)
            jac = k_mag**2 * sin(ph)
            total += integrand(kx, ky, kz) * jac * d_theta * d_phi

    return total


def amplification_factor_integral(Re_x_range, alpha_i_interp, method='simpson'):
    Re = np.asarray(Re_x_range)
    if callable(alpha_i_interp):
        alpha_i = alpha_i_interp(Re)
    else:
        alpha_i = np.asarray(alpha_i_interp)
        if len(alpha_i) != len(Re):
            raise ValueError("alpha_i_interp 数组长度与 Re 不匹配")

    if method == 'trapz':
        N = np.zeros_like(Re)
        for i in range(1, len(Re)):
            N[i] = N[i - 1] - 0.5 * (Re[i] - Re[i - 1]) * (alpha_i[i] + alpha_i[i - 1])
    elif method == 'simpson':
        N = np.zeros_like(Re)
        for i in range(2, len(Re), 2):
            h = Re[i] - Re[i - 2]
            N[i] = N[i - 2] - h / 6.0 * (alpha_i[i - 2] + 4.0 * alpha_i[i - 1] + alpha_i[i])
            if i - 1 < len(N):
                N[i - 1] = 0.5 * (N[i - 2] + N[i])
    else:
        raise ValueError(f"未知积分方法: {method}")

    return Re, N


def chebyshev_transform(fvals, N=None):
    if N is None:
        N = len(fvals) - 1
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    j = np.arange(N + 1)
    x = np.cos(pi * j / N)

    coeffs = np.zeros(N + 1)
    for k in range(N + 1):
        Tk = np.cos(k * np.arccos(np.clip(x, -1.0, 1.0)))
        coeffs[k] = (2.0 / N) * np.sum(fvals * Tk / c)
    coeffs[0] *= 0.5
    coeffs[N] *= 0.5
    return coeffs
