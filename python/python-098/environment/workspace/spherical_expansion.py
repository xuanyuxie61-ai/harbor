# -*- coding: utf-8 -*-

import numpy as np
from math import factorial


def associated_legendre(l_max, m, x):
    x = float(x)
    m_abs = abs(m)
    if l_max < m_abs:
        return np.zeros(l_max + 1)

    plm = np.zeros(l_max + 1)

    p_mm = 1.0
    if m_abs > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m_abs + 1):
            p_mm *= -fact * somx2
            fact += 2.0
    plm[m_abs] = p_mm

    if l_max > m_abs:
        p_mp1m = x * (2.0 * m_abs + 1.0) * p_mm
        plm[m_abs + 1] = p_mp1m

    for l in range(m_abs + 2, l_max + 1):
        plm[l] = ((2.0 * l - 1.0) * x * plm[l - 1] -
                  (l + m_abs - 1.0) * plm[l - 2]) / (l - m_abs)
    return plm


def spherical_harmonic_y(l, m, theta, phi):
    if not (-l <= m <= l):
        raise ValueError(f"m={m} 不在 [-l, l] = [{-l}, {l}] 范围内")
    theta = float(theta)
    phi = float(phi)
    x = np.cos(theta)
    plm = associated_legendre(l, abs(m), x)

    m_abs = abs(m)
    norm_coeff = np.sqrt(
        (2.0 * l + 1.0) / (4.0 * np.pi) *
        factorial(l - m_abs) / factorial(l + m_abs)
    )

    y_val = norm_coeff * plm[l] * np.exp(1j * m * phi)
    return y_val


def expand_far_field_spherical(field_samples, theta_grid, phi_grid, l_max):
    field_samples = np.asarray(field_samples, dtype=complex)
    theta_grid = np.asarray(theta_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    ntheta = len(theta_grid)
    nphi = len(phi_grid)
    if field_samples.shape != (ntheta, nphi):
        raise ValueError("field_samples shape 与 theta_grid/phi_grid 不匹配")

    coeffs = {}
    dtheta = np.gradient(theta_grid)
    dphi = np.gradient(phi_grid)


    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    dTH, dPH = np.meshgrid(dtheta, dphi, indexing='ij')
    weights = np.sin(TH) * dTH * dPH

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ylm_grid = np.zeros((ntheta, nphi), dtype=complex)
            for i in range(ntheta):
                for j in range(nphi):
                    ylm_grid[i, j] = spherical_harmonic_y(l, m, theta_grid[i], phi_grid[j])

            integrand = field_samples * np.conj(ylm_grid) * weights
            a_lm = np.sum(integrand)
            coeffs[(l, m)] = a_lm
    return coeffs


def reconstruct_far_field(coeffs, theta_grid, phi_grid):
    theta_grid = np.asarray(theta_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    ntheta = len(theta_grid)
    nphi = len(phi_grid)
    field = np.zeros((ntheta, nphi), dtype=complex)
    for (l, m), a_lm in coeffs.items():
        for i in range(ntheta):
            for j in range(nphi):
                field[i, j] += a_lm * spherical_harmonic_y(l, m, theta_grid[i], phi_grid[j])
    return field


def vector_spherical_harmonic_m(l, m, theta, phi):
    ylm = spherical_harmonic_y(l, m, theta, phi)

    delta = 1e-6
    ylm_p = spherical_harmonic_y(l, m, min(theta + delta, np.pi - 1e-8), phi)
    ylm_m = spherical_harmonic_y(l, m, max(theta - delta, 1e-8), phi)
    dydtheta = (ylm_p - ylm_m) / (2.0 * delta)
    return dydtheta


def vector_spherical_harmonic_n(l, m, theta, phi):
    ylm = spherical_harmonic_y(l, m, theta, phi)
    return l * (l + 1.0) * ylm


def scattering_coefficients_mie(l_max, k, a, eps_r, mu_r):
    ka = k * a
    a_coeffs = np.zeros(l_max + 1, dtype=complex)
    if ka <= 0.0:
        return a_coeffs

    a_coeffs[1] = 1j * (2.0 / 3.0) * (ka ** 3) * (eps_r - 1.0) / (eps_r + 2.0)

    for l in range(2, l_max + 1):
        a_coeffs[l] = a_coeffs[1] * (ka ** (2 * l - 2)) / (2.0 ** l)

        if abs(a_coeffs[l]) > 1e6:
            a_coeffs[l] = 0.0
    return a_coeffs
