# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma


def disk_monomial_integral(e1, e2):
    if e1 < 0 or e2 < 0:
        raise ValueError("Exponents must be non-negative.")
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    num = 2.0 * gamma(0.5 * (e1 + 1)) * gamma(0.5 * (e2 + 1))
    den = (e1 + e2 + 2) * gamma(0.5 * (e1 + e2 + 2))
    return num / den


def disk_gaussian_integral(sigma, n_theta=128, n_r=64):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    r = np.linspace(0.0, 1.0, n_r)
    dtheta = 2.0 * np.pi / n_theta
    dr = 1.0 / (n_r - 1) if n_r > 1 else 1.0
    val = 0.0
    for ti in theta:
        for ri in r[1:]:
            x = ri * np.cos(ti)
            y = ri * np.sin(ti)
            val += np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) * ri * dr * dtheta
    return val


def transfer_probability(b, R_grazing, sigma_b, P0=0.1):
    return P0 * np.exp(-((b - R_grazing) ** 2) / (sigma_b ** 2))


def transfer_cross_section(R_grazing, sigma_b, P0=0.1, b_min=None, b_max=30.0):
    if b_min is None:
        b_min = max(0.0, R_grazing - 3.0 * sigma_b)



    n = 2000
    b_vals = np.linspace(b_min, b_max, n)
    db = b_vals[1] - b_vals[0]
    P_vals = transfer_probability(b_vals, R_grazing, sigma_b, P0)
    integrand = 2.0 * np.pi * b_vals * P_vals
    return np.trapz(integrand, b_vals)


def coulomb_breakup_cross_section(E_beam, Z_p, Z_t, A_p, A_t,
                                  E_bind, n_points=1000):
    from constants import FINE_STRUCTURE, HBAR_C

    mu = (A_p * A_t) / (A_p + A_t) * 938.0
    v = np.sqrt(2.0 * E_beam / mu) * HBAR_C
    if v <= 0:
        return 0.0
    b_min = 1.2 * (A_p ** (1.0 / 3.0) + A_t ** (1.0 / 3.0))


    omega_min = E_bind
    omega_max = 10.0 * E_bind
    omega = np.linspace(omega_min, omega_max, n_points)
    domega = omega[1] - omega[0]



    Gamma = 0.5
    sigma_gamma = (16.0 * np.pi / (9.0 * HBAR_C)) * E_bind * Gamma / (
        (omega - E_bind) ** 2 + (Gamma / 2.0) ** 2)


    xi = omega * b_min / v
    xi = np.where(xi < 1e-6, 1e-6, xi)
    from scipy.special import kv
    n_gamma = (2.0 * FINE_STRUCTURE * Z_p ** 2 / (np.pi * omega)) * (
        xi * kv(0, xi) * kv(1, xi)
        - 0.5 * xi ** 2 * (kv(1, xi) ** 2 - kv(0, xi) ** 2))
    n_gamma = np.where(n_gamma < 0, 0.0, n_gamma)

    integrand = n_gamma * sigma_gamma
    return np.trapz(integrand, omega)


def angular_momentum_coupling_weight(j1, j2, J, M):
    if abs(M) > J or J < abs(j1 - j2) or J > j1 + j2:
        return 0.0
    m1_vals = np.arange(-j1, j1 + 0.5, 1.0)
    count = 0
    total = 0
    for m1 in m1_vals:
        m2 = M - m1
        if abs(m2) <= j2 + 1e-6:
            total += 1
            if abs(m1 + m2 - M) < 1e-6:
                count += 1
    return count / max(total, 1)
