# -*- coding: utf-8 -*-

import numpy as np


def poly_eval(coeffs, z):
    coeffs = np.asarray(coeffs, dtype=complex)
    z = np.asarray(z, dtype=complex)
    result = np.zeros_like(z, dtype=complex)

    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-12, max_iter=1000):
    coeffs = np.asarray(coeffs, dtype=complex)
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("多项式次数必须 >= 1。")
    if abs(coeffs[-1]) < 1e-30:
        raise ValueError("首项系数不能为零。")


    coeffs = coeffs / coeffs[-1]


    R = 1.0 + np.max(np.abs(coeffs[:-1]))


    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    roots = R * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()
        for i in range(n):
            zi = roots_old[i]
            denom = 1.0 + 0.0j
            for j in range(n):
                if i != j:
                    diff = zi - roots[j]
                    if abs(diff) < 1e-30:
                        diff = 1e-30 * (1.0 + 1.0j)
                    denom *= diff
            pz = poly_eval(coeffs, zi)
            roots[i] = zi - pz / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            return roots, True

    return roots, False


def bohm_gross_polynomial(ne, Te, k, omega0):
    raise NotImplementedError("Hole 2: 请实现 bohm_gross_polynomial 函数体")


def solve_langmuir_wave_dispersion(ne, Te, k, omega0):
    coeffs = bohm_gross_polynomial(ne, Te, k, omega0)
    roots, converged = wdk_roots(coeffs, tol=1e-14, max_iter=2000)

    if not converged:

        from physics_constants import plasma_frequency, E_MASS, K_BOLTZMANN
        omega_p = plasma_frequency(ne)
        v_te = np.sqrt(K_BOLTZMANN * Te / E_MASS)
        omega_r = np.sqrt(omega_p**2 + 3.0 * k**2 * v_te**2)

        k_lambda = k * np.sqrt(K_BOLTZMANN * Te / (ne * (1.602176634e-19)**2 / (8.8541878128e-12 * E_MASS)))
        if k_lambda > 0:
            gamma = -np.sqrt(np.pi / 8.0) * (omega_p / (k_lambda**3)) * np.exp(-1.0 / (2.0 * k_lambda**2) - 1.5)
        else:
            gamma = 0.0
        return omega_r, gamma, omega_r + 1j * gamma, roots



    from physics_constants import plasma_frequency
    omega_p = plasma_frequency(ne)
    best_idx = -1
    best_score = -np.inf
    for idx, r in enumerate(roots):
        if abs(r) < 1e-10:
            continue

        score = -abs(abs(r.real) - omega_p) / max(omega_p, 1.0)

        if r.real > 0:
            score += 0.1
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        best_idx = np.argmax(np.abs(roots))

    root_selected = roots[best_idx]
    omega_r = abs(root_selected.real)
    gamma = root_selected.imag

    return omega_r, gamma, root_selected, roots


def srs_three_wave_coupling_roots(ne, Te, k_s, omega0, E0):
    from physics_constants import (plasma_frequency, C_LIGHT, E_MASS,
                                    K_BOLTZMANN, quiver_velocity, srs_growth_rate)

    omega_p = plasma_frequency(ne)
    v_te = np.sqrt(K_BOLTZMANN * Te / E_MASS)
    v_osc = quiver_velocity(E0, omega0)
    gamma_0 = srs_growth_rate(ne, E0, omega0)


    k_0 = omega0 / C_LIGHT
    k_p = k_0 - k_s


    A = omega_p**2 + 3.0 * k_s**2 * v_te**2
    B = omega_p**2 + C_LIGHT**2 * k_p**2






    c4 = 1.0 + 0.0j
    c3 = -2.0 * omega0 + 0.0j
    c2 = (omega0**2 - B - A) + 0.0j
    c1 = 2.0 * omega0 * A + 0.0j
    c0 = -A * (omega0**2 - B) - gamma_0**4 + 0.0j

    coeffs = np.array([c0, c1, c2, c3, c4], dtype=complex)
    roots, converged = wdk_roots(coeffs, tol=1e-12, max_iter=2000)


    target = omega0 - omega_p
    best_idx = -1
    best_score = -np.inf
    for idx, r in enumerate(roots):
        score = -abs(abs(r.real) - target) / max(target, 1.0) - abs(r.imag) / max(omega_p, 1.0)
        if r.real > 0:
            score += 0.05
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        best_idx = 0

    root_sel = roots[best_idx]
    omega_s_r = abs(root_sel.real)
    gamma_srs = root_sel.imag

    return omega_s_r, gamma_srs, roots


def plasma_dispersion_function_derivative(zeta, n_terms=50):
    zeta = complex(zeta)
    if abs(zeta) < 0.1:

        Zp = -2.0 + 0.0j
        term = 1.0
        for n in range(1, n_terms):
            term *= (-2.0 * zeta) / (2.0 * n + 1.0)
            Zp += term
        Zp = Zp / np.sqrt(np.pi)
        Zp += 1j * np.sqrt(np.pi) * zeta * np.exp(-zeta**2)
    else:

        Zp = 0.0 + 0.0j
        for n in range(1, n_terms + 1):
            coeff = 1.0
            for m in range(n):
                coeff *= (2.0 * m + 1.0) / 2.0
            Zp += coeff / (zeta ** (2.0 * n))
        Zp = -Zp
    return Zp
