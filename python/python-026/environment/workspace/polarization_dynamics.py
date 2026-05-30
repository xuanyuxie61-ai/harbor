# -*- coding: utf-8 -*-

import numpy as np


def jones_matrix_propagation(omega, omega_p, omega_c, b_hat, dz):
    from physics_constants import C_LIGHT

    if omega <= 0 or dz < 0:
        raise ValueError("omega 必须为正，dz 必须非负。")



    bz = float(b_hat[2]) if b_hat is not None else 1.0
    bz = np.clip(bz, -1.0, 1.0)


    phi = (omega_p**2 * omega_c * bz) / (2.0 * C_LIGHT * omega**3) * dz


    delta = (omega_p**2 * omega_c**2) / (4.0 * C_LIGHT * omega**4) * dz


    T = np.array([
        [np.cos(phi) + 1j * np.sin(phi) * np.cos(delta), -np.sin(phi) + 1j * np.sin(phi) * np.sin(delta)],
        [np.sin(phi) - 1j * np.sin(phi) * np.sin(delta), np.cos(phi) - 1j * np.sin(phi) * np.cos(delta)]
    ], dtype=complex)

    return T


def apply_jones_matrix(T, E_in):
    E_in = np.asarray(E_in, dtype=complex)
    if E_in.shape != (2,):
        raise ValueError("E_in 必须是二维复向量。")
    E_out = T @ E_in
    return E_out


def polarization_ellipse_parameters(E):
    E = np.asarray(E, dtype=complex)
    if E.shape != (2,):
        raise ValueError("E 必须是二维复向量。")

    Ex, Ey = E[0], E[1]
    S0 = abs(Ex)**2 + abs(Ey)**2
    if S0 < 1e-30:
        return 0.0, 0.0, 0.0, np.array([0.0, 0.0, 0.0, 0.0])

    S1 = abs(Ex)**2 - abs(Ey)**2
    S2 = 2.0 * (Ex.real * Ey.real + Ex.imag * Ey.imag)
    S3 = 2.0 * (Ex.real * Ey.imag - Ex.imag * Ey.real)
    S = np.array([S0, S1, S2, S3])

    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3 / S0, -1.0, 1.0))
    epsilon = np.tan(abs(chi))
    epsilon = min(epsilon, 1.0)

    return psi, chi, epsilon, S


def faraday_rotation_integral(ne_profile, B_parallel, z_vals, omega):
    from physics_constants import E_CHARGE, E_MASS, EPSILON_0, C_LIGHT

    if len(z_vals) < 2 or len(ne_profile) != len(z_vals) or len(B_parallel) != len(z_vals):
        raise ValueError("输入数组长度不一致。")

    prefactor = E_CHARGE**3 / (2.0 * EPSILON_0 * E_MASS**2 * C_LIGHT * omega**2)
    integrand = ne_profile * B_parallel
    theta_F = prefactor * np.trapezoid(integrand, z_vals)
    return float(theta_F)


def circle_map_matrix_polarization(A, n_points=200):
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A 必须是 2x2 矩阵。")

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    x_in = np.array([np.cos(theta), np.sin(theta)])
    x_out = A @ x_in


    U, s, Vt = np.linalg.svd(A)
    aspect_ratio = s[0] / max(s[1], 1e-30)

    return x_in, x_out, aspect_ratio


def stokes_to_poincare_sphere(S):
    S = np.asarray(S, dtype=float)
    S0 = S[0]
    if S0 < 1e-30:
        return np.array([1.0, 0.0, 0.0])
    XYZ = S[1:4] / S0

    norm = np.linalg.norm(XYZ)
    if norm > 1e-10:
        XYZ = XYZ / norm
    return XYZ


def evolve_polarization_along_ray(omega, ne_func, B_func, z_vals, E0):
    from physics_constants import plasma_frequency, E_CHARGE, E_MASS

    z_vals = np.asarray(z_vals, dtype=float)
    N = len(z_vals)
    E_history = np.zeros((N, 2), dtype=complex)
    stokes_history = np.zeros((N, 4), dtype=float)

    E = np.asarray(E0, dtype=complex)
    E_history[0, :] = E
    _, _, _, S = polarization_ellipse_parameters(E)
    stokes_history[0, :] = S

    for i in range(1, N):
        dz = z_vals[i] - z_vals[i - 1]
        if dz <= 0:
            E_history[i, :] = E
            stokes_history[i, :] = S
            continue

        z_mid = 0.5 * (z_vals[i] + z_vals[i - 1])
        ne = ne_func(z_mid)
        B = B_func(z_mid)
        omega_p = plasma_frequency(ne)
        B_mag = np.linalg.norm(B)
        omega_c = E_CHARGE * B_mag / E_MASS if B_mag > 0 else 0.0
        b_hat = B / B_mag if B_mag > 0 else np.array([0.0, 0.0, 1.0])

        T = jones_matrix_propagation(omega, omega_p, omega_c, b_hat, dz)
        E = apply_jones_matrix(T, E)
        E_history[i, :] = E
        _, _, _, S = polarization_ellipse_parameters(E)
        stokes_history[i, :] = S

    return E_history, stokes_history
