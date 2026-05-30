# -*- coding: utf-8 -*-

import numpy as np


def rk4_integrate(f, t_span, y0, n_steps):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=complex)
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    y = np.zeros((n_steps + 1, y0.shape[0]), dtype=complex)
    y[0] = y0

    for i in range(n_steps):
        ti = t[i]
        ui = y[i]
        k1 = f(ti, ui)
        k2 = f(ti + dt / 2.0, ui + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, ui + dt * k2 / 2.0)
        k4 = f(ti + dt, ui + dt * k3)
        y[i + 1] = ui + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, y


def rk23_integrate(f, t_span, y0, n_steps):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=complex)
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    m = y0.shape[0]
    y = np.zeros((n_steps + 1, m), dtype=complex)
    e = np.zeros((n_steps + 1, m), dtype=float)
    y[0] = y0
    e[0] = 0.0

    for i in range(n_steps):
        k1 = dt * f(t[i], y[i])
        k2 = dt * f(t[i] + dt, y[i] + k1)
        k3 = dt * f(t[i] + 0.5 * dt, y[i] + 0.25 * k1 + 0.25 * k2)
        y2 = y[i] + 0.5 * (k1 + k2)
        y3 = y[i] + (k1 + k2 + 4.0 * k3) / 6.0
        y[i + 1] = y3
        e[i + 1] = np.abs(y3 - y2)
    return t, y, e


def propagate_plane_wave_scalar(k0, z_span, n_eff_func, E0, n_steps=200):
    E0 = complex(E0)

    def f(z, E):
        nz = n_eff_func(z)

        if np.isreal(nz):
            nz = complex(max(np.real(nz), 1.0), min(np.imag(nz), 0.0))
        else:
            nz = complex(max(np.real(nz), 1.0), min(np.imag(nz), 0.0))
        return np.array([1j * k0 * nz * E[0]], dtype=complex)

    t, y = rk4_integrate(f, z_span, np.array([E0], dtype=complex), n_steps)
    return t, y[:, 0]


def propagate_coupled_modes(k0, z_span, n_matrix_func, E0_vec, n_steps=200):
    E0_vec = np.asarray(E0_vec, dtype=complex)
    if E0_vec.shape[0] != 2:
        raise ValueError("耦合模式仅支持 2 模式")

    def f(z, E):
        N = n_matrix_func(z)
        N = np.asarray(N, dtype=complex)

        for i in range(2):
            N[i, i] = complex(max(np.real(N[i, i]), 1.0),
                              min(np.imag(N[i, i]), 0.0))
        return 1j * k0 * (N @ E)

    t, y = rk4_integrate(f, z_span, E0_vec, n_steps)
    return t, y


def angular_spectrum_propagate(field, k0, z, dx, dy):
    field = np.asarray(field, dtype=complex)
    ny, nx = field.shape


    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)


    kz2 = k0 ** 2 - KX ** 2 - KY ** 2

    kz = np.zeros_like(kz2, dtype=complex)
    propagating = kz2 >= 0
    kz[propagating] = np.sqrt(kz2[propagating])
    evanescent = kz2 < 0
    kz[evanescent] = 1j * np.sqrt(-kz2[evanescent])


    spectrum = np.fft.fft2(field)
    transfer = np.exp(1j * kz * z)
    propagated_spectrum = spectrum * transfer
    propagated_field = np.fft.ifft2(propagated_spectrum)
    return propagated_field


def effective_medium_profile(z, n_substrate, n_air, thickness,
                             profile_type='linear'):
    if profile_type == 'linear':
        return n_air + (n_substrate - n_air) * (z / thickness)
    elif profile_type == 'quadratic':
        t = z / thickness
        return n_air + (n_substrate - n_air) * (t ** 2)
    elif profile_type == 'exponential':
        return n_air + (n_substrate - n_air) * (1.0 - np.exp(-3.0 * z / thickness))
    else:

        return n_air + (n_substrate - n_air) * (z / thickness)
