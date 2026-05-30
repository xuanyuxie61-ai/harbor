# -*- coding: utf-8 -*-

import numpy as np


def sphere_distance1(lat1, lon1, lat2, lon2, r):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2

    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def ll_to_xyz(r, n, lat, lon):
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_ll(r, xyz):
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]


    rho = np.sqrt(x ** 2 + y ** 2)
    rho = np.where(rho < 1e-15, 1e-15, rho)

    lat = np.arctan2(z, rho)
    lon = np.arctan2(y, x)
    return lat, lon


def spherical_harmonic(l, m, theta, phi):
    from scipy.special import sph_harm


    ylm_complex = sph_harm(abs(m), l, phi, theta)
    if m >= 0:
        return np.real(ylm_complex) if m == 0 else np.real(ylm_complex) * np.sqrt(2.0)
    else:
        return np.imag(ylm_complex) * np.sqrt(2.0)


def spherical_harmonic_manual(l, m, theta, phi):
    x = np.cos(theta)


    def associated_legendre(l_val, m_val, x_val):

        m_abs = abs(m_val)

        if m_abs > l_val:
            return np.zeros_like(x_val)


        p_mm = np.ones_like(x_val)
        if m_abs > 0:
            somx2 = np.sqrt(np.maximum(0.0, 1.0 - x_val ** 2))
            fact = 1.0
            for i in range(1, m_abs + 1):
                p_mm *= -fact * somx2
                fact += 2.0

        if l_val == m_abs:
            return p_mm

        p_mmp1 = x_val * (2 * m_abs + 1) * p_mm
        if l_val == m_abs + 1:
            return p_mmp1

        for ll in range(m_abs + 2, l_val + 1):
            p_ll = (x_val * (2 * ll - 1) * p_mmp1 - (ll + m_abs - 1) * p_mm) / (ll - m_abs)
            p_mm = p_mmp1
            p_mmp1 = p_ll

        return p_mmp1

    p_lm = associated_legendre(l, m, x)


    from math import factorial, sqrt, pi
    nlm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))

    if m > 0:
        return nlm * p_lm * np.cos(m * phi) * sqrt(2.0)
    elif m < 0:
        return nlm * p_lm * np.sin(abs(m) * phi) * sqrt(2.0)
    else:
        return nlm * p_lm


def generate_turbulent_initial_field(nx, ny, nz, max_l=8, seed=42):
    rng = np.random.default_rng(seed)


    theta = np.linspace(0, np.pi, nx)
    phi = np.linspace(0, 2 * np.pi, ny)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    u_surf = np.zeros((nx, ny), dtype=float)
    v_surf = np.zeros((nx, ny), dtype=float)


    for l in range(1, max_l + 1):
        k = float(l)
        energy = k ** 4 * np.exp(-2.0 * (k / 4.0) ** 2)
        amplitude = np.sqrt(energy) * rng.standard_normal()

        for m in range(-l, l + 1):
            coeff = amplitude * rng.standard_normal()
            try:
                ylm = spherical_harmonic_manual(l, m, Theta, Phi)
                u_surf += coeff * ylm
                v_surf += coeff * ylm * 0.5
            except Exception:
                pass


    u = np.repeat(u_surf[:, :, np.newaxis], nz, axis=2)
    v = np.repeat(v_surf[:, :, np.newaxis], nz, axis=2)
    w = rng.standard_normal(size=(nx, ny, nz)) * 0.1


    u = u - np.mean(u)
    v = v - np.mean(v)
    w = w - np.mean(w)

    return u, v, w


def spherical_boundary_condition(u, v, w, r, radius, bc_type='no_slip'):
    if bc_type == 'no_slip':
        mask = np.abs(r - radius) < 1e-6
        u = np.where(mask, 0.0, u)
        v = np.where(mask, 0.0, v)
        w = np.where(mask, 0.0, w)
    elif bc_type == 'free_slip':

        mask = np.abs(r - radius) < 1e-6

        x = np.linspace(-1, 1, u.shape[0])
        y = np.linspace(-1, 1, u.shape[1])
        z = np.linspace(-1, 1, u.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


        nx = X / (r + 1e-15)
        ny = Y / (r + 1e-15)
        nz = Z / (r + 1e-15)


        un = u * nx + v * ny + w * nz


        u = np.where(mask, u - un * nx, u)
        v = np.where(mask, v - un * ny, v)
        w = np.where(mask, w - un * nz, w)

    return u, v, w
