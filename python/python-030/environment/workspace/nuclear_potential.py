# -*- coding: utf-8 -*-

import numpy as np
from constants import (
    WS_V0, WS_R0, WS_A, WS_VSO, WS_RSO, WS_ASO,
    HBAR_C, FINE_STRUCTURE
)


def spherical_harmonic_y20(theta):
    return np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * np.cos(theta) ** 2 - 1.0)


def spherical_harmonic_y30(theta):
    c = np.cos(theta)
    return np.sqrt(7.0 / (16.0 * np.pi)) * (5.0 * c ** 3 - 3.0 * c)


def spherical_harmonic_y40(theta):
    c = np.cos(theta)
    return np.sqrt(9.0 / (256.0 * np.pi)) * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0)


def deformed_radius(theta, A, beta2=0.0, beta3=0.0, beta4=0.0):
    R0 = WS_R0 * (A ** (1.0 / 3.0))
    y20 = spherical_harmonic_y20(theta)
    y30 = spherical_harmonic_y30(theta)
    y40 = spherical_harmonic_y40(theta)
    return R0 * (1.0 + beta2 * y20 + beta3 * y30 + beta4 * y40)


def woods_saxon(r, V0, R, a):
    r = np.asarray(r, dtype=float)

    arg = (r - R) / a

    arg = np.clip(arg, -500.0, 500.0)
    return V0 / (1.0 + np.exp(arg))


def woods_saxon_derivative(r, V0, R, a):
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    arg = np.clip(arg, -500.0, 500.0)
    e = np.exp(arg)
    return -(V0 / a) * e / ((1.0 + e) ** 2)


def spin_orbit_potential(r, l, s, A, Vso0=None, Rso=None, aso=None):
    if Vso0 is None:
        Vso0 = WS_VSO
    if Rso is None:
        Rso = WS_RSO * (A ** (1.0 / 3.0))
    if aso is None:
        aso = WS_ASO

    r = np.asarray(r, dtype=float)
    ls = 0.5 * (l * (l + 1.0))


    dfd = woods_saxon_derivative(r, 1.0, Rso, aso)

    r_safe = np.where(r < 0.2, 1.0, r)
    Vso = Vso0 * (1.0 / r_safe) * dfd * ls
    Vso = np.where(r < 0.2, 0.0, Vso)
    return Vso


def coulomb_potential(r, Z, A):
    r = np.asarray(r, dtype=float)
    Rc = 1.2 * (A ** (1.0 / 3.0))
    e2 = 1.439964
    Vc = np.empty_like(r)
    inside = r <= Rc
    outside = ~inside
    Vc[inside] = (Z * e2 / (2.0 * Rc)) * (3.0 - (r[inside] / Rc) ** 2)
    Vc[outside] = Z * e2 / r[outside]
    return Vc


def build_mean_field_potential(r, Z, N, beta2=0.0, beta3=0.0, beta4=0.0,
                               return_components=False):
    A = Z + N
    R0 = WS_R0 * (A ** (1.0 / 3.0))

    delta_def = 0.05 * beta2 ** 2 + 0.02 * beta3 ** 2 + 0.01 * beta4 ** 2
    Req = R0 * (1.0 - delta_def)

    V_central = woods_saxon(r, WS_V0, Req, WS_A)
    V_coul = coulomb_potential(r, Z, A)


    l_repr = 2
    V_so = spin_orbit_potential(r, l_repr, 0.5, A)

    V_total = V_central + V_so + V_coul

    if return_components:
        return V_total, {
            'central': V_central,
            'spin_orbit': V_so,
            'coulomb': V_coul
        }
    return V_total


def build_neutron_potential(r, N, Z, beta2=0.0, beta3=0.0, beta4=0.0):











    raise NotImplementedError("HOLE 2: build_neutron_potential is not implemented.")


def build_proton_potential(r, Z, N, beta2=0.0, beta3=0.0, beta4=0.0):
    A = Z + N
    R0 = WS_R0 * (A ** (1.0 / 3.0))
    delta_def = 0.05 * beta2 ** 2 + 0.02 * beta3 ** 2 + 0.01 * beta4 ** 2
    Req = R0 * (1.0 - delta_def)
    V_central = woods_saxon(r, WS_V0, Req, WS_A)
    V_so = spin_orbit_potential(r, 2, 0.5, A)
    V_coul = coulomb_potential(r, Z, A)
    return V_central + V_so + V_coul
