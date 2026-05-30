# -*- coding: utf-8 -*-

import numpy as np


E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837015e-31
EPSILON_0 = 8.8541878128e-12
MU_0 = 4.0 * np.pi * 1.0e-7
C_LIGHT = 299792458.0
K_BOLTZMANN = 1.380649e-23
PLANCK_H = 6.62607015e-34


def plasma_frequency(ne):
    raise NotImplementedError("Hole 1: 请实现 plasma_frequency 函数体")


def debye_length(ne, Te):
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    if np.any(ne <= 0):
        raise ValueError("电子密度 ne 必须为正数。")
    if np.any(Te <= 0):
        raise ValueError("电子温度 Te 必须为正数。")
    lambda_d = np.sqrt(EPSILON_0 * K_BOLTZMANN * Te / (ne * E_CHARGE**2))
    return lambda_d


def critical_density(omega0):
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    nc = EPSILON_0 * E_MASS * omega0**2 / E_CHARGE**2
    return nc


def refractive_index(ne, omega0):
    ne = np.asarray(ne, dtype=float)
    omega_p = plasma_frequency(ne)
    ratio = (omega_p / omega0) ** 2
    ratio = np.clip(ratio, 0.0, 1.0)
    eta = np.sqrt(1.0 - ratio)
    return eta


def quiver_velocity(E0, omega0):
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    v_osc = E_CHARGE * E0 / (E_MASS * omega0)
    return v_osc


def ponderomotive_potential(E0, omega0):
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    Up = E_CHARGE**2 * E0**2 / (4.0 * E_MASS * omega0**2)
    return Up


def ponderomotive_force_gradient(E0, omega0, grad_E2):
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    coeff = -E_CHARGE**2 / (4.0 * E_MASS * omega0**2)
    Fp = coeff * np.asarray(grad_E2, dtype=float)
    return Fp


def srs_growth_rate(ne, E0, omega0):
    omega_p = plasma_frequency(ne)
    v_osc = quiver_velocity(E0, omega0)
    gamma0 = (v_osc / (2.0 * C_LIGHT)) * np.sqrt(omega_p * omega0)
    return gamma0


def landau_damping_rate(ne, Te, k):
    omega_p = plasma_frequency(ne)
    lambda_d = debye_length(ne, Te)
    if k <= 0 or lambda_d <= 0:
        raise ValueError("k 和 λ_D 必须为正。")
    k_lambda = k * lambda_d
    gamma_L = np.sqrt(np.pi / 8.0) * (omega_p / (k**3 * lambda_d**3)) * \
              np.exp(-1.0 / (2.0 * k_lambda**2) - 1.5)
    return gamma_L


def coulomb_logarithm(ne, Te, Z=1):
    Te_eV = Te / E_CHARGE
    if ne <= 0 or Te_eV <= 0:
        raise ValueError("ne 和 Te 必须为正。")
    ln_lambda = 23.5 - np.log(np.sqrt(ne) * Z / Te_eV**1.5)
    if ln_lambda < 1.0:
        ln_lambda = 1.0
    return ln_lambda


def electron_ion_collision_frequency(ne, Te, Z=1):
    ln_lambda = coulomb_logarithm(ne, Te, Z)
    nu_ei = (Z * ne * E_CHARGE**4 * ln_lambda) / \
            (3.0 * (2.0 * np.pi)**1.5 * EPSILON_0**2 * np.sqrt(E_MASS) * (K_BOLTZMANN * Te)**1.5)
    return nu_ei


def inverse_bremsstrahlung_absorption(ne, Te, omega0, Z=1):
    nu_ei = electron_ion_collision_frequency(ne, Te, Z)
    omega_p = plasma_frequency(ne)
    eta = refractive_index(ne, omega0)
    eta_safe = max(eta, 1e-6)
    kappa_ib = (nu_ei / C_LIGHT) * (omega_p / omega0)**2 * (1.0 / eta_safe)
    if np.isinf(kappa_ib) or np.isnan(kappa_ib):
        kappa_ib = 0.0
    return kappa_ib


def laser_wavenumber(ne, omega0):
    eta = refractive_index(ne, omega0)
    k = (omega0 / C_LIGHT) * eta
    return k


def laser_intensity_from_E0(E0):
    I = 0.5 * C_LIGHT * EPSILON_0 * E0**2
    return I


def laser_E0_from_intensity(I):
    if I < 0:
        raise ValueError("激光强度必须为非负数。")
    E0 = np.sqrt(2.0 * I / (C_LIGHT * EPSILON_0))
    return E0
