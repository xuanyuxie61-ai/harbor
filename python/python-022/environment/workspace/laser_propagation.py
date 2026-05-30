
import numpy as np
from typing import Tuple
from icf_parameters import LP, PC
from utils import safe_divide, clamp_array


def critical_density(wavelength: float) -> float:
    omega = 2.0 * np.pi * PC.SPEED_OF_LIGHT / wavelength
    nc = PC.VACUUM_PERMITTIVITY * PC.ELECTRON_MASS * omega**2 / PC.ELEMENTARY_CHARGE**2
    return nc


def plasma_refractive_index(n_e: float, n_c: float) -> float:
    ratio = n_e / max(n_c, 1.0e-10)
    if ratio >= 1.0:
        return 0.0
    return np.sqrt(max(1.0 - ratio, 0.0))


def electron_ion_collision_freq(n_e: float, T_e: float, Z_eff: float) -> float:
    if T_e <= 0.0 or n_e <= 0.0:
        return 0.0
    ln_lambda = max(23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0)), 2.0)
    nu = 2.9e-12 * Z_eff * n_e * ln_lambda / (T_e**1.5)
    return max(nu, 0.0)


def inverse_bremsstrahlung_coeff(n_e: float, T_e: float, Z_eff: float,
                                  wavelength: float) -> float:
    nc = critical_density(wavelength)
    if nc <= 1.0e-10 or n_e <= 0.0:
        return 0.0

    nu_ei = electron_ion_collision_freq(n_e, T_e, Z_eff)
    ratio = n_e / nc
    kappa = ratio**2 * Z_eff * nu_ei / PC.SPEED_OF_LIGHT
    return max(kappa, 0.0)


def nlse_envelope_discrete(E: np.ndarray, z: float, dz: float,
                           n_e_profile: np.ndarray, n_c: float,
                           dr: float) -> np.ndarray:
    n = len(E)
    dEdz = np.zeros(n, dtype=complex)
    k0 = 2.0 * np.pi / LP.WAVELENGTH

    for j in range(n):
        r_j = (j + 0.5) * dr
        ne = n_e_profile[j]
        n_index = plasma_refractive_index(ne, n_c)
        k = k0 * n_index


        if j == 0:
            laplace = (E[j + 1] - 2.0 * E[j] + E[j]) / dr**2
        elif j == n - 1:
            laplace = (E[j] - 2.0 * E[j] + E[j - 1]) / dr**2
        else:
            second_deriv = (E[j + 1] - 2.0 * E[j] + E[j - 1]) / dr**2
            first_deriv = (E[j + 1] - E[j - 1]) / (2.0 * dr)
            laplace = second_deriv + first_deriv / max(r_j, 1.0e-15)

        dEdz[j] = 1j / (2.0 * k0) * laplace + 1j * (k - k0) * E[j]


    return E + dz * dEdz


def compute_laser_deposition_1d(r_cells: np.ndarray, r_nodes: np.ndarray,
                                rho: np.ndarray, T_e: np.ndarray,
                                Z_eff: np.ndarray,
                                beam_power: float,
                                total_time: float,
                                n_samples: int = 101) -> np.ndarray:
    n_cells = len(r_cells)
    deposition = np.zeros(n_cells)

    if beam_power <= 0.0:
        return deposition

    nc = critical_density(LP.WAVELENGTH)


    n_e = np.zeros(n_cells)
    for i in range(n_cells):
        n_e[i] = Z_eff[i] * rho[i] * PC.AVOGADRO / (2.5 * 1.0e-3)


    critical_surface_idx = -1
    for i in range(n_cells - 1, -1, -1):
        if n_e[i] < nc:
            critical_surface_idx = i
            break

    if critical_surface_idx < 0:
        return deposition


    absorption_efficiency = 0.8
    P_incident = beam_power * absorption_efficiency


    P_current = P_incident

    for i in range(n_cells - 1, critical_surface_idx - 1, -1):
        dr = r_nodes[i + 1] - r_nodes[i]
        kappa = inverse_bremsstrahlung_coeff(n_e[i], T_e[i], Z_eff[i], LP.WAVELENGTH)


        attenuation = np.exp(-kappa * dr)
        P_out = P_current * attenuation
        P_absorbed = P_current - P_out


        vol = 4.0 * np.pi / 3.0 * (r_nodes[i + 1]**3 - r_nodes[i]**3)
        if vol > 1.0e-30:
            deposition[i] = P_absorbed / vol

        P_current = P_out
        if P_current < 1.0e-6 * P_incident:
            break

    return deposition


def laser_power_time(t: float) -> float:
    return LP.power_profile(t)
