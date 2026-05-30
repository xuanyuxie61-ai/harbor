
import numpy as np
from typing import Tuple
from icf_parameters import PC, EOS
from quadrature_rules import compute_fermi_dirac_integral


def electron_thermal_debroglie(T: float) -> float:
    if T <= 0.0:
        return 1.0e30
    return PC.PLANCK / np.sqrt(2.0 * np.pi * PC.ELECTRON_MASS * PC.BOLTZMANN * T)


def electron_number_density(rho, Z_eff, A_avg):
    A_avg = np.asarray(A_avg)
    result = np.zeros_like(np.asarray(rho), dtype=float)
    mask = A_avg > 0.0
    result[mask] = np.asarray(Z_eff)[mask] * np.asarray(rho)[mask] * PC.AVOGADRO / (A_avg[mask] * 1.0e-3)
    return result


def fermi_energy(n_e: float) -> float:
    if n_e <= 0.0:
        return 0.0
    prefactor = PC.PLANCK**2 / (2.0 * PC.ELECTRON_MASS)
    return prefactor * (3.0 * n_e / (8.0 * np.pi))**(2.0 / 3.0)


def degeneracy_parameter(n_e: float, T: float) -> float:
    ef = fermi_energy(n_e)
    if ef <= 1.0e-30 or T <= 0.0:
        return 1.0e30
    return PC.BOLTZMANN * T / ef


def electron_pressure_ideal(n_e: float, T: float) -> float:
    if n_e <= 0.0 or T <= 0.0:
        return 0.0
    return n_e * PC.BOLTZMANN * T


def electron_pressure_degenerate(n_e: float) -> float:
    if n_e <= 0.0:
        return 0.0
    ef = fermi_energy(n_e)
    return (2.0 / 5.0) * n_e * ef


def electron_pressure_full(n_e: float, T: float) -> float:
    p_ideal = electron_pressure_ideal(n_e, T)
    p_deg = electron_pressure_degenerate(n_e)
    return np.sqrt(p_deg**2 + p_ideal**2)


def electron_internal_energy(n_e: float, T: float) -> float:
    if n_e <= 0.0 or T <= 0.0:
        return 0.0

    theta = degeneracy_parameter(n_e, T)
    if theta > 10.0:

        return 1.5 * n_e * PC.BOLTZMANN * T
    elif theta < 0.1:

        ef = fermi_energy(n_e)
        return 0.6 * n_e * ef * (1.0 + 1.25 * theta**2)
    else:


        lambda_t = electron_thermal_debroglie(T)
        fugacity = n_e * lambda_t**3 / 2.0
        eta = np.log(max(fugacity, 1.0e-30))
        f32 = compute_fermi_dirac_integral(1, eta, n_quad=32)
        return (3.0 / 2.0) * n_e * PC.BOLTZMANN * T * (2.0 * f32
              / (3.0 * np.sqrt(np.pi) * max(fugacity, 1.0e-30)))


def coulomb_correction_pressure(n_e: float, Z_eff: float, T: float) -> float:
    if n_e <= 0.0 or T <= 0.0 or Z_eff <= 0.0:
        return 0.0


    a_ws = (3.0 / (4.0 * np.pi * n_e))**(1.0 / 3.0)
    coulomb_energy = PC.ELEMENTARY_CHARGE**2 / (4.0 * np.pi * PC.VACUUM_PERMITTIVITY * a_ws)
    gamma = coulomb_energy / (PC.BOLTZMANN * T)

    if gamma >= 1.0:

        gamma = min(gamma, 10.0)
        corr = -0.3 * gamma * n_e * PC.BOLTZMANN * T
    else:
        corr = -EOS.COULOMB_CORRECTION * gamma * n_e * PC.BOLTZMANN * T

    return corr


def radiation_pressure(T: float) -> float:
    if T <= 0.0:
        return 0.0
    return (4.0 / 3.0) * PC.STEFAN_BOLTZMANN * T**4 / PC.SPEED_OF_LIGHT


def radiation_energy_density(T: float) -> float:
    if T <= 0.0:
        return 0.0
    return PC.STEFAN_BOLTZMANN * T**4 / PC.SPEED_OF_LIGHT


def total_pressure(rho: float, T_e: float, T_i: float,
                   Z_eff: float, A_avg: float) -> float:
    if rho <= 0.0 or T_e < 0.0 or T_i < 0.0:
        return 0.0

    n_e = electron_number_density(rho, Z_eff, A_avg)
    n_i = n_e / max(Z_eff, 1.0e-10)

    p_ion = n_i * PC.BOLTZMANN * T_i
    p_e = electron_pressure_full(n_e, T_e)
    p_coul = coulomb_correction_pressure(n_e, Z_eff, T_e)
    p_rad = radiation_pressure(T_e)

    return max(p_ion + p_e + p_coul + p_rad, 1.0e-20)


def total_internal_energy(rho: float, T_e: float, T_i: float,
                          Z_eff: float, A_avg: float) -> float:
    if rho <= 0.0:
        return 0.0

    n_e = electron_number_density(rho, Z_eff, A_avg)
    n_i = n_e / max(Z_eff, 1.0e-10)

    eps_ion_vol = 1.5 * n_i * PC.BOLTZMANN * T_i
    eps_e_vol = electron_internal_energy(n_e, T_e)
    eps_rad_vol = radiation_energy_density(T_e)

    eps_total_vol = eps_ion_vol + eps_e_vol + eps_rad_vol
    return eps_total_vol / rho


def sound_speed(rho: float, T_e: float, T_i: float,
                Z_eff: float, A_avg: float) -> float:
    if rho <= 0.0:
        return 1.0e-10
    p = total_pressure(rho, T_e, T_i, Z_eff, A_avg)
    gamma_eff = EOS.GAMMA_IDEAL

    p_rad = radiation_pressure(T_e)
    if p_rad > 0.5 * p:
        gamma_eff = 4.0 / 3.0
    return np.sqrt(gamma_eff * p / rho)


def ionization_state_Saha(rho: float, T: float, Z_nuc: float,
                          ionization_energy: float) -> float:

    raise NotImplementedError("Saha ionization model not implemented")


def compute_eos_table(rho_vals: np.ndarray, T_vals: np.ndarray,
                      Z_nuc: float, A_avg: float,
                      ionization_energy: float) -> dict:
    nr, nt = len(rho_vals), len(T_vals)
    P_table = np.zeros((nr, nt))
    E_table = np.zeros((nr, nt))
    C_table = np.zeros((nr, nt))
    Z_table = np.zeros((nr, nt))

    for i, rho in enumerate(rho_vals):
        for j, T in enumerate(T_vals):
            Z_eff = ionization_state_Saha(rho, T, Z_nuc, ionization_energy)
            Z_table[i, j] = Z_eff
            P_table[i, j] = total_pressure(rho, T, T, Z_eff, A_avg)
            E_table[i, j] = total_internal_energy(rho, T, T, Z_eff, A_avg)
            C_table[i, j] = sound_speed(rho, T, T, Z_eff, A_avg)

    return {
        "pressure": P_table,
        "energy": E_table,
        "sound_speed": C_table,
        "ionization": Z_table,
        "rho": rho_vals,
        "T": T_vals,
    }
