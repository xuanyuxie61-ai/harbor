
import numpy as np
from constants import (
    GF, KM_TO_EV_INV,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH,
    THETA_12, THETA_23, THETA_13
)
from pmns_matrix import build_pmns_matrix, build_mass_matrix


def log_gamma_pike_hill(x):
    if x <= 0.0:
        return 0.0, 1

    y = float(x)
    if x < 7.0:
        f = 1.0
        z = y
        while z < 7.0:
            f *= z
            z += 1.0
        y = z
        f = -np.log(f)
    else:
        f = 0.0

    z = 1.0 / y / y
    value = f + (y - 0.5) * np.log(y) - y + 0.918938533204673

    value += (((
        -0.000595238095238 * z
        + 0.000793650793651) * z
        - 0.002777777777778) * z
        + 0.083333333333333) / y

    return value, 0


def fermi_dirac_distribution(energy, temperature, chemical_potential=0.0, eta=-1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    arg = (energy - chemical_potential) / temperature


    if eta == -1:
        if arg > 700:
            return 0.0
        return 1.0 / (np.exp(arg) + 1.0)
    elif eta == 1:
        if arg > 700:
            return 0.0
        if arg < -700:
            return 0.0
        return 1.0 / (np.exp(arg) - 1.0)
    else:
        return 1.0 / (np.exp(arg) + eta)


def build_vacuum_hamiltonian(energy_gev, theta12=None, theta23=None,
                             theta13=None, delta_cp=None,
                             delta_m2_21=None, delta_m2_31=None,
                             hierarchy='normal'):
    if energy_gev <= 0:
        raise ValueError("energy_gev must be positive")









    raise NotImplementedError("HOLE 2: build_vacuum_hamiltonian 核心计算尚未实现")


def build_matter_hamiltonian(energy_gev, matter_potential_ev,
                             theta12=None, theta23=None,
                             theta13=None, delta_cp=None,
                             delta_m2_21=None, delta_m2_31=None,
                             hierarchy='normal'):
    H_vac = build_vacuum_hamiltonian(
        energy_gev, theta12, theta23, theta13, delta_cp,
        delta_m2_21, delta_m2_31, hierarchy
    )

    V = float(matter_potential_ev)
    V_mat = np.diag([V, 0.0, 0.0])

    H_mat = H_vac + V_mat
    return H_mat


def solve_hamiltonian_eigen(H):

    eigenvalues, eigenvectors = np.linalg.eigh(H)



    U_matter = eigenvectors

    return eigenvalues, eigenvectors, U_matter


def effective_mixing_angles_in_matter(energy_gev, matter_potential_ev,
                                       theta12=None, theta23=None,
                                       theta13=None, delta_cp=None,
                                       delta_m2_21=None, delta_m2_31=None,
                                       hierarchy='normal'):
    from constants import THETA_12, THETA_23, THETA_13

    t12 = THETA_12 if theta12 is None else theta12
    t23 = THETA_23 if theta23 is None else theta23
    t13 = THETA_13 if theta13 is None else theta13
    dm21 = DELTA_M2_21 if delta_m2_21 is None else delta_m2_21

    E_eV = energy_gev * 1e9
    V = matter_potential_ev



    A = 2.0 * E_eV * V
    cos2t12 = np.cos(2.0 * t12)
    sin2t12 = np.sin(2.0 * t12)

    denom12 = (cos2t12 - A / dm21) ** 2 + sin2t12 ** 2
    sin2_2theta12_m = sin2t12 ** 2 / denom12


    sin2_2theta12_m = max(0.0, min(1.0, sin2_2theta12_m))
    theta12_m = 0.5 * np.arcsin(np.sqrt(sin2_2theta12_m))


    theta23_m = t23


    cos2t13 = np.cos(2.0 * t13)
    sin2t13 = np.sin(2.0 * t13)

    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH
    dm_ee = np.abs(dm31 - (np.sin(t12) ** 2) * dm21)
    denom13 = (cos2t13 - A / dm_ee) ** 2 + sin2t13 ** 2
    sin2_2theta13_m = sin2t13 ** 2 / denom13
    sin2_2theta13_m = max(0.0, min(1.0, sin2_2theta13_m))
    theta13_m = 0.5 * np.arcsin(np.sqrt(sin2_2theta13_m))

    return {
        'theta12_m': float(theta12_m),
        'theta23_m': float(theta23_m),
        'theta13_m': float(theta13_m)
    }


def msw_resonance_density(energy_gev, theta=None, delta_m2=None):
    from constants import THETA_12, DELTA_M2_21

    t = THETA_12 if theta is None else theta
    dm2 = DELTA_M2_21 if delta_m2 is None else delta_m2
    E_eV = energy_gev * 1e9

    cos2t = np.cos(2.0 * t)
    numerator = dm2 * cos2t
    denominator = 2.0 * np.sqrt(2.0) * GF * E_eV



    GF_eV2 = GF * 1e18
    denominator = 2.0 * np.sqrt(2.0) * GF_eV2 * E_eV

    ne_res_eV3 = numerator / denominator


    hbarc_cm = 1.973269804e-5
    ne_res_cm3 = ne_res_eV3 * (hbarc_cm ** 3)

    return float(ne_res_cm3)


def hierarchy_discrimination_significance(delta_m2_31, sigma_dm31=0.03e-3):
    if sigma_dm31 <= 0:
        raise ValueError("sigma_dm31 must be positive")

    significance = abs(delta_m2_31) / sigma_dm31

    if delta_m2_31 > 0:
        hierarchy = 'normal'
    elif delta_m2_31 < 0:
        hierarchy = 'inverted'
    else:
        hierarchy = 'undetermined'

    return float(significance), hierarchy


def compute_oscillation_wavelengths(energy_gev, delta_m2_21=None, delta_m2_31=None):
    dm21 = DELTA_M2_21 if delta_m2_21 is None else delta_m2_21
    dm31 = DELTA_M2_31 if delta_m2_31 is None else delta_m2_31
    dm32 = dm31 - dm21


    factor = 2.48
    return {
        'L_21': factor * energy_gev / abs(dm21),
        'L_31': factor * energy_gev / abs(dm31),
        'L_32': factor * energy_gev / abs(dm32)
    }


def mass_sum_bounds(hierarchy='normal', m_lightest_eV=0.0):
    if m_lightest_eV < 0:
        raise ValueError("m_lightest_eV must be non-negative")

    dm21 = DELTA_M2_21
    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH

    if hierarchy == 'normal':
        m1 = m_lightest_eV
        m2 = np.sqrt(m1 ** 2 + dm21)
        m3 = np.sqrt(m1 ** 2 + dm31)
    else:
        m3 = m_lightest_eV
        m1 = np.sqrt(m3 ** 2 + abs(dm31))
        m2 = np.sqrt(m3 ** 2 + abs(dm31) + dm21)

    return {
        'm1': float(m1),
        'm2': float(m2),
        'm3': float(m3),
        'sum': float(m1 + m2 + m3)
    }
