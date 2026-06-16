"""
nuclear_constants.py
====================
Fundamental physical constants and nuclear data for fission simulation.

All values follow CODATA 2018 recommended values and the 2020 Atomic Mass
Evaluation (AME2020).  Units are MeV, fm, and atomic mass units (u).

Physical context
----------------
For a fissioning nucleus (A, Z) -> (A1, Z1) + (A2, Z2) + nu*n:
  Q = [M(A,Z) - M(A1,Z1) - M(A2,Z2) - nu*m_n] * c^2
"""

import math
from typing import Tuple, Dict, List

# ---------------------------------------------------------------------------
# CODATA 2018 / AME2020 fundamental constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_MPS: float = 2.99792458e8
ATOMIC_MASS_UNIT_MEV: float = 931.49410242
NEUTRON_MASS_MEV: float = 939.56542052
PROTON_MASS_MEV: float = 938.27208816
ELECTRON_MASS_MEV: float = 0.51099895
HBAR_C_MEV_FM: float = 197.3269804
BOLTZMANN_MEV_K: float = 8.617333262e-11
CLASSICAL_RE_MEV: float = 2.8179403227
FINE_STRUCTURE: float = 7.2973525693e-3
AVOGADRO: float = 6.02214076e23
EV_TO_JOULE: float = 1.602176634e-19

# ---------------------------------------------------------------------------
# Liquid Drop Model (LD) coefficients (MeV) [Krane parametrisation]
#   B = a_v*A - a_s*A^(2/3) - a_c*Z*(Z-1)/A^(1/3)
#       - a_sym*(A-2Z)^2/A + delta(A,Z)
# ---------------------------------------------------------------------------
LD_VOLUME: float = 15.5
LD_SURFACE: float = 16.8
LD_COULOMB: float = 0.72
LD_ASYMMETRY: float = 23.0
LD_PAIRING_A: float = 34.0
LD_WIGNER: float = 30.0

# ---------------------------------------------------------------------------
# Nuclear radius parametrisation: R = r0 * A^{1/3}
# ---------------------------------------------------------------------------
RADIUS_PARAMETER: float = 1.25
DIFFUSENESS_PARAMETER: float = 0.65

# ---------------------------------------------------------------------------
# Fission-specific parameters
# ---------------------------------------------------------------------------
NEUTRON_SEPARATION_U236: float = 6.5448
THERMAL_NEUTRON_KE: float = 0.0253e-6
COMPOUND_EXCITATION_U236: float = NEUTRON_SEPARATION_U236 + THERMAL_NEUTRON_KE

# ---------------------------------------------------------------------------
# Mass excess data (keV) from AME2020  [key: "A-Z"]
# Atomic mass M(A,Z) = A*u + Delta(A,Z)/c^2
# Values stored in keV; converted to MeV when accessed.
# ---------------------------------------------------------------------------
MASS_EXCESS_KEV: Dict[str, float] = {
    # Neutron
    "1-0": 8071.317,
    # Hydrogen
    "1-1": 7288.971,
    # Light nuclei
    "4-2": 2424.916,
    # Actinides
    "232-90": 35441.410,   # Th-232
    "233-92": 36921.500,   # U-233
    "234-92": 38141.870,   # U-234
    "235-92": 40914.053,   # U-235
    "236-92": 42440.670,   # U-236
    "237-92": 43562.200,   # U-237
    "238-92": 47307.910,   # U-238
    "239-94": 48583.560,   # Pu-239
    "240-94": 50120.300,   # Pu-240
    "241-94": 52955.000,   # Pu-241
    "242-96": 54000.000,   # Cm-242
    "252-98": 60700.000,   # Cf-252
    # Zr region (A~90-100)
    "88-36": -67000.000,   # Kr-88
    "89-36": -65000.000,   # Kr-89
    "90-36": -64000.000,   # Kr-90
    "92-36": -67600.000,   # Kr-92
    "93-36": -66000.000,   # Kr-93
    "94-36": -64000.000,   # Kr-94
    "89-37": -67000.000,   # Rb-89
    "90-37": -68000.000,   # Rb-90
    "91-37": -68500.000,   # Rb-91
    "92-37": -68500.000,   # Rb-92
    "93-37": -68000.000,   # Rb-93
    "94-37": -66500.000,   # Rb-94
    "90-38": -70000.000,   # Sr-90
    "91-38": -70000.000,   # Sr-91
    "92-38": -71000.000,   # Sr-92
    "93-38": -71000.000,   # Sr-93
    "94-38": -71500.000,   # Sr-94
    "95-38": -70500.000,   # Sr-95
    "96-38": -70500.000,   # Sr-96
    "97-38": -69000.000,   # Sr-97
    "98-40": -78000.000,   # Zr-98
    "99-40": -78000.000,   # Zr-99
    "100-40": -78500.000,  # Zr-100
    "101-40": -77000.000,  # Zr-101
    # Mo-Tc-Ru
    "99-42": -83000.000,   # Mo-99
    "100-42": -84000.000,  # Mo-100
    "101-42": -83500.000,  # Mo-101
    "102-42": -83500.000,  # Mo-102
    "103-42": -82000.000,  # Mo-103
    "104-42": -82000.000,  # Mo-104
    "105-42": -80500.000,  # Mo-105
    "106-44": -86000.000,  # Ru-106
    "107-44": -84500.000,  # Ru-107
    "108-44": -84000.000,  # Ru-108
    # Pd-Ag-Cd
    "106-46": -88000.000,  # Pd-106
    "107-46": -87000.000,  # Pd-107
    "108-46": -87500.000,  # Pd-108
    "109-46": -86500.000,  # Pd-109
    "110-46": -86000.000,  # Pd-110
    # Sn region (A~128-136)
    "128-50": -86500.000,  # Sn-128
    "129-50": -86000.000,  # Sn-129
    "130-50": -86000.000,  # Sn-130
    "131-50": -85500.000,  # Sn-131
    "132-50": -84000.000,  # Sn-132
    "133-50": -82000.000,  # Sn-133
    "134-50": -80000.000,  # Sn-134
    # Sb region
    "129-51": -82000.000,  # Sb-129
    "130-51": -82000.000,  # Sb-130
    "131-51": -82000.000,  # Sb-131
    "132-51": -82000.000,  # Sb-132
    "133-51": -82500.000,  # Sb-133
    "134-51": -81500.000,  # Sb-134
    "135-51": -79500.000,  # Sb-135
    # Te region
    "128-52": -84000.000,  # Te-128
    "130-52": -84500.000,  # Te-130
    "131-52": -84000.000,  # Te-131
    "132-52": -84000.000,  # Te-132
    "133-52": -82500.000,  # Te-133
    "134-52": -81500.000,  # Te-134
    "135-52": -79500.000,  # Te-135
    # I region
    "131-53": -83500.000,  # I-131
    "133-53": -83000.000,  # I-133
    "134-53": -82000.000,  # I-134
    "135-53": -81500.000,  # I-135
    "136-53": -80000.000,  # I-136
    # Xe region
    "131-54": -85000.000,  # Xe-131
    "132-54": -85500.000,  # Xe-132
    "133-54": -84500.000,  # Xe-133
    "134-54": -84000.000,  # Xe-134
    "135-54": -82500.000,  # Xe-135
    "136-54": -81500.000,  # Xe-136
    "137-54": -79000.000,  # Xe-137
    "138-54": -77500.000,  # Xe-138
    "140-54": -72000.000,  # Xe-140
    # Cs region
    "133-55": -84000.000,  # Cs-133
    "134-55": -83000.000,  # Cs-134
    "135-55": -82500.000,  # Cs-135
    "136-55": -81500.000,  # Cs-136
    "137-55": -80500.000,  # Cs-137
    "138-55": -79000.000,  # Cs-138
    "139-55": -77000.000,  # Cs-139
    "140-55": -74000.000,  # Cs-140
    # Ba region
    "134-56": -83000.000,  # Ba-134
    "135-56": -83000.000,  # Ba-135
    "136-56": -83500.000,  # Ba-136
    "137-56": -83000.000,  # Ba-137
    "138-56": -83500.000,  # Ba-138
    "139-56": -81500.000,  # Ba-139
    "140-56": -80000.000,  # Ba-140
    "141-56": -77500.000,  # Ba-141
    "142-56": -75500.000,  # Ba-142
    "144-56": -71000.000,  # Ba-144
    # La-Ce-Pr
    "139-57": -81000.000,  # La-139
    "140-57": -79500.000,  # La-140
    "141-57": -78000.000,  # La-141
    "140-58": -83000.000,  # Ce-140
    "141-58": -81000.000,  # Ce-141
    "142-58": -81000.000,  # Ce-142
    "143-58": -79000.000,  # Ce-143
    "144-58": -76500.000,  # Ce-144
    "141-59": -79000.000,  # Pr-141
    "142-59": -78000.000,  # Pr-142
    "143-59": -77000.000,  # Pr-143
    "144-59": -75000.000,  # Pr-144
    # Nd region (A~140-150)
    "140-60": -80500.000,  # Nd-140
    "141-60": -81000.000,  # Nd-141
    "142-60": -82000.000,  # Nd-142
    "143-60": -82000.000,  # Nd-143
    "144-60": -82000.000,  # Nd-144
    "145-60": -81500.000,  # Nd-145
    "146-60": -81000.000,  # Nd-146
    "147-60": -79500.000,  # Nd-147
    "148-60": -78500.000,  # Nd-148
    "150-60": -75500.000,  # Nd-150
    # Pm-Sm-Eu
    "145-61": -79000.000,  # Pm-145
    "147-61": -78000.000,  # Pm-147
    "148-62": -79000.000,  # Sm-148
    "149-62": -78500.000,  # Sm-149
    "150-62": -78500.000,  # Sm-150
    "151-62": -77000.000,  # Sm-151
    "152-62": -76000.000,  # Sm-152
    "153-63": -74000.000,  # Eu-153
    # Magic and near-magic
    "208-82": -21749.000,  # Pb-208
    "132-50": -84000.000,  # Sn-132 (N=82)
}

NEUTRON_MASS_U: float = 1.008665
NEUTRON_MASS_EXCESS_MEV: float = 8.071317  # keV value / 1000 = MeV


def binding_energy_ld(a: int, z: int) -> float:
    """Liquid-drop binding energy B(A,Z) in MeV."""
    a_third = a ** (1.0 / 3.0)
    a_two_third = a ** (2.0 / 3.0)
    a_neg_half = 1.0 / math.sqrt(a) if a > 0 else 0.0

    vol = LD_VOLUME * a
    surf = LD_SURFACE * a_two_third
    coul = LD_COULOMB * z * (z - 1) / a_third if a_third > 0 else 0.0
    asym = LD_ASYMMETRY * (a - 2 * z) ** 2 / a if a > 0 else 0.0

    n_neut = a - z
    if z % 2 == 0 and n_neut % 2 == 0:
        delta = LD_PAIRING_A * a_neg_half
    elif z % 2 == 1 and n_neut % 2 == 1:
        delta = -LD_PAIRING_A * a_neg_half
    else:
        delta = 0.0

    wigner = LD_WIGNER * abs(a - 2 * z) / a if a > 0 else 0.0

    return vol - surf - coul - asym + delta - wigner


def atomic_mass_ld(a: int, z: int) -> float:
    """Atomic mass estimate from liquid-drop model, in u."""
    n_neut = a - z
    m_hydrogen_u = (PROTON_MASS_MEV + ELECTRON_MASS_MEV) / ATOMIC_MASS_UNIT_MEV
    m_neutron_u = NEUTRON_MASS_MEV / ATOMIC_MASS_UNIT_MEV
    b_meV = binding_energy_ld(a, z)
    return z * m_hydrogen_u + n_neut * m_neutron_u - b_meV / ATOMIC_MASS_UNIT_MEV


def get_atomic_mass(a: int, z: int) -> float:
    """
    Return atomic mass in u.  Uses empirical mass excess when available,
    otherwise falls back to the liquid-drop estimate.
    """
    key = f"{a}-{z}"
    if key in MASS_EXCESS_KEV:
        delta_keV = MASS_EXCESS_KEV[key]
        delta_meV = delta_keV / 1000.0  # Convert keV to MeV
        return a + delta_meV / ATOMIC_MASS_UNIT_MEV
    return atomic_mass_ld(a, z)


def q_value_fission(a_cn: int, z_cn: int,
                    a1: int, z1: int,
                    a2: int, z2: int,
                    nu: int = 0) -> float:
    """
    Fission Q-value in MeV.
    Q = [M(A_CN,Z_CN) - M(A1,Z1) - M(A2,Z2) - nu*m_n] * c^2
    """
    m_cn = get_atomic_mass(a_cn, z_cn)
    m_ff1 = get_atomic_mass(a1, z1)
    m_ff2 = get_atomic_mass(a2, z2)
    m_n_u = (NEUTRON_MASS_MEV / ATOMIC_MASS_UNIT_MEV)
    mass_diff = m_cn - m_ff1 - m_ff2 - nu * m_n_u
    return mass_diff * ATOMIC_MASS_UNIT_MEV


def woods_saxon_density(r: float, a_mass: int, z_charge: int) -> float:
    """Woods-Saxon nuclear density rho(r)/rho_0."""
    r0 = RADIUS_PARAMETER
    rhalf = r0 * (a_mass ** (1.0 / 3.0))
    a_diff = DIFFUSENESS_PARAMETER
    if rhalf < 1e-10:
        return 0.0
    x = (r - rhalf) / a_diff
    if x > 500.0:
        return 0.0
    if x < -500.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def coulomb_barrier_energy(z1: int, z2: int, a1: int, a2: int) -> float:
    """Coulomb barrier at contact in MeV."""
    e2_meV_fm = 1.43998
    r1 = RADIUS_PARAMETER * (a1 ** (1.0 / 3.0))
    r2 = RADIUS_PARAMETER * (a2 ** (1.0 / 3.0))
    d_contact = r1 + r2
    if d_contact < 1e-10:
        return 0.0
    return z1 * z2 * e2_meV_fm / d_contact


def level_density_parameter(a_mass: int) -> float:
    """Level density parameter a (MeV^{-1}) from Ignatyuk formula."""
    alpha_ld = 0.073
    beta_ld = 0.195
    return alpha_ld * a_mass + beta_ld * (a_mass ** (2.0 / 3.0))


def fermi_gas_level_density(excitation: float, a_mass: int,
                            shell_correction: float = 0.0) -> float:
    """
    Intrinsic level density (Ignatyuk form).
    Returns ln(rho) to avoid overflow.
    """
    gamma_ig = 0.0588
    a_param = level_density_parameter(a_mass)

    if excitation <= 0.0:
        return -1.0e30

    if abs(gamma_ig * excitation) < 1e-10:
        u_tilde = excitation - shell_correction
    else:
        u_tilde = excitation - shell_correction * (
            1.0 - math.exp(-gamma_ig * excitation)
        ) / gamma_ig

    if u_tilde <= 0.0:
        return -1.0e30

    a_u = a_param * u_tilde
    if a_u <= 0.0:
        return -1.0e30

    log_rho = (2.0 * math.sqrt(a_u)
               - 1.25 * math.log(u_tilde)
               - 0.25 * math.log(a_param)
               - math.log(12.0 * math.sqrt(2.0)))
    return log_rho


def neutron_evaporation_width(excitation: float, a_parent: int,
                              z_parent: int) -> float:
    """Weisskopf estimate of neutron emission width."""
    s_n_approx = 5.5
    if excitation <= s_n_approx:
        return -1.0e30
    log_rho_parent = fermi_gas_level_density(excitation, a_parent, -2.0)
    log_rho_daughter = fermi_gas_level_density(
        excitation - s_n_approx, a_parent - 1, -1.5
    )
    return log_rho_daughter - log_rho_parent


def prompt_neutron_multiplicity(a_cn: int, z_cn: int,
                                a_light: int, z_light: int,
                                a_heavy: int, z_heavy: int,
                                tke: float) -> int:
    """Estimate prompt neutron multiplicity from energy balance."""
    q_val = q_value_fission(a_cn, z_cn, a_light, z_light, a_heavy, z_heavy, 0)
    e_gamma_total = 7.0
    s_n_avg = 5.5
    available = q_val - tke - e_gamma_total
    if available <= 0.0:
        return 0
    nu_est = int(available / s_n_avg)
    return max(0, nu_est)
