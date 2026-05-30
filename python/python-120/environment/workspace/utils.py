
import numpy as np
from typing import Tuple




BOLTZMANN_KB = 1.380649e-23
AVOGADRO_NA = 6.02214076e23
ELEMENTARY_CHARGE = 1.602176634e-19
PLANCK_H = 6.62607015e-34
EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1.0e-10
FS_TO_S = 1.0e-15
AMU_TO_KG = 1.66053906660e-27


PT_LATTICE_CONSTANT = 3.924e-10
PT_ATOMIC_MASS = 195.084


MASS_CO = 28.010
MASS_O2 = 31.998


def kb_t_ev(temperature_k: float) -> float:
    if temperature_k < 0.0:
        raise ValueError("温度必须非负")
    return BOLTZMANN_KB * temperature_k / ELEMENTARY_CHARGE


def maxwell_boltzmann_speed(mass_amu: float, temperature_k: float) -> float:
    if mass_amu <= 0.0 or temperature_k < 0.0:
        raise ValueError("mass_amu > 0 且 temperature_k >= 0")
    m_kg = mass_amu * AMU_TO_KG
    return np.sqrt(2.0 * BOLTZMANN_KB * temperature_k / m_kg)


def de_broglie_thermal_wavelength(mass_amu: float, temperature_k: float) -> float:
    if mass_amu <= 0.0 or temperature_k <= 0.0:
        raise ValueError("mass_amu > 0 且 temperature_k > 0")
    m_kg = mass_amu * AMU_TO_KG
    return PLANCK_H / np.sqrt(2.0 * np.pi * m_kg * BOLTZMANN_KB * temperature_k)


def grid_uniform_1d(xmin: float, xmax: float, nstep: int) -> np.ndarray:
    if nstep < 2:
        raise ValueError("nstep 必须 >= 2")
    if xmax <= xmin:
        raise ValueError("xmax 必须 > xmin")
    i = np.arange(1, nstep + 1)
    return ((nstep - i) * xmin + (i - 1) * xmax) / (nstep - 1)


def grid_uniform_nd(ndim: int, nstep: int, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if nstep < 2:
        raise ValueError("nstep 必须 >= 2")
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != (ndim,) or x2.shape != (ndim,):
        raise ValueError("x1, x2 形状必须为 (ndim,)")

    linspaces = [np.linspace(x1[d], x2[d], nstep) for d in range(ndim)]
    if ndim == 2:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    elif ndim == 3:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    else:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    return grid


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1e-300
    result[mask] = a[mask] / b[mask]
    return result


def morse_potential(r: np.ndarray, d_e: float, a_param: float, r_e: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    dr = r - r_e
    return d_e * (1.0 - np.exp(-a_param * dr)) ** 2 - d_e


def lennard_jones_potential(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def arrhenius_rate(pre_exponential: float, activation_energy_ev: float,
                   temperature_k: float) -> float:
    if temperature_k <= 0.0:
        return 0.0
    kb_t = kb_t_ev(temperature_k)
    return pre_exponential * np.exp(-activation_energy_ev / kb_t)


def sticking_coefficient_langmuir(pressure_pa: float, temperature_k: float,
                                  alpha0: float, e_ads_ev: float) -> float:
    if temperature_k <= 0.0:
        return 0.0
    kb_t = kb_t_ev(temperature_k)
    return alpha0 * np.exp(-e_ads_ev / kb_t)
