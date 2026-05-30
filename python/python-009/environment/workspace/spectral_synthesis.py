
import numpy as np
from scipy.special import wofz
from typing import Tuple, Dict, Optional


class LineProfile:

    @staticmethod
    def doppler_width(nu0: float, T: float, mu: float) -> float:
        c = 2.99792458e10
        k_B = 1.380649e-16
        m_u = 1.660539e-24
        m = mu * m_u
        return (nu0 / c) * np.sqrt(2.0 * k_B * T / m)

    @staticmethod
    def lorentz_width(P: float, T: float, T_ref: float = 296.0,
                      gamma_ref: float = 0.1, n_coeff: float = 0.5) -> float:
        P_atm = P / 1.01325e5
        gamma = gamma_ref * P_atm * (T_ref / T)**n_coeff

        return gamma * 2.99792458e10

    @staticmethod
    def voigt_profile(nu: np.ndarray, nu0: float, alpha_d: float,
                       gamma_l: float) -> np.ndarray:
        nu = np.asarray(nu, dtype=np.float64)
        dx = nu - nu0
        sigma = alpha_d / np.sqrt(2.0 * np.log(2.0))
        gamma = gamma_l * 0.5


        if sigma < 1e-30:
            return np.zeros_like(nu)

        z = (dx + 1j * gamma) / (sigma * np.sqrt(2.0))

        voigt = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
        return np.maximum(voigt, 0.0)

    @staticmethod
    def joukowsky_mapped_profile(nu: np.ndarray, nu0: float, width: float,
                                  strength: float = 1.0) -> np.ndarray:
        nu = np.asarray(nu, dtype=np.float64)
        dx = nu - nu0
        sigma = width / np.sqrt(2.0)
        gamma = width * 0.3

        if sigma < 1e-30:
            return np.zeros_like(nu)

        z = (dx + 1j * gamma) / (sigma * np.sqrt(2.0))

        profile = np.abs(np.imag(wofz(z))) / (sigma * np.sqrt(2.0 * np.pi))
        return strength * profile


class MolecularCrossSection:

    def __init__(self, species: str):
        self.species = species
        self.line_data = self._load_simplified_line_data(species)

    def _load_simplified_line_data(self, species: str) -> Dict:

        if species == 'H2O':
            lines = {
                'nu0': np.array([1500.0, 1600.0, 3750.0, 5350.0, 6350.0]),
                'S0': np.array([1e-19, 5e-20, 2e-19, 1e-20, 3e-21]),
                'E_low': np.array([100.0, 200.0, 500.0, 1000.0, 1500.0]),
                'gamma_air': np.array([0.08, 0.07, 0.09, 0.06, 0.05]),
                'n_air': np.array([0.5, 0.5, 0.5, 0.5, 0.5])
            }
        elif species == 'CH4':
            lines = {
                'nu0': np.array([1300.0, 2900.0, 4300.0, 6000.0]),
                'S0': np.array([5e-20, 2e-19, 8e-21, 1e-21]),
                'E_low': np.array([50.0, 300.0, 800.0, 1200.0]),
                'gamma_air': np.array([0.06, 0.07, 0.05, 0.04]),
                'n_air': np.array([0.5, 0.5, 0.5, 0.5])
            }
        elif species == 'CO':
            lines = {
                'nu0': np.array([2100.0, 4200.0, 6350.0]),
                'S0': np.array([2e-19, 1e-20, 5e-22]),
                'E_low': np.array([200.0, 600.0, 1000.0]),
                'gamma_air': np.array([0.05, 0.04, 0.03]),
                'n_air': np.array([0.5, 0.5, 0.5])
            }
        elif species == 'CO2':
            lines = {
                'nu0': np.array([2300.0, 4600.0, 6200.0]),
                'S0': np.array([1e-19, 5e-21, 2e-21]),
                'E_low': np.array([150.0, 500.0, 900.0]),
                'gamma_air': np.array([0.07, 0.06, 0.05]),
                'n_air': np.array([0.5, 0.5, 0.5])
            }
        elif species == 'Na':
            lines = {
                'nu0': np.array([16973.0, 16956.0]),
                'S0': np.array([1e-15, 2e-15]),
                'E_low': np.array([0.0, 0.0]),
                'gamma_air': np.array([0.2, 0.2]),
                'n_air': np.array([0.5, 0.5])
            }
        else:
            lines = {
                'nu0': np.array([3000.0]),
                'S0': np.array([1e-21]),
                'E_low': np.array([100.0]),
                'gamma_air': np.array([0.05]),
                'n_air': np.array([0.5])
            }
        return lines

    def line_strength_temperature(self, T: float, T_ref: float = 296.0) -> np.ndarray:
        c2 = 1.4387770
        S0 = self.line_data['S0']
        E_low = self.line_data['E_low']
        nu0 = self.line_data['nu0']

        if T <= 0 or T_ref <= 0:
            raise ValueError("温度必须为正")

        Q_ratio = (T_ref / T)**1.5
        boltzmann = np.exp(-c2 * E_low * (1.0 / T - 1.0 / T_ref))
        stimulated_emission = (1.0 - np.exp(-c2 * nu0 / T)) / (1.0 - np.exp(-c2 * nu0 / T_ref))
        stimulated_emission = np.where(nu0 / T > 50.0, 1.0, stimulated_emission)

        return S0 * Q_ratio * boltzmann * stimulated_emission

    def compute_cross_section(self, wavenumber: np.ndarray, T: float, P: float) -> np.ndarray:
        wavenumber = np.asarray(wavenumber, dtype=np.float64)
        sigma_total = np.zeros_like(wavenumber)

        S_T = self.line_strength_temperature(T)
        nu0_lines = self.line_data['nu0']
        gamma_air = self.line_data['gamma_air']
        n_air = self.line_data['n_air']

        c = 2.99792458e10

        for j in range(len(nu0_lines)):
            nu0_j = nu0_lines[j]



            mu_mol = {'H2O': 18.0, 'CH4': 16.0, 'CO': 28.0, 'CO2': 44.0, 'Na': 23.0}.get(self.species, 30.0)
            alpha_d_cm = nu0_j * 3.58e-7 * np.sqrt(T / mu_mol)


            P_atm = P / 1.01325e5
            gamma_l_cm = gamma_air[j] * P_atm * (296.0 / T)**n_air[j]

            if alpha_d_cm < 1e-30:
                continue


            profile = LineProfile.voigt_profile(
                wavenumber, nu0_j, alpha_d_cm, gamma_l_cm
            )
            sigma_total += S_T[j] * profile

        return np.maximum(sigma_total, 0.0)


class RayleighScattering:

    @staticmethod
    def cross_section_H2(wavelength_um: np.ndarray) -> np.ndarray:
        wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
        if np.any(wavelength_um <= 0):
            raise ValueError("波长必须为正")
        sigma_0 = 1.2e-28
        lam_0 = 1.0
        return sigma_0 * (lam_0 / wavelength_um)**4

    @staticmethod
    def cross_section_He(wavelength_um: np.ndarray) -> np.ndarray:
        return RayleighScattering.cross_section_H2(wavelength_um) * 0.1

    @staticmethod
    def effective_cross_section(wavelength_um: np.ndarray,
                                vmr_H2: float = 0.85,
                                vmr_He: float = 0.15) -> np.ndarray:
        return vmr_H2 * RayleighScattering.cross_section_H2(wavelength_um) + \
               vmr_He * RayleighScattering.cross_section_He(wavelength_um)
