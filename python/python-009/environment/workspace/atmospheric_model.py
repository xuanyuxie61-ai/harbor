
import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.special import erf


class AtmosphericProfile:


    K_B = 1.380649e-23
    AMU = 1.66053906660e-27
    G = 6.67430e-11
    M_SUN = 1.98847e30
    R_JUPITER = 6.9911e7
    R_EARTH = 6.371e6

    def __init__(self, planet_mass_kg: float, planet_radius_m: float,
                 star_mass_kg: float, orbital_distance_m: float):
        if planet_mass_kg <= 0 or planet_radius_m <= 0:
            raise ValueError("行星质量和半径必须为正")
        if star_mass_kg <= 0 or orbital_distance_m <= 0:
            raise ValueError("恒星质量和轨道距离必须为正")

        self.M_p = planet_mass_kg
        self.R_p = planet_radius_m
        self.M_star = star_mass_kg
        self.a = orbital_distance_m


        self.T_eq = self._equilibrium_temperature()

    def _equilibrium_temperature(self, albedo: float = 0.0, redistribution: float = 0.25) -> float:
        R_star = 6.957e8
        T_star = 5778.0
        T_eq = T_star * np.sqrt(R_star / (2.0 * self.a)) * ((1.0 - albedo) * redistribution)**0.25
        return T_eq

    def gravity(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        r = self.R_p + z
        r = np.maximum(r, 1e-6)
        return self.G * self.M_p / r**2

    def guillot_temperature_profile(self, pressure: np.ndarray,
                                     T_int: float = 100.0,
                                     T_irr: float = None,
                                     gamma: float = 16.0 / 3.0,
                                     kappa_ir: float = 1e-2,
                                     kappa_v1: float = 6e-3,
                                     kappa_v2: float = 1e-4,
                                     f: float = 1.0 / 4.0) -> np.ndarray:
        pressure = np.asarray(pressure, dtype=np.float64)
        if np.any(pressure <= 0):
            raise ValueError("压强必须为正")

        if T_irr is None:
            T_irr = self.T_eq

        g_surf = self.gravity(0.0)
        tau = pressure * kappa_ir / g_surf

        term1 = 0.75 * T_int**4 * (2.0 / 3.0 + tau)

        term2_coeff = 2.0 / 3.0 + 1.0 / (gamma * np.sqrt(3.0))
        term2_exp = (gamma / np.sqrt(3.0) - 1.0 / (gamma * np.sqrt(3.0))) * np.exp(-gamma * tau * np.sqrt(3.0))
        term2 = 0.75 * T_irr**4 * f * (term2_coeff + term2_exp)

        T4 = term1 + term2
        T4 = np.maximum(T4, 1e-10)
        return T4**0.25

    def isothermal_profile(self, pressure: np.ndarray, T0: float) -> np.ndarray:
        pressure = np.asarray(pressure, dtype=np.float64)
        if np.any(pressure <= 0):
            raise ValueError("压强必须为正")
        return np.full_like(pressure, T0, dtype=np.float64)

    def hydrostatic_pressure_grid(self, n_layers: int, P_top: float, P_bot: float) -> np.ndarray:
        if n_layers < 2:
            raise ValueError("层数至少为 2")
        if P_top <= 0 or P_bot <= 0 or P_top >= P_bot:
            raise ValueError("压强范围不合法，需要 P_top < P_bot 且均为正")

        logP = np.linspace(np.log10(P_top), np.log10(P_bot), n_layers)
        return 10.0**logP

    def scale_height(self, T: float, mu: float, z: float = 0.0) -> float:
        if T <= 0 or mu <= 0:
            raise ValueError("温度和平均分子量必须为正")
        g = self.gravity(z)
        if g <= 0:
            raise ValueError("重力加速度必须为正")
        return self.K_B * T / (mu * self.AMU * g)

    def altitude_from_pressure(self, pressure: np.ndarray, T: np.ndarray,
                               mu: np.ndarray, P_ref: float = None) -> np.ndarray:
        pressure = np.asarray(pressure, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        if pressure.shape != T.shape:
            raise ValueError("压强和温度数组形状必须一致")
        if mu.shape != pressure.shape and mu.size != 1:
            raise ValueError("mu 必须是标量或与压强同形状数组")
        if mu.size == 1:
            mu = np.full_like(pressure, float(mu))




        return np.zeros_like(pressure)


class ChemicalEquilibrium:

    def __init__(self, species_list: List[str]):
        self.species = species_list
        self.molar_masses = {
            'H2': 2.016, 'He': 4.003, 'H2O': 18.015, 'CH4': 16.043,
            'CO': 28.010, 'CO2': 44.010, 'NH3': 17.031, 'N2': 28.014,
            'O2': 31.999, 'Na': 22.990, 'K': 39.098, 'TiO': 63.866,
            'VO': 66.941, 'FeH': 56.853, 'HCN': 27.026, 'C2H2': 26.038,
            'PH3': 33.998, 'H2S': 34.082
        }

    def equilibrium_abundance(self, species: str, T: np.ndarray, P: np.ndarray,
                              metallicity: float = 1.0, C_O_ratio: float = 0.54) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        if T.shape != P.shape:
            raise ValueError("温度和压强数组形状必须一致")
        if np.any(T <= 0) or np.any(P <= 0):
            raise ValueError("温度和压强必须为正")

        P_bar = P / 1e5


        if species == 'H2':
            vmr = 0.85 * np.ones_like(T)
        elif species == 'He':
            vmr = 0.15 * np.ones_like(T)
        elif species == 'H2O':

            vmr = 1e-3 * metallicity * (P_bar**0.1) * np.exp(-8000.0 / T)
            vmr = np.minimum(vmr, 1e-2 * metallicity)
        elif species == 'CH4':

            vmr = 1e-4 * metallicity * (P_bar**0.05) * np.exp(-12000.0 / T)
            if C_O_ratio > 1.0:
                vmr *= (C_O_ratio / 0.54)**0.5
            vmr = np.minimum(vmr, 5e-3 * metallicity)
        elif species == 'CO':

            vmr = 1e-3 * metallicity * (P_bar**0.15) * np.exp(-4000.0 / T)
            vmr = np.minimum(vmr, 1e-2 * metallicity)
        elif species == 'CO2':
            vmr = 1e-6 * metallicity * (P_bar**0.2) * np.exp(-6000.0 / T)
            vmr = np.minimum(vmr, 1e-4 * metallicity)
        elif species == 'NH3':
            vmr = 1e-5 * metallicity * (P_bar**0.1) * np.exp(-10000.0 / T)
            vmr = np.minimum(vmr, 1e-3 * metallicity)
        elif species == 'Na':
            vmr = 1e-7 * metallicity * np.exp(-5000.0 / T)
        elif species == 'K':
            vmr = 5e-8 * metallicity * np.exp(-5000.0 / T)
        elif species == 'TiO':
            vmr = 1e-9 * metallicity * np.exp(-12000.0 / T)
        elif species == 'VO':
            vmr = 5e-10 * metallicity * np.exp(-12000.0 / T)
        elif species == 'HCN':
            vmr = 1e-7 * metallicity * (C_O_ratio / 0.54) * np.exp(-9000.0 / T)
        else:
            vmr = 1e-12 * np.ones_like(T)

        vmr = np.maximum(vmr, 1e-30)
        return vmr

    def mean_molecular_weight(self, abundances: Dict[str, np.ndarray]) -> np.ndarray:
        if not abundances:
            raise ValueError("丰度字典为空")

        shape = None
        total_mass = None
        total_moles = None

        for sp, vmr in abundances.items():
            vmr = np.asarray(vmr, dtype=np.float64)
            if shape is None:
                shape = vmr.shape
                total_mass = np.zeros(shape, dtype=np.float64)
                total_moles = np.zeros(shape, dtype=np.float64)
            if vmr.shape != shape:
                raise ValueError(f"物种 {sp} 的丰度数组形状不一致")

            mu_i = self.molar_masses.get(sp, 20.0)
            total_mass += vmr * mu_i
            total_moles += vmr

        total_moles = np.maximum(total_moles, 1e-30)
        return total_mass / total_moles

    def sample_abundance_uncertainty(self, vmr_mean: np.ndarray,
                                      sigma_log: float = 0.5,
                                      n_samples: int = 1,
                                      seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        vmr_mean = np.asarray(vmr_mean, dtype=np.float64)
        log_vmr = np.log10(np.maximum(vmr_mean, 1e-30))
        noise = np.random.normal(0.0, sigma_log, size=(n_samples,) + vmr_mean.shape)
        samples = 10.0**(log_vmr + noise)
        return np.maximum(samples, 1e-30)


class CloudModel:

    def __init__(self, P_cloud_top: float = 1e2, P_cloud_base: float = 1e4,
                 cloud_opacity: float = 1.0, particle_radius_m: float = 1e-6):
        self.P_top = P_cloud_top
        self.P_base = P_cloud_base
        self.cloud_opacity = cloud_opacity
        self.r_particle = particle_radius_m

    def cloud_optical_depth(self, pressure: np.ndarray) -> np.ndarray:
        pressure = np.asarray(pressure, dtype=np.float64)
        logP = np.log10(pressure)
        logP_c = 0.5 * (np.log10(self.P_top) + np.log10(self.P_base))
        sigma_P = 0.5 * abs(np.log10(self.P_base) - np.log10(self.P_top))

        if sigma_P < 1e-10:
            return np.zeros_like(pressure)

        tau = self.cloud_opacity * np.exp(-((logP - logP_c) / sigma_P)**2)
        tau = np.where((pressure >= self.P_top) & (pressure <= self.P_base), tau, 0.0)
        return tau
