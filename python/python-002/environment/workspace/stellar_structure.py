# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional
from numerical_utils import safe_divide, brent_root, tridiag_solve, solve_linear


class StellarStructure:


    G = 6.67430e-8
    A_RAD = 7.5657e-15
    C_LIGHT = 2.99792458e10
    R_GAS = 8.314462618e7
    SIGMA_SB = 5.670374419e-5
    KAPPA_ES = 0.2 * (1.0 + 0.74)

    def __init__(self, M_total: float, R_init: float, composition: Optional[np.ndarray] = None):
        self.M_total = M_total
        self.R_init = R_init
        if composition is None:
            self.composition = np.array([0.7, 0.0, 0.28, 0.01, 0.005, 0.003, 0.001, 0.001])
        else:
            self.composition = np.array(composition, dtype=np.float64)
            self.composition /= np.sum(self.composition)

    def mean_molecular_weight(self, X: np.ndarray) -> float:
        X_h1 = X[0]
        Y_he = X[2]
        Z_metal = np.sum(X[3:])
        if X_h1 + Y_he + Z_metal <= 0:
            return 0.6
        mu_inv = 2.0 * X_h1 + 3.0 * Y_he / 4.0 + Z_metal / 2.0
        mu_inv = max(mu_inv, 0.5)
        return 1.0 / mu_inv

    def equation_of_state(self, rho: float, T: float, X: np.ndarray) -> Tuple[float, float, float, float]:
        mu = self.mean_molecular_weight(X)
        P_gas = self.R_GAS / mu * rho * T
        P_rad = self.A_RAD / 3.0 * T ** 4
        P = P_gas + P_rad

        if P <= 0:
            P = 1e-5
            P_gas = 1e-5
            P_rad = 1e-10

        beta = P_gas / P
        gamma_gas = 5.0 / 3.0

        Gamma1 = beta + (4.0 - 3.0 * beta) ** 2 * (gamma_gas - 1.0) / (
                beta + 12.0 * (gamma_gas - 1.0) * (1.0 - beta))
        return P, P_gas, P_rad, Gamma1

    def opacity(self, rho: float, T: float, X: np.ndarray) -> float:
        Z_metal = np.sum(X[3:])
        kappa_es = self.KAPPA_ES

        kappa_kramers = 4.4e24 * (Z_metal + 0.001) * rho * T ** (-3.5)
        kappa = kappa_es + kappa_kramers

        kappa = np.clip(kappa, 1e-4, 1e6)
        return kappa

    def temperature_gradients(self, m: float, r: float, P: float, T: float,
                              L: float, rho: float, X: np.ndarray) -> Tuple[float, float, float, bool]:
        if m <= 0 or r <= 0 or T <= 0:
            return 0.0, 0.4, 0.4, False

        kappa = self.opacity(rho, T, X)

        nabla_rad = (3.0 * kappa * L * P) / (16.0 * np.pi * self.A_RAD * self.C_LIGHT
                                               * self.G * m * T ** 4)
        nabla_rad = max(0.0, min(nabla_rad, 10.0))


        mu = self.mean_molecular_weight(X)
        P_gas = self.R_GAS / mu * rho * T
        beta = P_gas / P if P > 0 else 1.0
        gamma = 5.0 / 3.0
        nabla_ad = (gamma - 1.0) / gamma * beta / (4.0 - 3.0 * beta)
        nabla_ad = max(0.1, min(nabla_ad, 0.5))

        if nabla_rad > nabla_ad:
            return nabla_rad, nabla_ad, nabla_ad, True
        else:
            return nabla_rad, nabla_ad, nabla_rad, False

    def hydrostatic_structure(self, mass: np.ndarray, rho: np.ndarray,
                              P_c: float) -> np.ndarray:
        n = len(mass)
        r = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        P[0] = P_c

        for i in range(n - 1):
            dm = mass[i + 1] - mass[i]
            if dm <= 0:
                continue
            r_avg = max(r[i], 1e-3)
            rho_avg = 0.5 * (rho[i] + rho[i + 1])

            dr = dm / (4.0 * np.pi * r_avg ** 2 * rho_avg)
            r[i + 1] = r[i] + dr


            m_avg = 0.5 * (mass[i] + mass[i + 1])
            dP = -self.G * m_avg * dm / (4.0 * np.pi * r_avg ** 4)
            P[i + 1] = P[i] + dP
            if P[i + 1] <= 0:
                P[i + 1] = 1e-5

        return r, P

    def solve_luminosity_profile(self, mass: np.ndarray, epsilon: np.ndarray,
                                 L_surface: float) -> np.ndarray:
        n = len(mass)
        L = np.zeros(n, dtype=np.float64)
        for i in range(n - 1):
            dm = mass[i + 1] - mass[i]
            eps_avg = 0.5 * (epsilon[i] + epsilon[i + 1])
            L[i + 1] = L[i] + eps_avg * dm

        if L[-1] > 0 and L_surface > 0:
            L *= L_surface / L[-1]
        return L

    def eddington_luminosity(self, M: float) -> float:
        return 4.0 * np.pi * self.G * M * self.C_LIGHT / self.KAPPA_ES

    def schwarzschild_radius(self, M: float) -> float:
        return 2.0 * self.G * M / self.C_LIGHT ** 2

    def sound_speed(self, rho: float, T: float, X: np.ndarray) -> float:
        P, _, _, Gamma1 = self.equation_of_state(rho, T, X)
        if rho <= 0:
            return 1e5
        return np.sqrt(Gamma1 * P / rho)

    def dynamical_timescale(self, M: float, R: float, rho_avg: float) -> float:
        if rho_avg <= 0:
            return 1e10
        return 1.0 / np.sqrt(self.G * rho_avg)

    def kelvin_helmholtz_timescale(self, M: float, R: float, L: float) -> float:
        if L <= 0 or R <= 0:
            return 1e20
        return self.G * M ** 2 / (R * L)

    def nuclear_timescale(self, M: float, L: float, f_nuc: float = 0.007) -> float:
        if L <= 0:
            return 1e20
        return f_nuc * M * self.C_LIGHT ** 2 / L

    def thin_data(self, arrays: Tuple[np.ndarray, ...], thin_factor: int = 4) -> Tuple[np.ndarray, ...]:
        if thin_factor <= 1:
            return arrays
        idx = np.arange(0, len(arrays[0]), thin_factor)
        return tuple(np.asarray(a)[idx] for a in arrays)
