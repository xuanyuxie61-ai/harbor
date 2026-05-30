# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional
from reaction_rates import NuclearReactionRates


class NuclearNetwork:


    SPECIES = ['H1', 'He3', 'He4', 'C12', 'N14', 'O16', 'Ne20', 'Mg24']
    MASS_NUMBERS = np.array([1, 3, 4, 12, 14, 16, 20, 24], dtype=np.float64)

    def __init__(self, rates_calculator: Optional[NuclearReactionRates] = None):
        self.rates = rates_calculator if rates_calculator is not None else NuclearReactionRates()
        self.n_species = len(self.SPECIES)

        self._build_stoichiometry()

    def _build_stoichiometry(self):
        n_react = 16
        self.nu = np.zeros((self.n_species, n_react), dtype=np.float64)
        idx = {s: i for i, s in enumerate(self.SPECIES)}


        self.nu[idx['H1'], 0] = -2.0

        self.nu[idx['H1'], 1] = -1.0
        self.nu[idx['He3'], 1] = +1.0

        self.nu[idx['He3'], 2] = -2.0
        self.nu[idx['He4'], 2] = +1.0
        self.nu[idx['H1'], 2] = +2.0

        self.nu[idx['He3'], 3] = -1.0
        self.nu[idx['He4'], 3] = -1.0


        self.nu[idx['H1'], 5] = -1.0
        self.nu[idx['He4'], 5] = +2.0

        self.nu[idx['H1'], 6] = -1.0


        self.nu[idx['He4'], 8] = +2.0

        self.nu[idx['C12'], 9] = -1.0
        self.nu[idx['H1'], 9] = -1.0

        self.nu[idx['C12'], 10] = +1.0

        self.nu[idx['H1'], 11] = -1.0
        self.nu[idx['N14'], 11] = +1.0

        self.nu[idx['N14'], 12] = -1.0
        self.nu[idx['H1'], 12] = -1.0

        self.nu[idx['H1'], 13] = +1.0

        self.nu[idx['H1'], 14] = -1.0
        self.nu[idx['C12'], 14] = +1.0
        self.nu[idx['He4'], 14] = +1.0

        self.nu[idx['He4'], 15] = -3.0
        self.nu[idx['C12'], 15] = +1.0

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        limit = int(np.sqrt(n)) + 1
        for d in range(3, limit, 2):
            if n % d == 0:
                return False
        return True

    def magic_number_stability(self, A: int) -> float:
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]

        dist = min(abs(A - m) for m in magic_numbers)
        stability = np.exp(-dist / 10.0)
        if self.is_prime(A):
            stability *= 1.1
        return min(stability, 1.0)

    def abundances_to_mass_fractions(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        mass_per_nucleon = self.MASS_NUMBERS * Y
        total = np.sum(mass_per_nucleon)
        if total <= 0:
            return np.ones_like(Y) / len(Y)
        X = mass_per_nucleon / total
        return np.clip(X, 1e-15, 1.0)

    def compute_derivatives(self, t: float, Y: np.ndarray,
                            T: float, rho: float) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        Y = np.maximum(Y, 1e-30)
        X = self.abundances_to_mass_fractions(Y)
        X_h1 = X[0]
        X_he4 = X[2]
        Z_cno = X[3] + X[4] + X[5]
        Z_metal = np.sum(X[3:])
        Y_val = X[2]

        rates_pp = self.rates.pp_chain_rates(T, rho, X_h1, Y_val, Z_metal)
        rates_cno = self.rates.cno_cycle_rates(T, rho, X_h1, Y_val, Z_metal)
        rate_3a = self.rates.triple_alpha_rate(T, rho, Y_val)






        raise NotImplementedError("Hole 2: 待实现反应率合并与核素丰度演化导数计算")

    def solve_network_rk4(self, Y0: np.ndarray, T: float, rho: float,
                          dt: float, n_steps: int = 1) -> np.ndarray:
        Y = np.asarray(Y0, dtype=np.float64).copy()
        for _ in range(n_steps):
            k1 = self.compute_derivatives(0.0, Y, T, rho)
            k2 = self.compute_derivatives(0.0, Y + 0.5 * dt * k1, T, rho)
            k3 = self.compute_derivatives(0.0, Y + 0.5 * dt * k2, T, rho)
            k4 = self.compute_derivatives(0.0, Y + dt * k3, T, rho)
            Y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Y = np.maximum(Y, 1e-30)
        return Y

    def solve_network_euler(self, Y0: np.ndarray, T: float, rho: float,
                            dt: float) -> np.ndarray:
        Y = np.asarray(Y0, dtype=np.float64)
        dY = self.compute_derivatives(0.0, Y, T, rho)
        Y_new = Y + dt * dY
        return np.maximum(Y_new, 1e-30)

    def energy_generation_rate(self, Y: np.ndarray, T: float, rho: float) -> float:
        X = self.abundances_to_mass_fractions(Y)
        X_h1 = X[0]
        Y_he = X[2]
        Z_cno = X[3] + X[4] + X[5]
        Z_metal = np.sum(X[3:])
        return self.rates.energy_generation(T, rho, X_h1, Y_he, Z_cno, Z_metal)
