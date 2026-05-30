# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple


NA = 6.02214076e23
K_B = 1.380649e-16
Q_PP = 6.55e18
Q_CNO = 1.53e19
Q_3A = 7.275e-5 * 1.602e-6






class NuclearReactionRates:

    def __init__(self):
        self.reaction_names = [
            "pp", "pd", "He3He3", "He3He4", "Be7e", "Li7p",
            "Be7p", "B8", "Be8", "C12p", "N13",
            "C13p", "N14p", "O15", "N15p", "triple_alpha"
        ]
        self.num_reactions = len(self.reaction_names)

    @staticmethod
    def _temperature_factor(T9: float, Z1Z2: float, mu: float) -> float:
        if T9 <= 0:
            return 0.0

        tau = 3.0 * (np.pi ** 2 * mu * 1.6605e-24 / (2.0 * (1.054e-27) ** 2)) ** (1.0 / 3.0) \
              * (Z1Z2 * 2.307e-19) ** (2.0 / 3.0) * (1.0 / (K_B * T9 * 1e9)) ** (1.0 / 3.0)
        return tau

    @staticmethod
    def screening_factor(rho: float, T: float, X: float, Y: float, Z_metal: float,
                         Z1: int, Z2: int) -> float:
        if T <= 0 or rho <= 0:
            return 1.0
        T7 = T / 1e7

        zeta = (2.0 ** 2 + 2.0) * X / 1.0 + (2.0 ** 2 + 2.0) * Y / 4.0
        zeta += Z_metal * (0.5 * (Z_metal ** 2 + Z_metal)) / (1.0 + Z_metal)
        zeta = max(zeta, 1e-10)

        H = 1.88 * Z1 * Z2 * (NA ** (1.0 / 3.0)) * 2.307e-19
        arg = H * (rho * zeta / (T7 ** 3)) ** 0.5 / (K_B * T)

        if arg > 10.0:
            arg = 10.0
        return np.exp(arg)

    def pp_chain_rates(self, T: float, rho: float, X: float, Y: float,
                       Z_metal: float) -> np.ndarray:
        if T <= 1e5 or rho <= 0 or X <= 0:
            return np.zeros(self.num_reactions, dtype=np.float64)
        T9 = T / 1e9
        T9_inv = 1.0 / T9

        rates = np.zeros(self.num_reactions, dtype=np.float64)



        f_pp = self.screening_factor(rho, T, X, Y, Z_metal, 1, 1)
        rates[0] = 4.01e-15 * T9 ** (-2.0 / 3.0) * np.exp(-3.380 * T9 ** (-1.0 / 3.0)) * f_pp
        rates[0] *= (1.0 + 0.123 * T9 ** (1.0 / 3.0) + 1.09 * T9 ** (2.0 / 3.0) + 0.938 * T9)


        rates[1] = 2.24e3 * T9 ** (-2.0 / 3.0) * np.exp(-3.720 * T9 ** (-1.0 / 3.0))
        rates[1] *= (1.0 + 0.112 * T9 ** (1.0 / 3.0) + 1.99 * T9 ** (2.0 / 3.0) + 1.56 * T9)


        f_33 = self.screening_factor(rho, T, X, Y, Z_metal, 2, 2)
        rates[2] = 6.04e10 * T9 ** (-2.0 / 3.0) * np.exp(-12.276 * T9 ** (-1.0 / 3.0)) * f_33
        rates[2] *= (1.0 + 0.034 * T9 ** (1.0 / 3.0) - 0.522 * T9 ** (2.0 / 3.0)
                     - 0.124 * T9 + 0.353 * T9 ** (4.0 / 3.0) + 0.213 * T9 ** (5.0 / 3.0))


        f_34 = self.screening_factor(rho, T, X, Y, Z_metal, 2, 2)
        rates[3] = 5.61e6 * T9 ** (-2.0 / 3.0) * np.exp(-12.826 * T9 ** (-1.0 / 3.0)) * f_34
        rates[3] *= (1.0 + 0.015 * T9 ** (1.0 / 3.0) + 0.238 * T9 ** (2.0 / 3.0)
                     + 0.030 * T9 + 0.042 * T9 ** (4.0 / 3.0) + 0.020 * T9 ** (5.0 / 3.0))



        if T9 < 0.01:
            rates[4] = 1.34e-10 * T9 ** (-0.5)
        else:
            rates[4] = 1.34e-10 * T9 ** (-0.5) * (1.0 - 0.537 * T9 ** (1.0 / 3.0)
                                                     + 3.86 * T9 ** (2.0 / 3.0)
                                                     - 5.48 * T9)


        f_Li = self.screening_factor(rho, T, X, Y, Z_metal, 1, 3)
        rates[5] = 1.096e9 * T9 ** (-2.0 / 3.0) * np.exp(-8.472 * T9 ** (-1.0 / 3.0)) * f_Li
        rates[5] *= (1.0 + 0.049 * T9 ** (1.0 / 3.0) - 0.134 * T9 ** (2.0 / 3.0)
                     + 0.010 * T9 + 0.019 * T9 ** (4.0 / 3.0))


        f_Be = self.screening_factor(rho, T, X, Y, Z_metal, 1, 4)
        rates[6] = 2.32e-3 * T9 ** (-2.0 / 3.0) * np.exp(-10.262 * T9 ** (-1.0 / 3.0)) * f_Be
        rates[6] *= (1.0 + 0.049 * T9 ** (1.0 / 3.0) + 0.213 * T9 ** (2.0 / 3.0)
                     + 0.028 * T9 + 0.019 * T9 ** (4.0 / 3.0))


        rates[7] = 0.9


        rates[8] = 1e16

        return rates

    def cno_cycle_rates(self, T: float, rho: float, X: float, Y: float,
                        Z_metal: float) -> np.ndarray:
        if T <= 1e6 or rho <= 0 or X <= 0:
            return np.zeros(self.num_reactions, dtype=np.float64)
        T9 = T / 1e9
        rates = np.zeros(self.num_reactions, dtype=np.float64)


        f_C = self.screening_factor(rho, T, X, Y, Z_metal, 1, 6)
        rates[9] = 2.04e7 * T9 ** (-2.0 / 3.0) * np.exp(-13.690 * T9 ** (-1.0 / 3.0)) * f_C
        rates[9] *= (1.0 + 0.03 * T9 ** (1.0 / 3.0) + 0.94 * T9 ** (2.0 / 3.0)
                     + 0.058 * T9 + 0.022 * T9 ** (4.0 / 3.0))


        rates[10] = 1.16e-3


        f_C13 = self.screening_factor(rho, T, X, Y, Z_metal, 1, 6)
        rates[11] = 8.01e7 * T9 ** (-2.0 / 3.0) * np.exp(-13.717 * T9 ** (-1.0 / 3.0)) * f_C13
        rates[11] *= (1.0 + 0.03 * T9 ** (1.0 / 3.0) + 0.94 * T9 ** (2.0 / 3.0)
                      + 0.058 * T9 + 0.022 * T9 ** (4.0 / 3.0))


        f_N = self.screening_factor(rho, T, X, Y, Z_metal, 1, 7)
        rates[12] = 4.90e7 * T9 ** (-2.0 / 3.0) * np.exp(-15.228 * T9 ** (-1.0 / 3.0)) * f_N
        rates[12] *= (1.0 + 0.027 * T9 ** (1.0 / 3.0) - 0.778 * T9 ** (2.0 / 3.0)
                      - 0.149 * T9 + 0.261 * T9 ** (4.0 / 3.0) + 0.127 * T9 ** (5.0 / 3.0))


        rates[13] = 5.68e-3


        f_N15 = self.screening_factor(rho, T, X, Y, Z_metal, 1, 7)
        rates[14] = 1.08e12 * T9 ** (-2.0 / 3.0) * np.exp(-15.251 * T9 ** (-1.0 / 3.0)) * f_N15
        rates[14] *= (1.0 + 0.027 * T9 ** (1.0 / 3.0) + 0.162 * T9 ** (2.0 / 3.0)
                      + 0.010 * T9 + 0.006 * T9 ** (4.0 / 3.0))

        return rates

    def triple_alpha_rate(self, T: float, rho: float, Y: float) -> float:
        if T <= 1e7 or rho <= 0 or Y <= 0:
            return 0.0
        T9 = T / 1e9



        f_3a = self.screening_factor(rho, T, 0.0, Y, 0.0, 2, 2)
        if T9 < 0.1:
            rate = 2.79e-8 * T9 ** (-3.0) * np.exp(-4.4027 / T9) * f_3a
        else:
            rate = (2.79e-8 * T9 ** (-3.0) * np.exp(-4.4027 / T9)
                    + 1.36e-7 * T9 ** (-3.0) * np.exp(-13.490 / T9)
                    + 2.60e-8 * T9 ** (-3.0) * np.exp(-15.541 / T9)) * f_3a
        return max(rate, 0.0)

    def branch_ratios(self, T: float, rho: float, X: float, Y: float,
                      Z_metal: float) -> Tuple[float, float, float]:
        rates = self.pp_chain_rates(T, rho, X, Y, Z_metal)
        r33 = rates[2]
        r34 = rates[3]
        r7e = rates[4]
        r7p = rates[6]

        denom = r33 + r34
        if denom <= 0:
            return 1.0, 0.0, 0.0

        r_I = r33 / denom
        denom2 = r7e + r7p
        if denom2 <= 0:
            r_II = 0.0
            r_III = 0.0
        else:
            r_II = r34 * r7e / (denom * denom2)
            r_III = r34 * r7p / (denom * denom2)


        total = r_I + r_II + r_III
        if total > 0:
            r_I /= total
            r_II /= total
            r_III /= total
        return r_I, r_II, r_III

    def energy_generation(self, T: float, rho: float, X: float, Y: float,
                          Z_cno: float, Z_metal: float) -> float:



        raise NotImplementedError("Hole 1: 待实现 energy_generation 核能源产生率公式")
