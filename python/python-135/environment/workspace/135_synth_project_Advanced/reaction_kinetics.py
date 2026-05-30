
import numpy as np
from utils import (
    arrhenius_rate, validate_positive, clip_concentration,
    R_GAS, STANDARD_TEMP
)


class AmineKinetics:

    def __init__(self, amine_type="MEA"):
        self.amine_type = amine_type.upper()

        if self.amine_type == "MEA":

            self.k2_A = 4.4e11
            self.k2_Ea = 44158.0
            self.pKa = 9.5
            self.dH_pKa = -30.0e3
            self.MW = 61.08
            self.rho_pure = 1012.0
        elif self.amine_type == "MDEA":

            self.k2_A = 2.3e10
            self.k2_Ea = 41840.0
            self.pKa = 8.5
            self.dH_pKa = -25.0e3
            self.MW = 119.16
            self.rho_pure = 1047.0
        elif self.amine_type == "PZ":

            self.k2_A = 5.4e11
            self.k2_Ea = 33500.0
            self.pKa = 9.8
            self.dH_pKa = -32.0e3
            self.MW = 86.14
            self.rho_pure = None
        else:
            raise ValueError(f"Unsupported amine type: {amine_type}")

    def k2(self, T):
        validate_positive(T, "Temperature")
        return arrhenius_rate(self.k2_A, self.k2_Ea, T)

    def pKa_T(self, T):
        validate_positive(T, "Temperature")
        return self.pKa + self.dH_pKa / (2.303 * R_GAS) * (1.0 / T - 1.0 / STANDARD_TEMP)

    def base_catalysis_contribution(self, T, amine_conc, OH_conc, H2O_conc=55.5e3):

        k_amine = self.k2(T)
        k_OH = 1.4e8 * np.exp(-13200.0 / (R_GAS * T))
        k_H2O = 2.0e3 * np.exp(-25000.0 / (R_GAS * T))
        return k_amine * amine_conc + k_OH * OH_conc + k_H2O * H2O_conc

    def reaction_rate(self, T, c_CO2, c_amine, c_OH=0.0):
        validate_positive(T, "Temperature")
        c_CO2 = clip_concentration(c_CO2)
        c_amine = clip_concentration(c_amine)
        k2_val = self.k2(T)

        k_minus1 = k2_val * 0.01
        kB_sum = self.base_catalysis_contribution(T, c_amine, c_OH)
        enhancement = kB_sum / (k_minus1 + kB_sum)
        rate = k2_val * c_CO2 * c_amine * enhancement
        return rate

    def carbamate_hydrolysis_rate(self, T, c_carbamate, c_H2O=55.5e3):
        k_h_A = 1.0e9
        k_h_Ea = 55000.0
        k_h = arrhenius_rate(k_h_A, k_h_Ea, T)
        return k_h * c_carbamate * c_H2O


class CO2LoadingCalculator:

    def __init__(self, amine_kinetics):
        self.kin = amine_kinetics

    def equilibrium_loading(self, T, P_CO2, c_amine_total):
        validate_positive(T, "Temperature")
        validate_positive(P_CO2, "CO2 partial pressure")
        validate_positive(c_amine_total, "Total amine concentration")

        H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / STANDARD_TEMP))


        K_eq = np.exp(31.8 - 6660.0 / T)
        sqrt_term = np.sqrt(K_eq * P_CO2 / H_CO2)
        alpha_eq = sqrt_term / (1.0 + 2.0 * sqrt_term)
        return np.clip(alpha_eq, 0.0, 0.55)

    def kinetic_loading_estimate(self, T, P_CO2, c_amine_total, contact_time):
        alpha_eq = self.equilibrium_loading(T, P_CO2, c_amine_total)

        k_ov = self.kin.k2(T) * c_amine_total
        approach = 1.0 - np.exp(-k_ov * contact_time)
        return alpha_eq * approach


def simulate_batch_absorption(T, P_CO2, c_amine0, t_span, amine_type="MEA", n_steps=1000):
    validate_positive(T, "Temperature")
    validate_positive(P_CO2, "CO2 partial pressure")
    validate_positive(c_amine0, "Initial amine concentration")

    kin = AmineKinetics(amine_type)
    H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / STANDARD_TEMP))
    k_La = 0.02

    dt = (t_span[1] - t_span[0]) / n_steps
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)


    y = np.zeros((n_steps + 1, 4))

    c_CO2_0 = P_CO2 / H_CO2
    y[0, :] = np.array([c_CO2_0, c_amine0, 0.0, 0.0])

    def rhs(t, state):
        c_co2, c_amine, c_carb, c_ammonium = state
        c_co2 = clip_concentration(c_co2)
        c_amine = clip_concentration(c_amine)
        r_rxn = kin.reaction_rate(T, c_co2, c_amine)
        r_hydr = kin.carbamate_hydrolysis_rate(T, c_carb)

        d_co2 = k_La * (P_CO2 / H_CO2 - c_co2) - r_rxn

        d_amine = -2.0 * r_rxn + r_hydr
        d_carb = r_rxn - r_hydr
        d_amm = r_rxn + r_hydr
        return np.array([d_co2, d_amine, d_carb, d_amm])


    from ode_integrators import bdf3_solver
    t, y = bdf3_solver(rhs, t_span, y[0, :], n_steps)

    return t, y
