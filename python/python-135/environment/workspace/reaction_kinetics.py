"""
CO2-Amine Reaction Kinetics Module
Implements the zwitterion mechanism and base-catalyzed hydration for CO2
absorption into aqueous amine solutions (MEA, MDEA, PZ).

Scientific basis:
- Zwitterion mechanism (Caplow, 1968; Danckwerts, 1979):
    CO2 + R1R2NH  <=>  R1R2NH+COO-   (step 1, k2/k-1)
    R1R2NH+COO- + B  ->  R1R2NCOO- + BH+   (step 2, kB)
- Overall rate: r = k2[CO2][Amine] / (1 + k-1 / sum(kB[B]))
- For primary amines (MEA), typically k-1 << kB[B], so:
    r ≈ k2[CO2][Amine]
"""

import numpy as np
from utils import (
    arrhenius_rate, validate_positive, clip_concentration,
    R_GAS, STANDARD_TEMP
)


class AmineKinetics:
    """
    Kinetic parameters for CO2 absorption into aqueous amine solutions.
    Data from literature (Versteeg et al., 1996; Aboudheir et al., 2003).
    """

    def __init__(self, amine_type="MEA"):
        self.amine_type = amine_type.upper()
        # Pre-exponential factors [m^3/(mol·s)] and activation energies [J/mol]
        if self.amine_type == "MEA":
            # MEA: monoethanolamine
            self.k2_A = 4.4e11  # m^3/(mol·s)
            self.k2_Ea = 44158.0  # J/mol
            self.pKa = 9.5  # at 298K
            self.dH_pKa = -30.0e3  # J/mol
            self.MW = 61.08  # g/mol
            self.rho_pure = 1012.0  # kg/m^3 at 298K
        elif self.amine_type == "MDEA":
            # MDEA: methyldiethanolamine
            self.k2_A = 2.3e10
            self.k2_Ea = 41840.0
            self.pKa = 8.5
            self.dH_pKa = -25.0e3
            self.MW = 119.16
            self.rho_pure = 1047.0
        elif self.amine_type == "PZ":
            # PZ: piperazine
            self.k2_A = 5.4e11
            self.k2_Ea = 33500.0
            self.pKa = 9.8
            self.dH_pKa = -32.0e3
            self.MW = 86.14
            self.rho_pure = None
        else:
            raise ValueError(f"Unsupported amine type: {amine_type}")

    def k2(self, T):
        """Second-order rate constant for CO2-amine reaction."""
        validate_positive(T, "Temperature")
        return arrhenius_rate(self.k2_A, self.k2_Ea, T)

    def pKa_T(self, T):
        """Temperature-dependent pKa using van't Hoff-like relation."""
        validate_positive(T, "Temperature")
        return self.pKa + self.dH_pKa / (2.303 * R_GAS) * (1.0 / T - 1.0 / STANDARD_TEMP)

    def base_catalysis_contribution(self, T, amine_conc, OH_conc, H2O_conc=55.5e3):
        """
        Base-catalyzed term for zwitterion deprotonation.
        Sum(kB[B]) = k_amine[Amine] + k_OH[OH-] + k_H2O[H2O]
        """
        # Relative base strengths (simplified)
        k_amine = self.k2(T)
        k_OH = 1.4e8 * np.exp(-13200.0 / (R_GAS * T))  # OH- catalysis
        k_H2O = 2.0e3 * np.exp(-25000.0 / (R_GAS * T))  # H2O catalysis
        return k_amine * amine_conc + k_OH * OH_conc + k_H2O * H2O_conc

    def reaction_rate(self, T, c_CO2, c_amine, c_OH=0.0):
        """
        Overall CO2 absorption reaction rate [mol/(m^3·s)].
        r = k2 * [CO2] * [Amine] * (kB[B] / (k-1 + kB[B]))
        For primary amines with fast deprotonation, this simplifies to k2[CO2][Amine].
        """
        validate_positive(T, "Temperature")
        c_CO2 = clip_concentration(c_CO2)
        c_amine = clip_concentration(c_amine)
        k2_val = self.k2(T)
        # Backward reaction (k-1) is typically small for primary amines
        k_minus1 = k2_val * 0.01  # Approximate ratio
        kB_sum = self.base_catalysis_contribution(T, c_amine, c_OH)
        enhancement = kB_sum / (k_minus1 + kB_sum)
        rate = k2_val * c_CO2 * c_amine * enhancement
        return rate

    def carbamate_hydrolysis_rate(self, T, c_carbamate, c_H2O=55.5e3):
        """
        Carbamate hydrolysis: RNHCOO- + H2O -> RNH2 + HCO3-
        This limits the CO2 loading capacity.
        """
        k_h_A = 1.0e9
        k_h_Ea = 55000.0
        k_h = arrhenius_rate(k_h_A, k_h_Ea, T)
        return k_h * c_carbamate * c_H2O


class CO2LoadingCalculator:
    """
    Calculate CO2 loading (alpha = mol CO2 / mol amine) from equilibrium and kinetics.
    """

    def __init__(self, amine_kinetics):
        self.kin = amine_kinetics

    def equilibrium_loading(self, T, P_CO2, c_amine_total):
        """
        Equilibrium CO2 loading using modified Kent-Eisenberg model.
        For the zwitterion mechanism (carbamate formation):
            CO2 + 2RNH2 <=> RNHCOO- + RNH3+
        Equilibrium constant:
            K_eq = [RNHCOO-][RNH3+] / ([CO2][RNH2]^2)
        At equilibrium with loading alpha:
            [RNHCOO-] = [RNH3+] = alpha * c_amine_total / (1 - alpha)
            [RNH2] = (1 - 2*alpha) * c_amine_total / (1 - alpha)
            [CO2] = P_CO2 / H_CO2
        Solving for alpha:
            K_eq = alpha^2 * (1 - alpha)^(-2) * (P_CO2/H_CO2)^(-1)
            sqrt(K_eq * P_CO2/H_CO2) = alpha / (1 - 2*alpha)
            alpha_eq = sqrt(K_eq*P_CO2/H_CO2) / (1 + 2*sqrt(K_eq*P_CO2/H_CO2))
        """
        validate_positive(T, "Temperature")
        validate_positive(P_CO2, "CO2 partial pressure")
        validate_positive(c_amine_total, "Total amine concentration")
        # Henry's law constant for CO2 in water [Pa·m^3/mol]
        H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / STANDARD_TEMP))
        # Equilibrium constant (temperature dependent) [m^3/mol]
        # Based on Kent-Eisenberg correlation: ln(K) = A - B/T
        K_eq = np.exp(31.8 - 6660.0 / T)
        sqrt_term = np.sqrt(K_eq * P_CO2 / H_CO2)
        alpha_eq = sqrt_term / (1.0 + 2.0 * sqrt_term)
        return np.clip(alpha_eq, 0.0, 0.55)  # MEA max loading ~0.5

    def kinetic_loading_estimate(self, T, P_CO2, c_amine_total, contact_time):
        """
        Estimate actual loading based on contact time (simplified film model).
        """
        alpha_eq = self.equilibrium_loading(T, P_CO2, c_amine_total)
        # Approach to equilibrium factor
        k_ov = self.kin.k2(T) * c_amine_total
        approach = 1.0 - np.exp(-k_ov * contact_time)
        return alpha_eq * approach


def simulate_batch_absorption(T, P_CO2, c_amine0, t_span, amine_type="MEA", n_steps=1000):
    """
    Simulate batch CO2 absorption into amine solution using explicit trapezoidal method.
    Based on biochemical_linear_ode and trapezoidal_explicit concepts.

    ODE system:
        d[CO2]/dt = k_L*a*(P_CO2/H - [CO2]) - r_rxn
        d[RNH2]/dt = -2*r_rxn + r_hydrolysis
        d[RNHCOO-]/dt = r_rxn - r_hydrolysis
        d[RNH3+]/dt = r_rxn + r_hydrolysis
    where r_rxn = k2[CO2][RNH2]
    """
    validate_positive(T, "Temperature")
    validate_positive(P_CO2, "CO2 partial pressure")
    validate_positive(c_amine0, "Initial amine concentration")

    kin = AmineKinetics(amine_type)
    H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / STANDARD_TEMP))
    k_La = 0.02  # Mass transfer coefficient * interfacial area [1/s]

    dt = (t_span[1] - t_span[0]) / n_steps
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)

    # State: [CO2, RNH2, RNHCOO-, RNH3+]
    y = np.zeros((n_steps + 1, 4))
    # Initial dissolved CO2 at equilibrium with gas phase
    c_CO2_0 = P_CO2 / H_CO2
    y[0, :] = np.array([c_CO2_0, c_amine0, 0.0, 0.0])

    def rhs(t, state):
        c_co2, c_amine, c_carb, c_ammonium = state
        c_co2 = clip_concentration(c_co2)
        c_amine = clip_concentration(c_amine)
        r_rxn = kin.reaction_rate(T, c_co2, c_amine)
        r_hydr = kin.carbamate_hydrolysis_rate(T, c_carb)
        # Gas-liquid mass transfer (continuous CO2 supply)
        d_co2 = k_La * (P_CO2 / H_CO2 - c_co2) - r_rxn
        # Amine consumption: 2 mol amine per mol CO2 (carbamate)
        d_amine = -2.0 * r_rxn + r_hydr
        d_carb = r_rxn - r_hydr
        d_amm = r_rxn + r_hydr
        return np.array([d_co2, d_amine, d_carb, d_amm])

    # Use BDF3 for stiff system (fast reaction)
    from ode_integrators import bdf3_solver
    t, y = bdf3_solver(rhs, t_span, y[0, :], n_steps)

    return t, y
