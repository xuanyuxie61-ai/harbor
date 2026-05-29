"""
Amine Regenerator (Stripper) Model
Models the thermal regeneration of CO2-rich amine solvent.
This is the energy-intensive step in post-combustion CO2 capture.

Key models:
- Equilibrium-stage model for stripper
- Rate-based model for desorption kinetics
- Energy balance for reboiler duty calculation
- Cycle time analysis (predator-prey-like periodic dynamics concept)
"""

import numpy as np
from utils import validate_positive, R_GAS, STANDARD_TEMP, clip_concentration
from ode_integrators import explicit_trapezoidal, bdf3_solver


class StripperModel:
    """
    Rate-based stripper model for CO2 desorption from rich amine.
    """

    def __init__(self, T_reboiler, P_stripper, n_stages=10):
        self.T_reb = T_reboiler
        self.P_strip = P_stripper
        self.n_stages = n_stages
        # Henry's constant [Pa·m^3/mol]
        self.H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T_reboiler - 1.0 / STANDARD_TEMP))

    def equilibrium_CO2_loading(self, T, P_CO2, c_amine):
        """
        Equilibrium CO2 loading at stripper conditions (high T, low P).
        alpha_eq = f(T, P_CO2)
        """
        validate_positive(T, "Temperature")
        validate_positive(P_CO2, "CO2 partial pressure")

        # Simplified correlation
        K_eq = 2.5e7 * np.exp(-52000.0 / (R_GAS * T))
        alpha_eq = np.sqrt(P_CO2 / (self.H_CO2 * K_eq * c_amine + 1e-20))
        return np.clip(alpha_eq, 0.0, 0.55)

    def desorption_rate(self, T, alpha, alpha_eq, c_amine, k_desorb=0.1):
        """
        Desorption rate driven by departure from equilibrium:
            r_desorb = k * (alpha - alpha_eq) * c_amine
        """
        validate_positive(T, "Temperature")
        driving_force = alpha - alpha_eq
        return k_desorb * driving_force * c_amine

    def stage_model(self, alpha_in, T_stage, P_CO2_stage, c_amine, L_flow, dt=1.0):
        """
        Single equilibrium/rate stage model.
        """
        alpha_eq = self.equilibrium_CO2_loading(T_stage, P_CO2_stage, c_amine)
        r_des = self.desorption_rate(T_stage, alpha_in, alpha_eq, c_amine)
        # Mass balance on stage
        dalpha = -r_des * dt / c_amine
        alpha_out = alpha_in + dalpha
        CO2_desorbed = r_des * dt
        return np.clip(alpha_out, 0.01, 0.55), CO2_desorbed

    def simulate_column(self, alpha_rich, T_profile, c_amine, L_flow, n_steps=100):
        """
        Simulate stripper column with temperature profile.
        """
        alphas = np.zeros(n_steps + 1)
        co2_flows = np.zeros(n_steps + 1)
        alphas[0] = alpha_rich

        for i in range(n_steps):
            T = T_profile[i] if i < len(T_profile) else T_profile[-1]
            # CO2 partial pressure decreases up the column
            P_CO2 = self.P_strip * (1.0 - 0.8 * i / n_steps)
            alpha_out, co2 = self.stage_model(
                alphas[i], T, P_CO2, c_amine, L_flow
            )
            alphas[i + 1] = alpha_out
            co2_flows[i + 1] = co2_flows[i] + co2

        return alphas, co2_flows

    def reboiler_duty(self, alpha_rich, alpha_lean, c_amine, T_reb, T_feed):
        """
        Calculate reboiler duty [kJ/kmol CO2].
        Components:
        1. Sensible heat to raise temperature
        2. Heat of reaction for desorption
        3. Heat to generate steam for stripping
        """
        validate_positive(T_reb, "Reboiler temperature")

        # Sensible heat [J/mol amine]
        Cp_amine = 150.0  # J/(mol·K)  (simplified)
        Q_sensible = Cp_amine * (T_reb - T_feed)

        # Heat of reaction [J/mol CO2]
        delta_H_rxn = 85.0e3  # Endothermic desorption
        Q_reaction = delta_H_rxn * (alpha_rich - alpha_lean)

        # Steam generation (latent heat)
        lambda_steam = 40.7e3  # J/mol at 100C
        steam_ratio = 1.2  # mol steam / mol CO2
        Q_steam = lambda_steam * steam_ratio

        total_duty = (Q_sensible + Q_reaction + Q_steam) / (alpha_rich - alpha_lean)
        return total_duty / 1000.0  # kJ/kmol CO2


class CyclicAbsorptionRegeneration:
    """
    Model cyclic operation with periodic absorption-regeneration.
    Inspired by predator_prey_ode_period periodic dynamics.
    """

    def __init__(self, absorber_params, stripper_params, cycle_time):
        self.abs_params = absorber_params
        self.strip_params = stripper_params
        self.t_cycle = cycle_time

    def cycle_dynamics(self, alpha_lean0, n_cycles=5, n_points=500):
        """
        Simulate cyclic dynamics:
            Absorption phase: alpha increases
            Regeneration phase: alpha decreases
        Returns periodic orbit characteristics.
        """
        t_total = n_cycles * self.t_cycle
        t = np.linspace(0, t_total, n_points)
        dt = t[1] - t[0]

        alpha = np.zeros(n_points)
        alpha[0] = alpha_lean0

        # Phase durations
        t_abs = self.t_cycle * 0.6  # 60% absorption

        k_abs = 0.05   # pseudo rate constants
        k_reg = 0.08
        alpha_eq_abs = 0.45
        alpha_eq_reg = 0.15

        for i in range(n_points - 1):
            t_mod = t[i] % self.t_cycle
            if t_mod < t_abs:
                # Absorption: dalpha/dt = k_abs * (alpha_eq_abs - alpha)
                dalpha = k_abs * (alpha_eq_abs - alpha[i])
            else:
                # Regeneration: dalpha/dt = -k_reg * (alpha[i] - alpha_eq_reg)
                dalpha = -k_reg * (alpha[i] - alpha_eq_reg)
            alpha[i + 1] = alpha[i] + dalpha * dt

        # Analyze periodicity
        peaks = []
        for i in range(1, n_points - 1):
            if alpha[i] > alpha[i - 1] and alpha[i] > alpha[i + 1]:
                peaks.append(t[i])

        periods = []
        for i in range(1, len(peaks)):
            periods.append(peaks[i] - peaks[i - 1])

        return {
            "t": t,
            "alpha": alpha,
            "periods": periods,
            "mean_period": np.mean(periods) if periods else None,
            "amplitude": np.max(alpha) - np.min(alpha)
        }


def simulate_rich_amine_flash(T_flash, P_flash, alpha_rich, c_amine):
    """
    Flash calculation for rich amine entering stripper.
    Returns flashed CO2 and updated loading.
    """
    validate_positive(T_flash, "Flash temperature")
    validate_positive(P_flash, "Flash pressure")

    H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T_flash - 1.0 / STANDARD_TEMP))
    K_eq = 2.5e7 * np.exp(-52000.0 / (R_GAS * T_flash))

    # Equilibrium CO2 concentration
    c_CO2_eq = P_flash / H_CO2
    # Loading after flash
    alpha_flash = c_CO2_eq * c_amine / (K_eq * c_amine ** 2 + 1e-20)
    alpha_flash = np.clip(alpha_flash, 0.0, alpha_rich)

    CO2_released = (alpha_rich - alpha_flash) * c_amine
    return alpha_flash, CO2_released


def energy_integration_analysis(Q_reboiler, Q_condenser, T_reb, T_cond):
    """
    Pinch analysis for heat integration in capture plant.
    Computes minimum utility requirements.
    """
    validate_positive(Q_reboiler, "Reboiler duty")
    validate_positive(Q_condenser, "Condenser duty")

    # Grand composite curve (simplified)
    # Hot composite: reboiler at T_reb
    # Cold composite: condenser at T_cond
    delta_T_min = 10.0  # K

    Q_recoverable = min(Q_reboiler, Q_condenser)
    pinch_T = T_cond + delta_T_min

    return {
        "reboiler_duty": Q_reboiler,
        "condenser_duty": Q_condenser,
        "recoverable_heat": Q_recoverable,
        "net_heat_demand": Q_reboiler - Q_recoverable,
        "pinch_temperature": pinch_T,
        "thermal_efficiency": Q_recoverable / Q_reboiler if Q_reboiler > 0 else 0.0
    }
