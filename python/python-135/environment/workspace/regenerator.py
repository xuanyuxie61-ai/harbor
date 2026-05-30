
import numpy as np
from utils import validate_positive, R_GAS, STANDARD_TEMP, clip_concentration
from ode_integrators import explicit_trapezoidal, bdf3_solver


class StripperModel:

    def __init__(self, T_reboiler, P_stripper, n_stages=10):
        self.T_reb = T_reboiler
        self.P_strip = P_stripper
        self.n_stages = n_stages

        self.H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T_reboiler - 1.0 / STANDARD_TEMP))

    def equilibrium_CO2_loading(self, T, P_CO2, c_amine):
        validate_positive(T, "Temperature")
        validate_positive(P_CO2, "CO2 partial pressure")


        K_eq = 2.5e7 * np.exp(-52000.0 / (R_GAS * T))
        alpha_eq = np.sqrt(P_CO2 / (self.H_CO2 * K_eq * c_amine + 1e-20))
        return np.clip(alpha_eq, 0.0, 0.55)

    def desorption_rate(self, T, alpha, alpha_eq, c_amine, k_desorb=0.1):
        validate_positive(T, "Temperature")
        driving_force = alpha - alpha_eq
        return k_desorb * driving_force * c_amine

    def stage_model(self, alpha_in, T_stage, P_CO2_stage, c_amine, L_flow, dt=1.0):
        alpha_eq = self.equilibrium_CO2_loading(T_stage, P_CO2_stage, c_amine)
        r_des = self.desorption_rate(T_stage, alpha_in, alpha_eq, c_amine)

        dalpha = -r_des * dt / c_amine
        alpha_out = alpha_in + dalpha
        CO2_desorbed = r_des * dt
        return np.clip(alpha_out, 0.01, 0.55), CO2_desorbed

    def simulate_column(self, alpha_rich, T_profile, c_amine, L_flow, n_steps=100):
        alphas = np.zeros(n_steps + 1)
        co2_flows = np.zeros(n_steps + 1)
        alphas[0] = alpha_rich

        for i in range(n_steps):
            T = T_profile[i] if i < len(T_profile) else T_profile[-1]

            P_CO2 = self.P_strip * (1.0 - 0.8 * i / n_steps)
            alpha_out, co2 = self.stage_model(
                alphas[i], T, P_CO2, c_amine, L_flow
            )
            alphas[i + 1] = alpha_out
            co2_flows[i + 1] = co2_flows[i] + co2

        return alphas, co2_flows

    def reboiler_duty(self, alpha_rich, alpha_lean, c_amine, T_reb, T_feed):
        validate_positive(T_reb, "Reboiler temperature")


        Cp_amine = 150.0
        Q_sensible = Cp_amine * (T_reb - T_feed)


        delta_H_rxn = 85.0e3
        Q_reaction = delta_H_rxn * (alpha_rich - alpha_lean)


        lambda_steam = 40.7e3
        steam_ratio = 1.2
        Q_steam = lambda_steam * steam_ratio

        total_duty = (Q_sensible + Q_reaction + Q_steam) / (alpha_rich - alpha_lean)
        return total_duty / 1000.0


class CyclicAbsorptionRegeneration:

    def __init__(self, absorber_params, stripper_params, cycle_time):
        self.abs_params = absorber_params
        self.strip_params = stripper_params
        self.t_cycle = cycle_time

    def cycle_dynamics(self, alpha_lean0, n_cycles=5, n_points=500):
        t_total = n_cycles * self.t_cycle
        t = np.linspace(0, t_total, n_points)
        dt = t[1] - t[0]

        alpha = np.zeros(n_points)
        alpha[0] = alpha_lean0


        t_abs = self.t_cycle * 0.6

        k_abs = 0.05
        k_reg = 0.08
        alpha_eq_abs = 0.45
        alpha_eq_reg = 0.15

        for i in range(n_points - 1):
            t_mod = t[i] % self.t_cycle
            if t_mod < t_abs:

                dalpha = k_abs * (alpha_eq_abs - alpha[i])
            else:

                dalpha = -k_reg * (alpha[i] - alpha_eq_reg)
            alpha[i + 1] = alpha[i] + dalpha * dt


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
    validate_positive(T_flash, "Flash temperature")
    validate_positive(P_flash, "Flash pressure")

    H_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T_flash - 1.0 / STANDARD_TEMP))
    K_eq = 2.5e7 * np.exp(-52000.0 / (R_GAS * T_flash))


    c_CO2_eq = P_flash / H_CO2

    alpha_flash = c_CO2_eq * c_amine / (K_eq * c_amine ** 2 + 1e-20)
    alpha_flash = np.clip(alpha_flash, 0.0, alpha_rich)

    CO2_released = (alpha_rich - alpha_flash) * c_amine
    return alpha_flash, CO2_released


def energy_integration_analysis(Q_reboiler, Q_condenser, T_reb, T_cond):
    validate_positive(Q_reboiler, "Reboiler duty")
    validate_positive(Q_condenser, "Condenser duty")




    delta_T_min = 10.0

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
