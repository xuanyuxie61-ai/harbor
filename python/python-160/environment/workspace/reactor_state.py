
import math
import numpy as np


class ReactorStateVector:

    def __init__(self, T=298.0, P=101325.0, n_species=7):
        self.T = float(T)
        self.P = float(P)
        self.y = np.zeros(n_species, dtype=float)
        self.y[0] = 0.21
        self.y[1] = 0.79
        self.X_biomass = 0.0
        self.X_char = 0.0
        self.z_position = 0.0

    def copy(self):
        new_state = ReactorStateVector(self.T, self.P, len(self.y))
        new_state.y = self.y.copy()
        new_state.X_biomass = self.X_biomass
        new_state.X_char = self.X_char
        new_state.z_position = self.z_position
        return new_state

    def norm(self):
        vec = np.array([self.T, self.P] + list(self.y) +
                       [self.X_biomass, self.X_char, self.z_position])
        return np.linalg.norm(vec)

    def difference(self, other):
        vec1 = np.array([self.T, self.P] + list(self.y) +
                        [self.X_biomass, self.X_char, self.z_position])
        vec2 = np.array([other.T, other.P] + list(other.y) +
                        [other.X_biomass, other.X_char, other.z_position])
        return vec1 - vec2

    def relative_difference(self, other):
        diff = self.difference(other)
        vec = np.array([self.T, self.P] + list(self.y) +
                       [self.X_biomass, self.X_char, self.z_position])
        denom = np.abs(vec)
        denom[denom < 1.0e-15] = 1.0e-15
        rel = np.abs(diff) / denom
        return float(np.max(rel))


class SequentialModularSimulator:

    def __init__(self, n_species=7, max_iter=50, tol=1.0e-6):
        self.n_species = n_species
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def simulate(self, initial_state, zone_functions, recycle_fraction=0.05):
        state = initial_state.copy()
        self.history = [state.copy()]
        feed_state = initial_state.copy()

        for it in range(self.max_iter):
            new_state = state.copy()
            for func in zone_functions:
                new_state = func(new_state)


            if recycle_fraction > 0.0:
                mixed = feed_state.copy()
                mixed.T = (1.0 - recycle_fraction) * feed_state.T + recycle_fraction * new_state.T
                mixed.P = feed_state.P
                mixed.y = (1.0 - recycle_fraction) * feed_state.y + recycle_fraction * new_state.y
                mixed.y = np.maximum(mixed.y, 0.0)
                y_sum = mixed.y.sum()
                if y_sum > 1.0e-15:
                    mixed.y = mixed.y / y_sum
                new_state = mixed

            self.history.append(new_state.copy())
            rel_diff = state.relative_difference(new_state)

            if rel_diff < self.tol:
                return new_state, True, it + 1


            alpha = 0.5
            state.T = alpha * new_state.T + (1.0 - alpha) * state.T
            state.P = alpha * new_state.P + (1.0 - alpha) * state.P
            state.y = alpha * new_state.y + (1.0 - alpha) * state.y
            state.y = np.maximum(state.y, 0.0)
            y_sum = state.y.sum()
            if y_sum > 1.0e-15:
                state.y = state.y / y_sum
            state.X_biomass = alpha * new_state.X_biomass + (1.0 - alpha) * state.X_biomass
            state.X_char = alpha * new_state.X_char + (1.0 - alpha) * state.X_char
            state.X_biomass = np.clip(state.X_biomass, 0.0, 1.0)
            state.X_char = np.clip(state.X_char, 0.0, 1.0)

        return state, False, self.max_iter


class ReactorZoneModel:

    @staticmethod
    def drying_zone(state, moisture_content=0.15, T_boil=373.15,
                    heat_of_vaporization=2.26e6):
        new_state = state.copy()

        new_state.T = T_boil

        new_state.y[2] += moisture_content * 0.5
        new_state.y[0] = max(new_state.y[0] - moisture_content * 0.1, 0.0)
        new_state.y[1] = max(new_state.y[1] - moisture_content * 0.05, 0.0)

        y_sum = new_state.y.sum()
        if y_sum > 1.0e-15:
            new_state.y = new_state.y / y_sum
        new_state.X_biomass = min(new_state.X_biomass + 0.2, 1.0)
        return new_state

    @staticmethod
    def pyrolysis_zone(state, T_pyro=673.0, tar_yield=0.15,
                       char_yield=0.25, gas_yield=0.60):
        new_state = state.copy()
        new_state.T = T_pyro

        add_gas = gas_yield * 0.3
        new_state.y[3] = min(new_state.y[3] + add_gas * 0.30, 0.6)
        new_state.y[4] = min(new_state.y[4] + add_gas * 0.20, 0.5)
        new_state.y[5] = min(new_state.y[5] + add_gas * 0.35, 0.6)
        new_state.y[6] = min(new_state.y[6] + add_gas * 0.15, 0.4)
        new_state.y[2] = max(new_state.y[2] - add_gas * 0.2, 0.0)
        y_sum = new_state.y.sum()
        if y_sum > 1.0e-15:
            new_state.y = new_state.y / y_sum
        new_state.X_char = min(new_state.X_char + char_yield, 1.0)
        new_state.X_biomass = max(new_state.X_biomass - 0.3, 0.0)
        return new_state

    @staticmethod
    def combustion_zone(state, T_comb=1273.0, equivalence_ratio=0.3):
        new_state = state.copy()
        new_state.T = T_comb

        o2_consumed = new_state.y[0] * equivalence_ratio
        new_state.y[0] = max(new_state.y[0] - o2_consumed, 0.0)
        new_state.y[3] = min(new_state.y[3] + o2_consumed * 1.8, 0.7)
        new_state.y[4] = min(new_state.y[4] + o2_consumed * 0.2, 0.5)
        y_sum = new_state.y.sum()
        if y_sum > 1.0e-15:
            new_state.y = new_state.y / y_sum
        new_state.X_char = max(new_state.X_char * 0.5, 0.0)
        return new_state

    @staticmethod
    def reduction_zone(state, T_red=1173.0, steam_ratio=1.0):
        new_state = state.copy()
        new_state.T = T_red

        h2o_consumed = new_state.y[2] * 0.6 * steam_ratio
        co2_consumed = new_state.y[4] * 0.4
        new_state.y[2] = max(new_state.y[2] - h2o_consumed, 0.0)
        new_state.y[4] = max(new_state.y[4] - co2_consumed, 0.0)
        new_state.y[3] = min(new_state.y[3] + h2o_consumed + 2.0 * co2_consumed, 0.8)
        new_state.y[5] = min(new_state.y[5] + h2o_consumed, 0.7)
        y_sum = new_state.y.sum()
        if y_sum > 1.0e-15:
            new_state.y = new_state.y / y_sum
        new_state.X_char = max(new_state.X_char * 0.1, 0.0)
        return new_state


class ConvergenceMonitor:

    def __init__(self):
        self.residuals = []
        self.temperatures = []
        self.conversions = []

    def record(self, state, residual):
        self.residuals.append(float(residual))
        self.temperatures.append(float(state.T))
        self.conversions.append(float(state.X_char))

    def convergence_rate(self):
        if len(self.residuals) < 3:
            return 1.0
        rates = []
        for i in range(1, len(self.residuals)):
            if abs(self.residuals[i - 1]) > 1.0e-15:
                rates.append(self.residuals[i] / self.residuals[i - 1])
        if not rates:
            return 1.0
        return float(np.median(rates))

    def estimated_iterations_to_convergence(self, tol=1.0e-6):
        if len(self.residuals) == 0:
            return 0
        r = self.convergence_rate()
        if r >= 1.0 or r < 1.0e-15:
            return 1000
        current = self.residuals[-1]
        if current <= tol:
            return 0
        return int(math.ceil(math.log(tol / current) / math.log(r)))
