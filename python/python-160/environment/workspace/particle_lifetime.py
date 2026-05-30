
import math
import numpy as np


class ParticleBurnoutModel:

    def __init__(self, d0=5.0e-3, rho_char=800.0, k_surf=0.01,
                 D_eff=1.0e-5, control='mixed'):
        self.d0 = float(d0)
        self.r0 = d0 / 2.0
        self.rho_char = float(rho_char)
        self.k_surf = float(k_surf)
        self.D_eff = float(D_eff)
        self.control = control

    def conversion_rate(self, X, T):
        if X <= 0.0:
            X = 0.0
        if X >= 1.0:
            return 0.0


        k_T = self.k_surf * math.exp(-80000.0 / (8.314 * T))

        if self.control == 'chemical':

            return k_T * (1.0 - X) ** (2.0 / 3.0)
        elif self.control == 'diffusion':

            return k_T * (1.0 - X) ** (-1.0 / 3.0)
        elif self.control == 'mixed':

            core_term = 1.0 - (1.0 - X) ** (1.0 / 3.0)
            if abs(core_term) < 1.0e-15:
                return k_T * 1.0e6
            return k_T * (1.0 - X) ** (2.0 / 3.0) / core_term
        else:
            return k_T * (1.0 - X)

    def integrate_conversion(self, T_profile, dt, max_iter=10000):
        X = 0.0
        t = 0.0
        history = [(t, X)]

        for _ in range(max_iter):
            if X >= 0.999:
                break

            T = self._interp_temperature(t, T_profile)
            dXdt = self.conversion_rate(X, T)
            X_new = X + dXdt * dt
            if X_new > 1.0:
                X_new = 1.0
            if abs(X_new - X) < 1.0e-12:
                break
            X = X_new
            t += dt
            history.append((t, X))

        return np.array(history)

    def _interp_temperature(self, t, T_profile):
        if not T_profile:
            return 1000.0
        times = np.array([tp[0] for tp in T_profile])
        temps = np.array([tp[1] for tp in T_profile])
        return np.interp(t, times, temps)

    def burnout_time(self, T_const, C_gas=10.0, tol=1.0e-6):
        if T_const <= 0.0 or self.k_surf <= 0.0 or C_gas <= 0.0:
            return 0.0

        k_s = self.k_surf * math.exp(-80000.0 / (8.314 * T_const))
        if abs(k_s) < 1.0e-15:
            return 1.0e15

        tau = self.rho_char * self.r0 / (k_s * C_gas)
        if self.control == 'chemical':
            return tau
        elif self.control == 'diffusion':

            return self.rho_char * self.r0 ** 2 / (6.0 * self.D_eff * C_gas)
        else:

            tau_diff = self.rho_char * self.r0 ** 2 / (6.0 * self.D_eff * C_gas)
            return tau + tau_diff


class ParticleMortalityTable:

    def __init__(self, max_age_seconds=3600.0, n_bins=60):
        self.max_age = float(max_age_seconds)
        self.n_bins = int(n_bins)
        self.bin_edges = np.linspace(0.0, self.max_age, n_bins + 1)
        self.counts = np.zeros(n_bins, dtype=float)
        self.survival = np.ones(n_bins, dtype=float)

    def populate_from_weibull(self, scale, shape):
        for i in range(self.n_bins):
            t_mid = 0.5 * (self.bin_edges[i] + self.bin_edges[i + 1])
            dt = self.bin_edges[i + 1] - self.bin_edges[i]
            if scale <= 0.0 or shape <= 0.0:
                self.counts[i] = 0.0
            else:
                pdf = (shape / scale) * (t_mid / scale) ** (shape - 1.0) * \
                      math.exp(-(t_mid / scale) ** shape)
                self.counts[i] = pdf * dt

        total = self.counts.sum()
        if total > 1.0e-15:
            self.counts = self.counts / total

        self.survival = 1.0 - np.cumsum(self.counts)
        self.survival = np.clip(self.survival, 0.0, 1.0)

    def expected_lifetime(self):
        expected = 0.0
        for i in range(self.n_bins):
            t_mid = 0.5 * (self.bin_edges[i] + self.bin_edges[i + 1])
            expected += t_mid * self.counts[i]
        return expected

    def median_lifetime(self):
        for i in range(self.n_bins):
            if self.survival[i] <= 0.5:

                if i == 0:
                    return 0.0
                t0 = self.bin_edges[i - 1]
                t1 = self.bin_edges[i]
                s0 = self.survival[i - 1]
                s1 = self.survival[i]
                if abs(s0 - s1) > 1.0e-15:
                    return t0 + (t1 - t0) * (0.5 - s0) / (s1 - s0)
                return t0
        return self.max_age

    def hazard_rate(self):
        hazard = np.zeros(self.n_bins, dtype=float)
        for i in range(self.n_bins):
            if self.survival[i] > 1.0e-15:
                hazard[i] = self.counts[i] / self.survival[i]
            else:
                hazard[i] = 0.0
        return hazard

    def remaining_life_expectancy(self, current_age):

        idx = int(current_age / self.max_age * self.n_bins)
        idx = min(idx, self.n_bins - 1)
        if self.survival[idx] < 1.0e-15:
            return 0.0
        remaining = 0.0
        for i in range(idx, self.n_bins):
            dt = self.bin_edges[i + 1] - self.bin_edges[i]
            remaining += self.survival[i] * dt
        return remaining / self.survival[idx]


class CollatzBurnoutSequence:

    @staticmethod
    def generate_sequence(X0, T, k, max_steps=1000, threshold=0.999):
        sequence = [float(X0)]
        X = float(X0)
        for _ in range(max_steps):
            if X >= threshold:
                break
            dX = k * (1.0 - X) ** (2.0 / 3.0)
            if dX < 1.0e-12:
                break
            X = min(X + dX, 1.0)
            sequence.append(X)
        return np.array(sequence)

    @staticmethod
    def sequence_statistics(sequences):
        lengths = [len(s) for s in sequences]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return {'mean_length': 0.0, 'max_length': 0}


        padded = []
        for s in sequences:
            arr = np.zeros(max_len, dtype=float)
            arr[0:len(s)] = s
            if len(s) < max_len:
                arr[len(s):] = 1.0
            padded.append(arr)

        mean_seq = np.mean(padded, axis=0)
        return {
            'mean_length': float(np.mean(lengths)),
            'max_length': max_len,
            'std_length': float(np.std(lengths)),
            'mean_sequence': mean_seq
        }
