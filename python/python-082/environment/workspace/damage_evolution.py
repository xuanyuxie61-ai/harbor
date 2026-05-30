
import numpy as np


class CyclicDamageModel:

    def __init__(self, epsilon=0.001, a_param=0.81, gamma=0.45,
                 omega=7.85, C_paris=1e-10, m_paris=3.5,
                 K_c=30.0e6, sigma_max=100.0e6, Y_geom=1.12):
        self.epsilon = epsilon
        self.a_param = a_param
        self.gamma = gamma
        self.omega = omega
        self.C_paris = C_paris
        self.m_paris = m_paris
        self.K_c = K_c
        self.sigma_max = sigma_max
        self.Y_geom = Y_geom

    def derivatives(self, t, y):
        d_f, d_m = y[0], y[1]

        d_f = np.clip(d_f, 0.0, 0.999)
        d_m = np.clip(d_m, 0.0, 0.999)


        dd_f = -(1.0 / self.epsilon) * (d_f ** 3 - self.a_param * d_f + d_m)
        dd_m = d_f - self.gamma + 0.1 * np.cos(self.omega * t)

        return np.array([dd_f, dd_m])

    def rk4_integrate(self, y0, t0, tstop, n_steps=10000):
        y0 = np.asarray(y0, dtype=float)
        dt = (tstop - t0) / n_steps
        t_array = np.linspace(t0, tstop, n_steps + 1)
        y_array = np.zeros((n_steps + 1, len(y0)))
        y_array[0] = y0

        for i in range(n_steps):
            t = t_array[i]
            y = y_array[i]
            k1 = self.derivatives(t, y)
            k2 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.derivatives(t + dt, y + dt * k3)
            y_array[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            y_array[i + 1] = np.clip(y_array[i + 1], 0.0, 1.0)

        return t_array, y_array

    def paris_law_cycles(self, a0, a_crit, n_cycles=100000):
        a = float(a0)
        crack_lengths = [a]
        cycles = [0]

        for N in range(1, n_cycles + 1):
            if a >= a_crit:
                break
            delta_K = self.Y_geom * self.sigma_max * np.sqrt(np.pi * a)
            da_dN = self.C_paris * (delta_K / self.K_c) ** self.m_paris
            a += da_dN
            cycles.append(N)
            crack_lengths.append(a)

        life = len(cycles) - 1
        return np.array(cycles), np.array(crack_lengths), life

    def vanderpol_period_estimate(self, mu):
        if mu == 0.0:
            return 2.0 * np.pi

        alpha = 2.338107
        b0 = 0.1723
        d = 0.4889

        T = ((3.0 - 2.0 * np.log(2.0)) * mu
             + 3.0 * alpha / (mu ** (1.0 / 3.0))
             - (1.0 / 3.0) * np.log(mu) / mu
             + (3.0 * np.log(2.0) - np.log(3.0) - 1.5 + b0 - 2.0 * d) / mu)
        return T

    def hysteresis_energy_per_cycle(self, stress_amplitude, strain_amplitude,
                                    n_points=100):
        theta = np.linspace(0, 2.0 * np.pi, n_points)

        phi = 0.15
        sigma = stress_amplitude * np.sin(theta)
        epsilon = strain_amplitude * np.sin(theta - phi)
        de = np.gradient(epsilon)
        energy = np.trapezoid(sigma, epsilon)
        return abs(energy)


def cumulative_damage_miner(stress_history, S_n_curve, N_f_curve):
    D = 0.0
    for S, n in stress_history:
        N_f = S_n_curve(S)
        if N_f <= 0:
            continue
        D += n / N_f
    return D
