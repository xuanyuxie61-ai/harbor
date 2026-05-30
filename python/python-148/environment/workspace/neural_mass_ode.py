
import numpy as np
from utils import sigmoid_activation, sawtooth_wave, rk4_step


class EIOscillator:

    def __init__(self,
                 a_ee=12.0, a_ei=4.0, a_ie=13.0, a_ii=11.0,
                 P_e=2.5, P_i=0.0,
                 theta_e=2.8, theta_i=4.0,
                 sigma_e=1.0, sigma_i=1.0,
                 k_e=1.5, k_i=0.5,
                 omega=2.0 * np.pi * 6.0,
                 sawtooth_amp=1.0):
        self.a_ee = a_ee
        self.a_ei = a_ei
        self.a_ie = a_ie
        self.a_ii = a_ii
        self.P_e = P_e
        self.P_i = P_i
        self.theta_e = theta_e
        self.theta_i = theta_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.k_e = k_e
        self.k_i = k_i
        self.omega = omega
        self.sawtooth_amp = sawtooth_amp

    def _dynamics(self, t, state):






        pass

    def simulate(self, E0=0.1, I0=0.05, t_span=(0.0, 5.0), dt=0.001):
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 2), dtype=float)
        state[0] = [E0, I0]
        for i in range(n_steps):
            state[i + 1] = rk4_step(self._dynamics, t[i], state[i], dt)
        return t, state

    def compute_lfp(self, state, k_E=1.0, k_I=1.5, noise_std=0.02, dt=0.001):
        E = state[:, 0]
        I = state[:, 1]
        lfp = k_E * E - k_I * I
        if noise_std > 0:
            lfp += noise_std * np.sqrt(dt) * np.random.randn(len(lfp))
        return lfp


class AIRPopulationDynamics:

    def __init__(self, N=1000, alpha=0.3, beta=0.1, gamma=0.05, delta=0.02):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _dynamics(self, t, state, E_input_func):
        A, Q, R = state

        A = max(A, 0.0)
        Q = max(Q, 0.0)
        R = max(R, 0.0)
        N = self.N

        E_t = E_input_func(t)
        conn_mod = sigmoid_activation(E_t, theta=0.5, sigma=0.3)

        dA = (self.alpha * conn_mod * Q * A / N
              - self.beta * A
              + self.gamma * R)
        dQ = (-self.alpha * conn_mod * Q * A / N
              + self.beta * A
              - self.delta * Q)
        dR = (self.delta * Q
              - self.gamma * R)
        return np.array([dA, dQ, dR], dtype=float)

    def simulate(self, E_input_func, A0=10, Q0=None, R0=0,
                 t_span=(0.0, 10.0), dt=0.01):
        if Q0 is None:
            Q0 = self.N - A0 - R0
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 3), dtype=float)
        state[0] = [float(A0), float(Q0), float(R0)]
        for i in range(n_steps):
            state[i + 1] = rk4_step(
                lambda ti, si: self._dynamics(ti, si, E_input_func),
                t[i], state[i], dt)

            s = state[i + 1]
            s = np.maximum(s, 0.0)
            total = np.sum(s)
            if total > 0:
                s = s * (self.N / total)
            state[i + 1] = s
        return t, state


class MultiPopulationArray:

    def __init__(self, n_channels=8, coupling_matrix=None, **ei_kwargs):
        self.n_channels = n_channels
        self.oscillators = [EIOscillator(**ei_kwargs) for _ in range(n_channels)]
        if coupling_matrix is None:

            C = np.eye(n_channels) * 0.0
            for i in range(n_channels - 1):
                C[i, i + 1] = 0.1
                C[i + 1, i] = 0.1
            coupling_matrix = C
        self.C = np.asarray(coupling_matrix, dtype=float)

        np.fill_diagonal(self.C, 0.0)

    def simulate(self, initial_states=None, t_span=(0.0, 3.0), dt=0.001):
        n = self.n_channels
        if initial_states is None:
            initial_states = np.random.rand(n, 2) * 0.1
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 2 * n), dtype=float)
        state[0] = np.asarray(initial_states, dtype=float).flatten()

        def full_dynamics(ti, y):
            dydt = np.zeros_like(y)
            E_vals = y[0::2]
            for ch in range(n):
                idx_e = 2 * ch
                idx_i = 2 * ch + 1

                local = self.oscillators[ch]._dynamics(ti, [y[idx_e], y[idx_i]])

                coupling = np.sum(self.C[ch, :] * (E_vals - y[idx_e]))
                dydt[idx_e] = local[0] + coupling
                dydt[idx_i] = local[1]
            return dydt

        for i in range(n_steps):
            state[i + 1] = rk4_step(full_dynamics, t[i], state[i], dt)
        return t, state

    def extract_lfp_channels(self, state, k_E=1.0, k_I=1.5, noise_std=0.02, dt=0.001):
        n = self.n_channels
        n_t = state.shape[0]
        lfp = np.zeros((n, n_t), dtype=float)
        for ch in range(n):
            E = state[:, 2 * ch]
            I = state[:, 2 * ch + 1]
            lfp[ch] = k_E * E - k_I * I
        if noise_std > 0:
            lfp += noise_std * np.sqrt(dt) * np.random.randn(n, n_t)
        return lfp
