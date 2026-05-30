
import numpy as np
from scipy.integrate import solve_ivp


class SimplifiedSGNModel:

    def __init__(self, tau_m=0.1, epsilon=0.08, a=0.7, b=0.8,
                 V_rest=-65.0, V_thresh=-40.0):
        self.tau_m = float(tau_m)
        self.epsilon = float(epsilon)
        self.a = float(a)
        self.b = float(b)
        self.V_rest = float(V_rest)
        self.V_thresh = float(V_thresh)

    def derivatives(self, t, y, stimulus_func):
        V_m, W = y

        v = (V_m - self.V_rest) / 25.0
        I_stim = stimulus_func(t) / 25.0

        dvdt = (v - v**3 / 3.0 - W + I_stim) / self.tau_m
        dWdt = self.epsilon * (v + self.a - self.b * W)

        dVdt = dvdt * 25.0
        return np.array([dVdt, dWdt])

    def simulate(self, t_span, y0, stimulus_func, method='RK45',
                 max_step=0.01):
        y0 = np.asarray(y0, dtype=float)
        if len(y0) != 2:
            raise ValueError("y0 长度必须为 2")


        def threshold_cross(t, y):
            return y[0] - self.V_thresh
        threshold_cross.terminal = False
        threshold_cross.direction = 1

        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, stimulus_func),
            t_span, y0, method=method, max_step=max_step,
            events=threshold_cross, dense_output=True
        )

        spike_times = sol.t_events[0].tolist() if sol.t_events[0] is not None else []
        return sol, spike_times


class DetailedSGNModel:

    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.4, T=310.15):
        self.C_m = float(C_m)
        self.g_Na = float(g_Na)
        self.g_K = float(g_K)
        self.g_L = float(g_L)
        self.E_Na = float(E_Na)
        self.E_K = float(E_K)
        self.E_L = float(E_L)
        self.T = float(T)

        self.q10 = 6.3
        self.T_ref = 310.15
        self.phi = self.q10 ** ((self.T - self.T_ref) / 10.0)

    def alpha_m(self, V):
        V = np.asarray(V, dtype=float)
        return np.where(
            np.abs(V + 40.0) < 1e-6,
            1.0,
            0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        )

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        V = np.asarray(V, dtype=float)
        return np.where(
            np.abs(V + 55.0) < 1e-6,
            0.1,
            0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        )

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def derivatives(self, t, y, stimulus_func):
        V_m, m, h, n = y
        phi = self.phi

        I_Na = self.g_Na * (m**3) * h * (V_m - self.E_Na)
        I_K = self.g_K * (n**4) * (V_m - self.E_K)
        I_L = self.g_L * (V_m - self.E_L)
        I_stim = stimulus_func(t)

        dVdt = (-I_Na - I_K - I_L + I_stim) / self.C_m
        dmdt = phi * (self.alpha_m(V_m) * (1.0 - m) - self.beta_m(V_m) * m)
        dhdt = phi * (self.alpha_h(V_m) * (1.0 - h) - self.beta_h(V_m) * h)
        dndt = phi * (self.alpha_n(V_m) * (1.0 - n) - self.beta_n(V_m) * n)

        return np.array([dVdt, dmdt, dhdt, dndt])

    def simulate(self, t_span, stimulus_func, V_rest=None, method='RK45',
                 max_step=0.005):
        if V_rest is None:
            V_rest = self.E_L


        m0 = self.alpha_m(V_rest) / (self.alpha_m(V_rest) + self.beta_m(V_rest))
        h0 = self.alpha_h(V_rest) / (self.alpha_h(V_rest) + self.beta_h(V_rest))
        n0 = self.alpha_n(V_rest) / (self.alpha_n(V_rest) + self.beta_n(V_rest))
        y0 = np.array([V_rest, m0, h0, n0])

        def event_spike(t, y):
            return y[0] - 0.0
        event_spike.terminal = False
        event_spike.direction = 1

        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, stimulus_func),
            t_span, y0, method=method, max_step=max_step,
            events=event_spike, dense_output=True
        )

        spike_times = sol.t_events[0].tolist() if sol.t_events[0] is not None else []
        return sol, spike_times


def biphasic_pulse(t, amplitude, phase_width_ms, interphase_gap_ms=0.05):
    T = 2.0 * phase_width_ms + interphase_gap_ms
    t_mod = t % T
    if t_mod < phase_width_ms:
        return amplitude
    elif t_mod < phase_width_ms + interphase_gap_ms:
        return 0.0
    elif t_mod < T:
        return -amplitude
    return 0.0
