
import numpy as np
from scipy.integrate import odeint


class DamageParameters:

    def __init__(self):

        self.X_T = 2500.0
        self.X_C = 2000.0
        self.Y_T = 80.0
        self.Y_C = 200.0
        self.S = 120.0
        self.S_T = 50.0


        self.sigma_f0 = 3500.0
        self.sigma_m0 = 120.0
        self.tau_s0 = 180.0
        self.m_f = 8.0
        self.m_m = 6.0
        self.m_s = 7.0
        self.k_f = 2.5
        self.k_m = 2.0
        self.k_s = 2.2


        self.epsilon_debond = 0.001
        self.gamma_debond = 0.45
        self.tau_interface = 60.0


        self.d_threshold = 0.99


class DamageState:

    def __init__(self, d_f=0.0, d_m=0.0, d_s=0.0, d_i=0.0):
        self.d_f = float(np.clip(d_f, 0.0, 0.99))
        self.d_m = float(np.clip(d_m, 0.0, 0.99))
        self.d_s = float(np.clip(d_s, 0.0, 0.99))
        self.d_i = float(np.clip(d_i, 0.0, 0.99))

    def to_array(self):
        return np.array([self.d_f, self.d_m, self.d_s, self.d_i])

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2], arr[3])

    def is_failed(self):
        return any(v > 0.95 for v in [self.d_f, self.d_m, self.d_s, self.d_i])


def hashin_failure_criteria(stress, params):
    sigma1, sigma2, tau12 = stress
    results = {}


    if sigma1 >= 0:
        results['fiber_tension'] = (sigma1 / params.X_T) ** 2 + (tau12 / params.S) ** 2

    else:
        results['fiber_compression'] = (abs(sigma1) / params.X_C) ** 2


    if sigma2 >= 0:
        results['matrix_tension'] = (sigma2 / params.Y_T) ** 2 + (tau12 / params.S) ** 2

    else:
        term1 = (sigma2 / (2.0 * params.S_T)) ** 2
        term2 = ((params.Y_C / (2.0 * params.S_T)) ** 2 - 1.0) * sigma2 / params.Y_C
        term3 = (tau12 / params.S) ** 2
        results['matrix_compression'] = term1 + term2 + term3

    return results


def damage_evolution_ode(y, N, stress_amplitude, params):
    d_f, d_m, d_s, d_i = y
    sigma1a, sigma2a, tau12a = stress_amplitude



















    raise NotImplementedError("Hole 3: damage_evolution_ode core computation needs implementation.")


def integrate_damage_cycles(initial_damage, stress_history, params, num_cycles):
    y0 = initial_damage.to_array()
    states = [y0.copy()]


    step = max(1, num_cycles // 100)
    N_current = 0
    y_current = y0.copy()

    while N_current < num_cycles:
        N_step = min(step, num_cycles - N_current)

        idx_start = min(N_current, len(stress_history) - 1)
        idx_end = min(N_current + N_step, len(stress_history))
        if idx_end <= idx_start:
            avg_stress = stress_history[-1]
        else:
            avg_stress = np.mean(stress_history[idx_start:idx_end], axis=0)


        N_span = np.linspace(0, float(N_step), max(5, N_step // 2 + 1))
        sol = odeint(damage_evolution_ode, y_current, N_span,
                     args=(avg_stress, params), rtol=1e-6, atol=1e-9)
        y_current = sol[-1].copy()

        y_current = np.clip(y_current, 0.0, 0.99)
        states.append(y_current.copy())
        N_current += N_step

    return np.array(states)


def compute_damage_dissipation_energy(damage_states, material, params):
    n = len(damage_states)
    if n < 2:
        return 0.0

    W_d = 0.0
    for i in range(1, n):
        d_prev = damage_states[i - 1]
        d_curr = damage_states[i]
        dd = d_curr - d_prev
        d_mid = 0.5 * (d_prev + d_curr)


        denom_f = max(2.0 * material.E1 * (1.0 - d_mid[0]) ** 2, 1e-12)
        denom_m = max(2.0 * material.E2 * (1.0 - d_mid[1]) ** 2, 1e-12)
        denom_s = max(2.0 * material.G12 * (1.0 - d_mid[2]) ** 2, 1e-12)

        Y_f = 1.0 / denom_f
        Y_m = 1.0 / denom_m
        Y_s = 1.0 / denom_s

        W_d += Y_f * dd[0] + Y_m * dd[1] + Y_s * dd[2]

    return W_d


def estimate_damage_period(stress_amplitude, params):
    sigma1a, sigma2a, tau12a = stress_amplitude
    d0 = 0.0

    N_f = np.inf
    if abs(sigma1a) > 0:
        rate_f = (abs(sigma1a) / params.sigma_f0) ** params.m_f
        if rate_f > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_f + 1.0) / ((params.k_f + 1.0) * rate_f))

    if abs(sigma2a) > 0:
        rate_m = (abs(sigma2a) / params.sigma_m0) ** params.m_m
        if rate_m > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_m + 1.0) / ((params.k_m + 1.0) * rate_m))

    if abs(tau12a) > 0:
        rate_s = (abs(tau12a) / params.tau_s0) ** params.m_s
        if rate_s > 0:
            N_f = min(N_f, (1.0 - d0) ** (params.k_s + 1.0) / ((params.k_s + 1.0) * rate_s))

    return N_f if N_f < np.inf else 1e6
