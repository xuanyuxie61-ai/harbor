
import numpy as np
from typing import Callable






def sawtooth_wave(t: float, omega: float = 2.0 * np.pi) -> float:
    if not np.isfinite(t):
        return 0.0
    frac = (omega * t / (2.0 * np.pi)) % 1.0
    return 2.0 * (frac - 0.5)






GRAZING_PARAMS = {
    'a': 0.2,
    'c1': 0.05,
    'c2': 0.05,
    'd1': 2.0,
    'd2': 2.0,
    'k': 2.0,
    'r1': 0.5,
    'gamma': 0.1,
}


def grazing_coupling(x1: float, x3: float, params: dict = None) -> float:
    if params is None:
        params = GRAZING_PARAMS
    gamma = params['gamma']

    x3_clipped = float(np.clip(x3, -100.0, 100.0))
    denom = 1.0 + x3_clipped ** 2
    if not np.isfinite(denom) or denom == 0.0:
        return 0.0
    return -gamma * x1 * x3_clipped / denom






class ControlledNonlinearOscillator:

    def __init__(self, omega0: float = 1.0, omega_s: float = 2.0 * np.pi,
                 dt: float = 0.005, params: dict = None,
                 state_bounds: tuple = (-10.0, 10.0)):
        self.omega0 = omega0
        self.omega_s = omega_s
        self.dt = dt
        self.params = params if params is not None else GRAZING_PARAMS.copy()
        self.state_bounds = state_bounds
        self.state = np.zeros(4)
        self.t = 0.0
        self.step_count = 0

    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        if initial_state is None:
            self.state = np.random.randn(4) * 0.1
        else:
            self.state = np.array(initial_state, dtype=float).copy()
        self.t = 0.0
        self.step_count = 0
        return self._get_observation()

    def _dynamics(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:

        state = np.clip(np.asarray(state, dtype=float), -50.0, 50.0)
        x1, x2, x3, x4 = state
        u1, u2, u3, u4 = action
        p = self.params


        dx1 = x2 + u1
        dx2 = -(self.omega0 ** 2) * x1 + sawtooth_wave(t, self.omega_s) \
               + grazing_coupling(x1, x3, p) + u2


        exp_arg1 = np.clip(-p['d1'] * x3, -50.0, 50.0)
        exp_arg2 = np.clip(-p['d2'] * x3, -50.0, 50.0)
        dx3 = p['r1'] * x3 * (1.0 - x3 / p['k']) \
              - p['c1'] * x4 * (1.0 - np.exp(exp_arg1)) + u3
        dx4 = -p['a'] * x4 + p['c2'] * x4 * (1.0 - np.exp(exp_arg2)) + u4


        deriv = np.array([dx1, dx2, dx3, dx4])
        deriv = np.clip(deriv, -100.0, 100.0)
        return deriv

    def step(self, action: np.ndarray, integrator: str = 'rk4') -> tuple:
        action = np.clip(np.asarray(action, dtype=float), -2.0, 2.0)

        if integrator == 'rk4':
            self.state = self._rk4_step(self.state, action, self.t, self.dt)
        elif integrator == 'euler':
            self.state = self.state + self.dt * self._dynamics(self.state, action, self.t)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")


        self.state = np.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        self.t += self.dt
        self.step_count += 1

        obs = self._get_observation()
        reward = self._compute_reward(action)
        done = self._check_done()
        info = {'t': self.t, 'step': self.step_count}
        return obs, reward, done, info

    def _rk4_step(self, state: np.ndarray, action: np.ndarray, t: float, h: float) -> np.ndarray:
        k1 = self._dynamics(state, action, t)
        k2 = self._dynamics(state + 0.5 * h * k1, action, t + 0.5 * h)
        k3 = self._dynamics(state + 0.5 * h * k2, action, t + 0.5 * h)
        k4 = self._dynamics(state + h * k3, action, t + h)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_observation(self) -> np.ndarray:
        noise = np.random.randn(4) * 0.02
        return self.state + noise

    def _compute_reward(self, action: np.ndarray) -> float:
        from special_functions import sine_integral
        s_norm = np.linalg.norm(self.state)
        a_norm = np.linalg.norm(action)

        if not np.isfinite(s_norm):
            s_norm = 0.0
        if not np.isfinite(a_norm):
            a_norm = 0.0
        si_term = sine_integral(min(float(s_norm), 50.0))
        reward = -0.5 * s_norm ** 2 - 0.1 * a_norm ** 2 + 0.5 * si_term * np.exp(-a_norm ** 2 / 4.0)
        return float(reward)

    def _check_done(self) -> bool:
        if self.step_count >= 500:
            return True
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            return True
        return False

    def reference_trajectory(self, t: float) -> np.ndarray:
        from scipy.special import jv
        x1 = jv(0, self.omega0 * t) * np.cos(self.omega_s * t)

        eps = 1.0e-6
        x1_p = jv(0, self.omega0 * (t + eps)) * np.cos(self.omega_s * (t + eps))
        x2 = (x1_p - x1) / eps
        x3 = 0.5 + 0.3 * np.sin(0.5 * t)
        x4 = 0.2 + 0.1 * np.cos(0.3 * t)
        return np.array([x1, x2, x3, x4])
