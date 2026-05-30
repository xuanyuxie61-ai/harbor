
import numpy as np
from typing import Callable, Tuple, List
from utils import clip_to_bounds, robust_sqrt


class CPGNetwork:

    def __init__(self, n_osc: int = 6, alpha: float = 50.0, mu: float = 1.0,
                 omega: float = 2.0 * np.pi * 1.0, coupling_strength: float = 5.0):
        self.n = n_osc
        self.alpha = alpha
        self.mu = mu
        self.omega = omega


        self.C = np.zeros((n_osc, n_osc))
        for i in range(n_osc):
            for j in range(n_osc):
                if i != j:

                    same_group = (i % 2) == (j % 2)
                    self.C[i, j] = coupling_strength if same_group else -coupling_strength * 0.5

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Hole 1: 请补全 CPGNetwork.rhs 的实现")

    def extract_phase(self, state: np.ndarray) -> np.ndarray:
        x = state[:self.n]
        y = state[self.n:]
        return np.arctan2(y, x)

    def extract_amplitude(self, state: np.ndarray) -> np.ndarray:
        return np.sqrt(state[:self.n] ** 2 + state[self.n:] ** 2)


class TrapezoidalIntegrator:

    def __init__(self, it_max: int = 10):
        self.it_max = it_max

    def integrate(self, f: Callable, tspan: Tuple[float, float], y0: np.ndarray,
                  n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = tspan
        h = (tf - t0) / n_steps
        dim = y0.size
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, dim))
        y[0] = y0.flatten()
        for i in range(n_steps):
            tn = t[i]
            yn = y[i]
            f_tn = f(tn, yn)







            raise NotImplementedError("Hole 2: 请补全 TrapezoidalIntegrator.integrate 的梯形固定点迭代")
        return t, y


class StanceSwingAutomaton:

    def __init__(self, n_legs: int = 6, stance_min_steps: int = 3,
                 phase_stance_center: float = 0.0, phase_stance_width: float = np.pi / 2):
        self.n = n_legs
        self.stance_min = stance_min_steps
        self.phi_c = phase_stance_center
        self.phi_w = phase_stance_width

        self.stance_counter = np.zeros(n_legs, dtype=int)

        self.neighbors = {i: [(i - 1) % n_legs, (i + 1) % n_legs] for i in range(n_legs)}

    def update(self, phase: np.ndarray) -> np.ndarray:
        s_new = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            phi = phase[i]

            in_stance_window = abs(((phi - self.phi_c + np.pi) % (2 * np.pi)) - np.pi) <= self.phi_w
            desired = 1 if in_stance_window else 0


            neighbor_swing_count = sum(1 for j in self.neighbors[i] if self.stance_counter[j] == 0)
            if neighbor_swing_count >= len(self.neighbors[i]):
                desired = 1


            if self.stance_counter[i] > 0 and self.stance_counter[i] < self.stance_min and desired == 0:
                desired = 1

            s_new[i] = desired
            if s_new[i] == 1:
                self.stance_counter[i] += 1
            else:
                self.stance_counter[i] = 0
        return s_new

    def reset(self):
        self.stance_counter = np.zeros(self.n, dtype=int)


class LegDynamics:

    def __init__(self, mass_matrix: np.ndarray, damping: np.ndarray, gravity: float = 9.81):
        self.M = np.asarray(mass_matrix, dtype=float)
        self.C_mat = np.asarray(damping, dtype=float)
        self.g = gravity
        self.M_inv = np.linalg.inv(self.M)

    def dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray,
                 f_contact: np.ndarray, J: np.ndarray) -> np.ndarray:
        G = J.T @ np.array([0.0, 0.0, self.g])
        rhs = tau + J.T @ f_contact - self.C_mat @ dq - G
        ddq = self.M_inv @ rhs
        return ddq

    def state_space_rhs(self, state: np.ndarray, tau: np.ndarray,
                        f_contact: np.ndarray, J_func: Callable) -> np.ndarray:
        n = state.size // 2
        q = state[:n]
        dq = state[n:]
        J = J_func(q)
        ddq = self.dynamics(q, dq, tau, f_contact, J)
        return np.concatenate((dq, ddq))
