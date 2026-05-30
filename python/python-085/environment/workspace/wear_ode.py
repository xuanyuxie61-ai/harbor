import numpy as np
from typing import Tuple, Callable, Optional
from utils import safe_divide


class ArchardWearModel:

    def __init__(self, wear_coeff: float = 1e-6, omega: float = 2.0 * np.pi,
                 v0: float = 0.01):
        self.k_w = wear_coeff
        self.omega = omega
        self.v0 = v0

    def tangential_velocity(self, t: float) -> float:
        return self.v0 * np.sin(self.omega * t)

    def wear_rate(self, t: float, h: float, pressure_func: Callable[[float], float]) -> float:
        p_n = pressure_func(t)
        v_t = self.tangential_velocity(t)
        return self.k_w * p_n * abs(v_t)

    def integrate_midpoint(self, h0: float, t_span: Tuple[float, float], n_steps: int,
                           pressure_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray]:
        a, b = t_span
        h = (b - a) / n_steps
        theta = 0.5
        it_max = 10

        t = np.linspace(a, b, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = h0

        for i in range(n_steps):
            t_m = t[i] + theta * h
            ym = y[i]

            for _ in range(it_max):
                ym_new = y[i] + theta * h * self.wear_rate(t_m, ym, pressure_func)
                if abs(ym_new - ym) < 1e-14:
                    ym = ym_new
                    break
                ym = ym_new
            y[i + 1] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i]

        return t, y

    def integrate_rk4(self, h0: float, t_span: Tuple[float, float], n_steps: int,
                      pressure_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray]:
        a, b = t_span
        h = (b - a) / n_steps
        t = np.linspace(a, b, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = h0

        for i in range(n_steps):
            k1 = self.wear_rate(t[i], y[i], pressure_func)
            k2 = self.wear_rate(t[i] + 0.5 * h, y[i] + 0.5 * h * k1, pressure_func)
            k3 = self.wear_rate(t[i] + 0.5 * h, y[i] + 0.5 * h * k2, pressure_func)
            k4 = self.wear_rate(t[i] + h, y[i] + h * k3, pressure_func)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return t, y

    def compute_wear_volume(self, t: np.ndarray, h: np.ndarray,
                            contact_area: float) -> float:
        return contact_area * float(np.max(h))


def coupled_wear_contact_step(wear_model: ArchardWearModel,
                               current_wear_depth: np.ndarray,
                               pressure: np.ndarray,
                               dt: float) -> np.ndarray:
    v_t = abs(wear_model.tangential_velocity(0.0))
    rate = wear_model.k_w * pressure * v_t
    return current_wear_depth + dt * rate
