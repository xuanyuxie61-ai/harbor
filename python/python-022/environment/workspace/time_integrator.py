
import numpy as np
from typing import Callable, Tuple
from icf_parameters import NP
from utils import clamp






class CliffRNG:

    def __init__(self, seed: float = 0.314159265):
        self.x = clamp(seed, 1.0e-10, 1.0 - 1.0e-10)

    def next(self) -> float:
        self.x = np.fmod(-100.0 * np.log(self.x), 1.0)
        if self.x <= 0.0 or self.x >= 1.0:
            self.x = 0.5
        return self.x

    def perturbation(self, magnitude: float = 1.0e-12) -> float:
        return magnitude * (2.0 * self.next() - 1.0)






class RKF45Integrator:

    def __init__(self, reltol: float = NP.ADAPTIVE_TOL,
                 abstol: float = NP.ADAPTIVE_TOL * 1.0e-3):
        self.reltol = reltol
        self.abstol = abstol
        self.cliff = CliffRNG(seed=0.2718281828)
        self.nfe = 0

    def _compute_ks(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        k1 = f(t, y)
        k2 = f(t + h / 4.0, y + h * k1 / 4.0)
        k3 = f(t + 3.0 * h / 8.0, y + h * (3.0 * k1 + 9.0 * k2) / 32.0)
        k4 = f(t + 12.0 * h / 13.0,
               y + h * (1932.0 * k1 - 7200.0 * k2 + 7296.0 * k3) / 2197.0)
        k5 = f(t + h,
               y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0
                        - 845.0 * k4 / 4104.0))
        k6 = f(t + h / 2.0,
               y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0
                        + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))
        self.nfe += 6
        return k1, k2, k3, k4, k5, k6

    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float, float, bool]:
        k1, k2, k3, k4, k5, k6 = self._compute_ks(f, t, y, h)


        y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0
                      + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0
                      + 2.0 * k6 / 55.0)


        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0
                      + 2197.0 * k4 / 4104.0 - k5 / 5.0)


        e = np.abs(y5 - y4)
        scale = self.reltol * np.maximum(np.abs(y5), np.abs(y)) + self.abstol
        err = np.max(e / scale)

        if err <= 1.0 or h <= NP.MIN_DT * 2.0:

            h_new = h * min(5.0, max(0.1, 0.9 * (1.0 / err)**0.2))
            h_new = clamp(h_new, NP.MIN_DT, NP.MAX_DT)

            y5 = y5 + self.cliff.perturbation(magnitude=self.abstol * h)
            return y5, t + h, h_new, True
        else:

            h_new = h * max(0.1, 0.9 * (1.0 / err)**0.2)
            h_new = clamp(h_new, NP.MIN_DT, h)
            return y, t, h_new, False

    def integrate(self, f: Callable, t0: float, y0: np.ndarray,
                  t_end: float, h0: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if h0 is None:
            h0 = NP.MAX_DT * 0.1

        t = t0
        y = np.array(y0, dtype=float)
        h = clamp(h0, NP.MIN_DT, NP.MAX_DT)

        t_list = [t]
        y_list = [y.copy()]
        dt_list = [h]

        max_steps = 100000
        step_count = 0

        while t < t_end and step_count < max_steps:
            h = min(h, t_end - t)
            y_new, t_new, h_new, accepted = self.step(f, t, y, h)

            if accepted:
                t = t_new
                y = y_new
                t_list.append(t)
                y_list.append(y.copy())

            h = h_new
            dt_list.append(h)
            step_count += 1

        return np.array(t_list), np.array(y_list), np.array(dt_list)






def explicit_euler_step(f: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h * f(t, y)


def adaptive_euler(f: Callable, t0: float, y0: np.ndarray,
                   t_end: float, tol: float = 1.0e-6) -> Tuple[np.ndarray, np.ndarray]:
    t = t0
    y = np.array(y0, dtype=float)
    h = NP.MAX_DT * 0.1

    t_list = [t]
    y_list = [y.copy()]

    while t < t_end and len(t_list) < 50000:
        h = min(h, t_end - t)

        y1 = explicit_euler_step(f, t, y, h)
        y2a = explicit_euler_step(f, t, y, h / 2.0)
        y2 = explicit_euler_step(f, t + h / 2.0, y2a, h / 2.0)

        err = np.max(np.abs(y1 - y2))
        if err < tol or h <= NP.MIN_DT:
            y = y2
            t += h
            t_list.append(t)
            y_list.append(y.copy())
            h = min(h * 1.5, NP.MAX_DT)
        else:
            h = max(h * 0.5, NP.MIN_DT)

    return np.array(t_list), np.array(y_list)
