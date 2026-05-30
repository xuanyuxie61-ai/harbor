
import numpy as np
from typing import Callable


class BiochemicalODE:

    def __init__(self, kf: float = 1.0, kr: float = 0.1, kcat: float = 0.5):
        self.kf = kf
        self.kr = kr
        self.kcat = kcat



        self.S = np.array([
            [-1,  1,  1],
            [-1,  1,  0],
            [ 1, -1, -1],
            [ 0,  0,  1]
        ])

    def reaction_rates(self, y: np.ndarray) -> np.ndarray:
        E, S, ES, P = y
        alpha = 0.05
        r1 = self.kf * E * S / (1.0 + alpha * ES)
        r2 = self.kr * ES
        r3 = self.kcat * ES / (1.0 + 0.1 * P)
        return np.array([r1, r2, r3])

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        r = self.reaction_rates(y)
        return self.S @ r

    def conserved_quantities(self, y: np.ndarray) -> np.ndarray:
        E, S, ES, P = y
        h1 = E + ES
        h2 = S + ES + P
        return np.array([h1, h2])

    def integrate_rk4(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if len(y0) != 4:
            raise ValueError("y0 must have length 4.")
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 4))
        y[0] = y0

        h0 = self.conserved_quantities(y0)

        for i in range(n_steps):
            yi = y[i]
            k1 = self.rhs(t[i], yi)
            k2 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k1)
            k3 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k2)
            k4 = self.rhs(t[i] + dt, yi + dt * k3)
            y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


            h = self.conserved_quantities(y[i + 1])
            dh = h - h0


            y[i + 1][0] -= dh[0] * 0.5
            y[i + 1][2] += dh[0] * 0.5
            y[i + 1][1] -= dh[1] * 0.3
            y[i + 1][2] -= dh[1] * 0.3
            y[i + 1][3] -= dh[1] * 0.4


            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return t, y


class ExtendedBrusselator:

    def __init__(self, a: float = 1.0, b: float = 3.0, D1: float = 0.01, D2: float = 0.01):
        self.a = a
        self.b = b
        self.D1 = D1
        self.D2 = D2

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        x1, x2 = y
        dx1 = self.a - (self.b + 1.0) * x1 + x1 ** 2 * x2
        dx2 = self.b * x1 - x1 ** 2 * x2
        return np.array([dx1, dx2])

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        x1, x2 = y
        df1_dx1 = -(self.b + 1.0) + 2.0 * x1 * x2
        df1_dx2 = x1 ** 2
        df2_dx1 = self.b - 2.0 * x1 * x2
        df2_dx2 = -x1 ** 2
        return np.array([[df1_dx1, df1_dx2],
                         [df2_dx1, df2_dx2]])

    def integrate(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = y0
        for i in range(n_steps):
            yi = y[i]
            k1 = self.rhs(t[i], yi)
            k2 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k1)
            k3 = self.rhs(t[i] + 0.5 * dt, yi + 0.5 * dt * k2)
            k4 = self.rhs(t[i] + dt, yi + dt * k3)
            y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, y

    def lyapunov_exponent_numerical(self, y0: np.ndarray, t_span: tuple, n_steps: int) -> float:
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        y = y0.copy()
        delta = np.array([1e-8, 0.0])
        norm_delta0 = np.linalg.norm(delta)
        lyap_sum = 0.0
        count = 0

        for i in range(n_steps):

            k1 = self.rhs(t0 + i * dt, y)
            k2 = self.rhs(t0 + i * dt + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t0 + i * dt + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t0 + i * dt + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


            J = self.jacobian(y)

            delta = delta + dt * (J @ delta)
            norm_d = np.linalg.norm(delta)
            if norm_d > 1e-6:
                lyap_sum += np.log(norm_d / norm_delta0)
                delta = delta / norm_d * norm_delta0
                count += 1

        if count == 0:
            return 0.0
        return lyap_sum / (count * dt)
