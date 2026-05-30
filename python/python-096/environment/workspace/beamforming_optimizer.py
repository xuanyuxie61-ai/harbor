
import numpy as np
from typing import Callable, Tuple, Optional


class RK12Solver:

    def __init__(self, yprime: Callable[[float, np.ndarray], np.ndarray]):
        self.yprime = yprime

    def solve(self, tspan: Tuple[float, float], y0: np.ndarray,
              n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y0 = np.asarray(y0, dtype=float).flatten()
        m = y0.size
        t = np.zeros(n_steps + 1, dtype=float)
        y = np.zeros((n_steps + 1, m), dtype=float)
        e = np.zeros((n_steps + 1, m), dtype=float)

        dt = (tspan[1] - tspan[0]) / n_steps
        t[0] = tspan[0]
        y[0, :] = y0
        e[0, :] = 0.0

        for i in range(n_steps):
            k1 = dt * self.yprime(t[i], y[i, :])
            yt = y[i, :] + k1
            k2 = dt * self.yprime(t[i] + dt, yt)
            t[i + 1] = t[i] + dt
            y[i + 1, :] = y[i, :] + 0.5 * (k1 + k2)
            e[i + 1, :] = 0.5 * (k2 - k1)

        return t, y, e

    def adaptive_solve(self, tspan: Tuple[float, float], y0: np.ndarray,
                       tol: float = 1e-6, max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        y0 = np.asarray(y0, dtype=float).flatten()
        t0, t1 = tspan
        dt = (t1 - t0) / 100.0
        t_list = [t0]
        y_list = [y0.copy()]
        t_curr = t0
        y_curr = y0.copy()
        steps = 0

        while t_curr < t1 and steps < max_steps:
            dt = min(dt, t1 - t_curr)
            k1 = dt * self.yprime(t_curr, y_curr)
            yt = y_curr + k1
            k2 = dt * self.yprime(t_curr + dt, yt)
            y_next = y_curr + 0.5 * (k1 + k2)
            err_est = np.max(np.abs(0.5 * (k2 - k1)))

            if err_est < tol or dt < 1e-12:
                t_curr += dt
                t_list.append(t_curr)
                y_list.append(y_next.copy())
                y_curr = y_next
                steps += 1
                if err_est < tol * 0.1:
                    dt *= 2.0
            else:
                dt *= 0.5
                if dt < 1e-14:
                    break

        return np.array(t_list), np.vstack(y_list)


class KeplerPerturbedArrayCoupling:

    def __init__(self, delta: float = 0.015, e: float = 0.6):
        self.delta = delta
        self.e = e

        p0 = 1.0 - e
        p1 = 0.0
        q0 = 0.0
        q1 = np.sqrt((1.0 + e) / max(1.0 - e, 1e-12))
        self.y0 = np.array([p0, p1, q0, q1], dtype=float)

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:



        raise NotImplementedError("Hole 2: 受摄开普勒 ODE 右端项 derivative 待实现")

    def conserved_quantity(self, y: np.ndarray) -> float:
        q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
        r = np.sqrt(q1 ** 2 + q2 ** 2)
        r = max(r, 1e-12)
        return 0.5 * (p1 ** 2 + p2 ** 2) - 1.0 / r - self.delta / (3.0 * r ** 3)


class LangfordBeamPhaseDynamics:

    def __init__(self, a: float = 0.95, b: float = 0.7, c: float = 0.6,
                 d: float = 3.5, e: float = 0.25, f: float = 0.1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.y0 = np.array([0.1, 1.0, 0.0], dtype=float)

    def derivative(self, t: float, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]
        dxdt = (z - self.b) * x - self.d * y
        dydt = self.d * x + (z - self.b) * y
        dzdt = (self.c + self.a * z - z ** 3 / 3.0
                - (x ** 2 + y ** 2) * (1.0 + self.e * z)
                + self.f * z * x ** 3)
        return np.array([dxdt, dydt, dzdt])

    def lyapunov_exponent_estimate(self, tspan: Tuple[float, float] = (0.0, 100.0),
                                   n_steps: int = 5000) -> float:
        solver = RK12Solver(self.derivative)
        t, y, _ = solver.solve(tspan, self.y0, n_steps)

        if y.shape[0] < 10:
            return 0.0
        diffs = np.linalg.norm(y[1:, :] - y[:-1, :], axis=1)
        dt = (tspan[1] - tspan[0]) / n_steps
        rates = np.log(diffs[1:] / np.maximum(diffs[:-1], 1e-18)) / dt
        return float(np.mean(rates))


class AdaptiveBeamformerODE:

    def __init__(self, n_elements: int, steering_angle: float = 0.0,
                 lambda_reg: float = 0.5, mu_reg: float = 0.1,
                 element_spacing: float = 0.5):
        self.n_elements = n_elements
        self.steering_angle = steering_angle
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.element_spacing = element_spacing
        self.k0 = 2.0 * np.pi

    def _array_response(self, theta: float) -> np.ndarray:
        n = np.arange(self.n_elements)
        return np.exp(1j * self.k0 * self.element_spacing * n * np.sin(theta))

    def gradient_flow_derivative(self, t: float, phase_vec: np.ndarray) -> np.ndarray:
        w = np.exp(1j * phase_vec)
        a0 = self._array_response(self.steering_angle)

        g_main = -2.0 * self.lambda_reg * (np.abs(np.vdot(a0, w)) - 1.0) * np.angle(np.vdot(a0, w)) * np.ones(self.n_elements)

        g_sidelobe = np.zeros(self.n_elements, dtype=float)
        sidelobe_angles = np.linspace(-np.pi / 2, np.pi / 2, 21)
        for th in sidelobe_angles:
            if abs(th - self.steering_angle) < 0.1:
                continue
            a = self._array_response(th)
            p = np.abs(np.vdot(a, w)) ** 2
            g_sidelobe += 2.0 * self.mu_reg * p * np.imag(np.conj(w) * a * np.vdot(a, w))

        g_signal = -2.0 * np.imag(np.conj(w) * w)
        return -(g_main + g_sidelobe + g_signal)

    def evolve(self, tspan: Tuple[float, float] = (0.0, 10.0),
               n_steps: int = 500, initial_phase: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if initial_phase is None:
            initial_phase = np.zeros(self.n_elements, dtype=float)
        solver = RK12Solver(self.gradient_flow_derivative)
        t, y, _ = solver.solve(tspan, initial_phase, n_steps)
        return t, y
