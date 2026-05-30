
import numpy as np
from typing import Callable, Tuple, Optional
from utils import rk4_step


class SwingEquation:

    def __init__(self, H: float, D: float, f_base: float = 50.0,
                 E_prime: float = 1.1, V_inf: float = 1.0, X: float = 0.5):
        self.H = H
        self.D = D
        self.omega_s = 2.0 * np.pi * f_base
        self.M = 2.0 * H / self.omega_s
        self.E_prime = E_prime
        self.V_inf = V_inf
        self.X = X

    def electrical_power(self, delta: float) -> float:
        if self.X <= 1e-12:
            raise ValueError("reactance X must be positive")
        return (self.E_prime * self.V_inf / self.X) * np.sin(delta)

    def derivative(self, t: float, state: np.ndarray,
                   P_m: float, fault_active: bool = False,
                   X_fault: float = 1.0) -> np.ndarray:
        delta, omega = state[0], state[1]
        X_eff = self.X + X_fault if fault_active else self.X
        X_eff = max(X_eff, 1e-12)
        Pe = (self.E_prime * self.V_inf / X_eff) * np.sin(delta)
        d_delta = omega - self.omega_s
        d_omega = (P_m - Pe - self.D * (omega - self.omega_s)) / self.M
        return np.array([d_delta, d_omega])

    def critical_clearing_angle(self, P_m: float) -> Optional[float]:
        P_max_post = self.E_prime * self.V_inf / self.X
        if P_m >= P_max_post:
            return None
        delta_0 = np.arcsin(P_m / P_max_post)
        delta_max = np.pi - delta_0

        def area_balance(delta_cr):
            A_acc = P_m * (delta_cr - delta_0)
            A_dec = P_max_post * (np.cos(delta_cr) - np.cos(delta_max)) \
                    - P_m * (delta_max - delta_cr)
            return A_acc - A_dec

        from scipy.optimize import brentq
        try:
            delta_cr = brentq(area_balance, delta_0, delta_max)
            return delta_cr
        except Exception:

            lo, hi = delta_0, delta_max
            for _ in range(100):
                mid = (lo + hi) * 0.5
                if area_balance(lo) * area_balance(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) * 0.5

    def simulate(self, t_span: Tuple[float, float], dt: float,
                 P_m: float, delta0: float, omega0: float,
                 fault_time: Optional[Tuple[float, float]] = None,
                 X_fault: float = 1.0) -> dict:
        t0, tf = t_span
        if dt <= 0:
            raise ValueError("dt must be positive")
        n_steps = int(np.ceil((tf - t0) / dt))
        times = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 2), dtype=np.float64)
        states[0] = [delta0, omega0]

        for k in range(n_steps):
            t = times[k]
            y = states[k]
            fault_active = False
            if fault_time is not None:
                t_on, t_off = fault_time
                fault_active = (t >= t_on and t < t_off)

            def f(t_local, y_local):
                return self.derivative(t_local, y_local, P_m,
                                       fault_active=fault_active, X_fault=X_fault)

            states[k + 1] = rk4_step(f, t, y, dt)

        return {
            "t": times,
            "delta": states[:, 0],
            "omega": states[:, 1],
            "stable": self._assess_stability(times, states[:, 0])
        }

    def _assess_stability(self, t: np.ndarray, delta: np.ndarray) -> bool:
        n = len(delta)
        tail = delta[int(0.8 * n):]
        if np.max(np.abs(tail)) > np.pi:
            return False

        peaks = []
        for i in range(1, len(tail) - 1):
            if tail[i] > tail[i - 1] and tail[i] > tail[i + 1]:
                peaks.append(tail[i])
        if len(peaks) >= 3:
            return peaks[-1] < peaks[-2] < peaks[0]
        return True


class MultiMachineStability:

    def __init__(self, n_gen: int, H: np.ndarray, D: np.ndarray,
                 E_prime: np.ndarray, Y_reduced: np.ndarray):
        self.n_gen = n_gen
        self.H = np.array(H, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64)
        self.E_prime = np.array(E_prime, dtype=np.float64)
        self.Y_reduced = np.array(Y_reduced, dtype=np.complex128)
        self.G = self.Y_reduced.real
        self.B = self.Y_reduced.imag
        self.omega_s = 2.0 * np.pi * 50.0

    def derivative(self, t: float, state: np.ndarray, P_m: np.ndarray) -> np.ndarray:
        n = self.n_gen
        delta = state[:n]
        omega = state[n:]
        d_delta = omega - self.omega_s
        d_omega = np.zeros(n, dtype=np.float64)
        for i in range(n):
            Pe = 0.0
            for j in range(n):
                angle_diff = delta[i] - delta[j]
                Pe += self.E_prime[i] * self.E_prime[j] * (
                    self.G[i, j] * np.cos(angle_diff)
                    + self.B[i, j] * np.sin(angle_diff)
                )
            M_i = 2.0 * self.H[i] / self.omega_s
            d_omega[i] = (P_m[i] - Pe - self.D[i] * (omega[i] - self.omega_s)) / M_i
        return np.concatenate([d_delta, d_omega])

    def simulate(self, t_span: Tuple[float, float], dt: float,
                 P_m: np.ndarray, delta0: np.ndarray, omega0: np.ndarray) -> dict:
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        times = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 2 * self.n_gen), dtype=np.float64)
        states[0, :self.n_gen] = delta0
        states[0, self.n_gen:] = omega0

        for k in range(n_steps):
            t = times[k]
            y = states[k]

            def f(_, y_local):
                return self.derivative(t, y_local, P_m)

            states[k + 1] = rk4_step(f, t, y, dt)

        return {
            "t": times,
            "delta": states[:, :self.n_gen],
            "omega": states[:, self.n_gen:]
        }
