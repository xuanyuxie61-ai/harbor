
import numpy as np
from typing import Tuple, Callable, Optional


class OrnsteinUhlenbeck:

    def __init__(self, kappa: float, mu: float, sigma: float,
                 s0: float, t_max: float, n_steps: int, seed: Optional[int] = None):
        if kappa <= 0.0:
            raise ValueError("均值回归速率 κ 必须为正.")
        if sigma < 0.0:
            raise ValueError("波动率 σ 必须非负.")
        if n_steps <= 0:
            raise ValueError("步数 N 必须为正整数.")

        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = t_max / n_steps

        if seed is not None:
            np.random.seed(seed)

    def exact_solution(self, t: np.ndarray) -> np.ndarray:


        raise NotImplementedError("Hole 1: 需补全 OU 过程精确矩公式")


    def simulate_rk2(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        s = np.zeros(self.n_steps + 1)
        s[0] = self.s0
        sqrt_dt = np.sqrt(self.dt)

        for n in range(self.n_steps):
            z_n = np.random.normal()
            drift_1 = -self.kappa * (s[n] - self.mu) * self.dt
            diff_1 = self.sigma * sqrt_dt * z_n
            k1 = drift_1 + diff_1

            drift_2 = -self.kappa * (s[n] + k1 - self.mu) * self.dt
            diff_2 = self.sigma * sqrt_dt * z_n
            k2 = drift_2 + diff_2

            s[n + 1] = s[n] + 0.5 * (k1 + k2)


            if s[n + 1] <= 0.0:
                s[n + 1] = 1e-6

        return t, s

    def simulate_exact_milstein(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        s = np.zeros(self.n_steps + 1)
        s[0] = self.s0

        exp_kdt = np.exp(-self.kappa * self.dt)
        std_factor = self.sigma * np.sqrt(
            (1.0 - np.exp(-2.0 * self.kappa * self.dt)) / (2.0 * self.kappa)
        )

        for n in range(self.n_steps):
            z_n = np.random.normal()
            s[n + 1] = s[n] * exp_kdt + self.mu * (1.0 - exp_kdt) + std_factor * z_n
            if s[n + 1] <= 0.0:
                s[n + 1] = 1e-6

        return t, s


class StiffRelaxation:

    def __init__(self, lam: float, omega: float, y0: float,
                 t_max: float, n_steps: int):
        if lam <= 0.0:
            raise ValueError("刚性参数 λ 必须为正.")
        if n_steps <= 0:
            raise ValueError("步数 N 必须为正整数.")

        self.lam = lam
        self.omega = omega
        self.y0 = y0
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = t_max / n_steps

    def derivative(self, t: float, y: float) -> float:
        return self.lam * (np.cos(self.omega * t) - y)

    def solve_rk2(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        y = np.zeros(self.n_steps + 1)
        y[0] = self.y0

        for n in range(self.n_steps):
            k1 = self.dt * self.derivative(t[n], y[n])
            k2 = self.dt * self.derivative(t[n] + self.dt, y[n] + k1)
            y[n + 1] = y[n] + 0.5 * (k1 + k2)

        return t, y

    def solve_exact(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        exp_lt = np.exp(-self.lam * t)
        factor = self.lam / (self.lam ** 2 + self.omega ** 2)
        y = (self.y0 * exp_lt
             + factor * (self.lam * np.cos(self.omega * t)
                         + self.omega * np.sin(self.omega * t)
                         - self.lam * exp_lt))
        return t, y


class StabilityAnalysis:

    @staticmethod
    def rk2_amplification(z_real: np.ndarray, z_imag: np.ndarray) -> np.ndarray:
        Z = z_real + 1j * z_imag
        R = 1.0 + Z + 0.5 * Z ** 2
        return np.abs(R)

    @staticmethod
    def is_stable(z_real: float, z_imag: float) -> bool:
        return StabilityAnalysis.rk2_amplification(
            np.array([[z_real]]), np.array([[z_imag]])
        )[0, 0] <= 1.0 + 1e-10

    @staticmethod
    def maximum_stable_step(lambda_max: float) -> float:
        if lambda_max <= 0.0:
            raise ValueError("lambda_max 必须为正.")


        raise NotImplementedError("Hole 2: 需补全 RK2 稳定性边界公式")



class ParameterSweep:

    def __init__(self, kappa_vals: np.ndarray, sigma_vals: np.ndarray,
                 mu: float = 100.0, s0: float = 100.0,
                 t_max: float = 1.0, n_steps: int = 10000):
        self.kappa_vals = kappa_vals
        self.sigma_vals = sigma_vals
        self.mu = mu
        self.s0 = s0
        self.t_max = t_max
        self.n_steps = n_steps

    def sweep_half_life(self) -> np.ndarray:
        k_grid, s_grid = np.meshgrid(self.kappa_vals, self.sigma_vals, indexing='ij')
        half_life = np.log(2.0) / k_grid
        return half_life

    def sweep_stationary_variance(self) -> np.ndarray:
        k_grid, s_grid = np.meshgrid(self.kappa_vals, self.sigma_vals, indexing='ij')
        var_inf = s_grid ** 2 / (2.0 * k_grid)
        return var_inf

    def sweep_peak_volatility(self, n_paths: int = 50) -> np.ndarray:
        m = len(self.kappa_vals)
        n = len(self.sigma_vals)
        peak_vals = np.full((m, n), np.nan)

        for i, kappa in enumerate(self.kappa_vals):
            for j, sigma in enumerate(self.sigma_vals):
                ou = OrnsteinUhlenbeck(
                    kappa=kappa, mu=self.mu, sigma=sigma,
                    s0=self.s0, t_max=self.t_max, n_steps=self.n_steps, seed=42
                )
                peaks = []
                for _ in range(n_paths):
                    t, s = ou.simulate_exact_milstein()
                    peaks.append(np.max(np.abs(s - self.mu)))
                peak_vals[i, j] = np.mean(peaks)

        return peak_vals
