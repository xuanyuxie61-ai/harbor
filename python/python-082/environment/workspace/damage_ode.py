# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple


class PhaseFieldDamageModel:

    def __init__(self, E0: float, Gc: float, c0: float = 3.0 / 8.0,
                 eta: float = 1e-6, tau: float = 1e-3):
        if E0 <= 0 or Gc <= 0 or tau <= 0:
            raise ValueError("E0, Gc, tau must be positive.")
        self.E0 = E0
        self.Gc = Gc
        self.c0 = c0
        self.eta = eta
        self.tau = tau
        self.Yc = Gc / c0

    def degradation(self, d: np.ndarray) -> np.ndarray:
        d = np.clip(d, 0.0, 1.0)
        return (1.0 - d) ** 2 + self.eta

    def elastic_energy(self, epsilon: float) -> float:
        return 0.5 * self.E0 * epsilon ** 2

    def damage_driving_force(self, epsilon: float, d: float) -> float:
        d_clip = np.clip(d, 0.0, 1.0)
        return 2.0 * (1.0 - d_clip) * self.elastic_energy(epsilon)

    def evolution_rate(self, epsilon: float, d: float) -> float:
        if d <= 0.0:
            return max(0.0, (self.damage_driving_force(epsilon, d) - self.Yc) / self.tau)
        elif d >= 1.0:
            return 0.0
        else:
            rate = (self.damage_driving_force(epsilon, d) - self.Yc) / self.tau
            return max(0.0, rate)

    def integrate(self, epsilon_history: Callable[[float], float],
                  d0: float, t_span: Tuple[float, float],
                  num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        def ode_func(t, y):
            d = y[0]
            eps = epsilon_history(t)
            dd_dt = self.evolution_rate(eps, d)
            return [dd_dt]

        sol = solve_ivp(ode_func, t_span, [d0], dense_output=True,
                        max_step=(t_span[1] - t_span[0]) / num_points,
                        method='RK45')
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        d_sol = sol.sol(t_eval)[0]

        d_sol = np.clip(d_sol, 0.0, 1.0)
        return t_eval, d_sol


class FastSlowDamageODE:

    def __init__(self, eps_fast: float = 1e-3,
                 sigma_threshold: float = 100e6,
                 sigma_uts: float = 1500e6):
        if eps_fast <= 0:
            raise ValueError("eps_fast must be positive.")
        self.eps = eps_fast
        self.sigma_th = sigma_threshold
        self.sigma_uts = sigma_uts

    def _a_parameter(self, sigma_eq: float) -> float:
        a0 = 3.0
        sigma_ref = 0.2 * self.sigma_uts
        return a0 * (1.0 - np.tanh((sigma_eq - self.sigma_th) / (sigma_ref + 1e-30)))

    def _gamma_parameter(self, sigma_eq: float) -> float:
        gamma0 = 0.5
        return gamma0 * (sigma_eq / (self.sigma_uts + 1e-30)) ** 2

    def rhs(self, t: float, state: np.ndarray,
            sigma_eq_func: Callable[[float], float]) -> np.ndarray:
        d, s = state
        sigma_eq = sigma_eq_func(t)
        a = self._a_parameter(sigma_eq)
        gamma = self._gamma_parameter(sigma_eq)


        dd_dt = -(d ** 3 - a * d + s) / (self.eps + 1e-30)

        ds_dt = d - gamma


        if d <= 0.0 and dd_dt < 0:
            dd_dt = 0.0
        if d >= 1.0 and dd_dt > 0:
            dd_dt = 0.0

        return np.array([dd_dt, ds_dt])

    def integrate(self, sigma_eq_func: Callable[[float], float],
                  d0: float, s0: float,
                  t_span: Tuple[float, float],
                  num_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def ode_func(t, y):
            return self.rhs(t, y, sigma_eq_func)


        sol = solve_ivp(ode_func, t_span, [d0, s0],
                        dense_output=True,
                        max_step=(t_span[1] - t_span[0]) / num_points,
                        method='BDF')
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y_sol = sol.sol(t_eval)
        d_sol = np.clip(y_sol[0], 0.0, 1.0)
        s_sol = y_sol[1]
        return t_eval, d_sol, s_sol


class FatigueDamageModel:

    def __init__(self, C: float = 5.0e-12, m: float = 9.0, q: float = 2.0,
                 sigma_ref: float = 1.0e6, sigma_uts: float = 1500e6):
        self.C = C
        self.m = m
        self.q = q
        self.sigma_ref = sigma_ref
        self.sigma_uts = sigma_uts

    def cycles_to_failure(self, delta_sigma: float, sigma_mean: float = 0.0) -> float:
        if delta_sigma <= 0:
            return np.inf

        goodman_factor = 1.0 - sigma_mean / (self.sigma_uts + 1e-30)
        if goodman_factor <= 0:
            return 0.0
        delta_sigma_eff = delta_sigma / goodman_factor
        exponent = (delta_sigma_eff / self.sigma_ref) ** self.m
        Nf = 1.0 / ((self.q + 1.0) * self.C * exponent)
        return Nf

    def damage_after_cycles(self, N: float, delta_sigma: float,
                            sigma_mean: float = 0.0) -> float:
        if N <= 0:
            return 0.0
        goodman_factor = 1.0 - sigma_mean / (self.sigma_uts + 1e-30)
        if goodman_factor <= 0:
            return 1.0
        delta_sigma_eff = delta_sigma / goodman_factor
        A = self.C * (delta_sigma_eff / self.sigma_ref) ** self.m
        val = 1.0 - (self.q + 1.0) * A * N
        if val <= 0:
            return 1.0
        return 1.0 - val ** (1.0 / (self.q + 1.0))

    def asymptotic_period_large_mu(self, delta_sigma: float) -> float:
        if delta_sigma <= 0:
            return np.inf
        mu_eff = (self.sigma_ref / delta_sigma) ** (self.m / 2.0)
        return (3.0 - 2.0 * np.log(2.0)) * mu_eff


if __name__ == "__main__":

    pf = PhaseFieldDamageModel(E0=100e9, Gc=500.0, tau=1e-4)
    eps_hist = lambda t: 0.01 * np.sin(2 * np.pi * 100 * t) + 0.005
    t, d = pf.integrate(eps_hist, d0=0.0, t_span=(0.0, 0.01))
    print("Phase-field damage final:", d[-1])


    fs = FastSlowDamageODE(eps_fast=1e-3)
    sigma_eq = lambda t: 200e6 + 100e6 * np.sin(2 * np.pi * 50 * t)
    t2, d2, s2 = fs.integrate(sigma_eq, d0=0.1, s0=0.0, t_span=(0.0, 0.02))
    print("Fast-slow damage final:", d2[-1], "s final:", s2[-1])


    fat = FatigueDamageModel(C=1e-11, m=8.0, q=2.5)
    Nf = fat.cycles_to_failure(delta_sigma=200e6, sigma_mean=50e6)
    print("Cycles to failure:", Nf)
    d_fat = fat.damage_after_cycles(N=Nf / 2, delta_sigma=200e6, sigma_mean=50e6)
    print("Damage at half life:", d_fat)
