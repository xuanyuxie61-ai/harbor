# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
from typing import Callable, Optional, Tuple


class WaveEquation1D:

    def __init__(self, L: float, nx: int,
                 rho: float, E: float, A: float = 1.0,
                 damping: float = 0.0):
        if nx < 3:
            raise ValueError("nx must be >= 3.")
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.rho = rho
        self.E = E
        self.A = A
        self.damping = damping
        self.c = np.sqrt(E / rho)
        self.x = np.linspace(0.0, L, nx)

    def _rhs_fd(self, y: np.ndarray, t: float,
                f_func: Optional[Callable], damage_field: Optional[np.ndarray]) -> np.ndarray:
        nx = self.nx
        u = y[:nx]
        v = y[nx:]


        dudt = v.copy()


        eps = np.zeros(nx)
        eps[1:-1] = (u[2:] - u[:-2]) / (2.0 * self.dx)
        eps[0] = (u[1] - u[0]) / self.dx
        eps[-1] = (u[-1] - u[-2]) / self.dx


        E_local = self.E * np.ones(nx)
        if damage_field is not None:
            if len(damage_field) != nx:
                raise ValueError("damage_field length must match nx.")
            g_d = (1.0 - np.clip(damage_field, 0.0, 1.0)) ** 2 + 1e-6
            E_local *= g_d

        sigma = E_local * eps


        dsigma_dx = np.zeros(nx)
        dsigma_dx[1:-1] = (sigma[2:] - sigma[:-2]) / (2.0 * self.dx)
        dsigma_dx[0] = (sigma[1] - sigma[0]) / self.dx
        dsigma_dx[-1] = (sigma[-1] - sigma[-2]) / self.dx


        f_ext = np.zeros(nx)
        if f_func is not None:
            f_ext = f_func(self.x, t)


        dvdt = (self.A * dsigma_dx - self.damping * v + f_ext) / (self.rho * self.A)


        dudt[0] = 0.0
        dvdt[0] = 0.0
        dvdt[-1] = 0.0

        return np.concatenate([dudt, dvdt])

    def solve(self, u0: np.ndarray, v0: np.ndarray,
              t_span: Tuple[float, float], nt: int,
              f_func: Optional[Callable] = None,
              damage_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.linspace(t_span[0], t_span[1], nt)
        y0 = np.concatenate([u0, v0])

        def rhs(y, t_val):
            return self._rhs_fd(y, t_val, f_func, damage_field)

        sol = odeint(rhs, y0, t)
        u_hist = sol[:, :self.nx]
        v_hist = sol[:, self.nx:]
        return t, u_hist, v_hist

    def modal_analysis(self, num_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        n = np.arange(1, num_modes + 1)
        omega_n = (2.0 * n - 1.0) * np.pi * self.c / (2.0 * self.L)
        phi = np.zeros((num_modes, self.nx))
        for i, nn in enumerate(n):
            phi[i, :] = np.sin((2.0 * nn - 1.0) * np.pi * self.x / (2.0 * self.L))
        return omega_n, phi

    def forced_response_amplitude(self, x_force: float, omega: float,
                                   num_modes: int = 20) -> np.ndarray:
        omega_n, phi = self.modal_analysis(num_modes=num_modes)
        zeta = self.damping / (2.0 * self.rho * self.A * omega_n + 1e-30)
        denom = np.sqrt((omega_n ** 2 - omega ** 2) ** 2 + (2.0 * zeta * omega_n * omega) ** 2)
        coeff = (2.0 / (self.rho * self.A * self.L)) * phi[:, int(x_force / self.L * (self.nx - 1))] / denom
        U = np.sum(coeff[:, None] * phi, axis=0)
        return U


class ImpactLoad:

    @staticmethod
    def half_sine_pulse(t: float, F0: float, duration: float) -> float:
        if t < 0 or t > duration:
            return 0.0
        return F0 * np.sin(np.pi * t / duration)

    @staticmethod
    def triangular_pulse(t: float, F0: float, duration: float) -> float:
        if t < 0 or t > duration:
            return 0.0
        if t <= duration / 2.0:
            return 2.0 * F0 * t / duration
        return 2.0 * F0 * (1.0 - t / duration)

    @staticmethod
    def blast_wave(t: float, F0: float, tau_rise: float, tau_decay: float) -> float:
        if t < 0:
            return 0.0
        return F0 * (1.0 - t / (tau_decay + 1e-30)) * np.exp(-t / (tau_rise + 1e-30))


class WaveReflectionAnalysis:

    @staticmethod
    def reflection_coefficient(rho1: float, c1: float, rho2: float, c2: float) -> float:
        Z1 = rho1 * c1
        Z2 = rho2 * c2
        return (Z2 - Z1) / (Z2 + Z1 + 1e-30)

    @staticmethod
    def transmission_coefficient(rho1: float, c1: float, rho2: float, c2: float) -> float:
        Z1 = rho1 * c1
        Z2 = rho2 * c2
        return 2.0 * Z2 / (Z2 + Z1 + 1e-30)

    @staticmethod
    def damage_reflection_approx(E0: float, rho: float, d_local: float) -> float:
        c0 = np.sqrt(E0 / rho)
        g_d = (1.0 - np.clip(d_local, 0.0, 0.99)) ** 2 + 1e-6
        c_d = c0 * np.sqrt(g_d)
        return WaveReflectionAnalysis.reflection_coefficient(rho, c0, rho, c_d)


if __name__ == "__main__":

    wave = WaveEquation1D(L=1.0, nx=101, rho=1600.0, E=100e9, A=1e-4, damping=100.0)


    omega_n, phi = wave.modal_analysis(num_modes=3)
    print("Natural frequencies (Hz):", omega_n / (2 * np.pi))


    U = wave.forced_response_amplitude(x_force=0.5, omega=omega_n[0] * 0.9)
    print("Forced response max amplitude:", np.max(np.abs(U)))


    R = WaveReflectionAnalysis.damage_reflection_approx(E0=100e9, rho=1600.0, d_local=0.3)
    print("Reflection coefficient for d=0.3:", R)
