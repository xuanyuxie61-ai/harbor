
import numpy as np
from scipy.integrate import odeint


class WavePropagation1D:

    def __init__(self, L, nx, E_func, rho, damping_ratio, forcing_params):
        self.L = float(L)
        self.nx = nx
        self.x = np.linspace(0.0, self.L, nx)
        self.dx = self.x[1] - self.x[0]
        self.E = np.array([E_func(xi) for xi in self.x])
        self.rho = float(rho)
        self.damping_ratio = float(damping_ratio)
        self.f_amp, self.f_omega, self.f_pos = forcing_params


        self.omega_n = np.pi / self.L * np.sqrt(np.mean(self.E) / self.rho)
        self.beta = 2.0 * self.damping_ratio * self.omega_n

    def deriv(self, w, t):
        nx = self.nx
        u = w[:nx]
        v = w[nx:]

        dudt = v.copy()
        dvdt = np.zeros(nx)


        for i in range(1, nx - 1):
            d2u = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / self.dx ** 2
            c2 = self.E[i] / self.rho
            dvdt[i] = c2 * d2u - self.beta * v[i]


        dvdt[0] = 0.0
        dvdt[-1] = 0.0
        dudt[0] = 0.0
        dudt[-1] = 0.0


        force = self.f_amp * np.cos(self.f_omega * t)

        for i in range(nx):
            dist = abs(self.x[i] - self.f_pos)
            if dist < 2.0 * self.dx:
                dvdt[i] += force / (self.rho * self.dx)

        return np.concatenate([dudt, dvdt])

    def solve(self, u0, v0, t_span):
        w0 = np.concatenate([u0, v0])
        sol = odeint(self.deriv, w0, t_span, rtol=1e-6, atol=1e-9)
        u_history = sol[:, :self.nx]
        v_history = sol[:, self.nx:]
        return u_history, v_history

    def compute_wave_speed(self):
        return np.sqrt(self.E / self.rho)

    def compute_attenuation_coefficient(self, frequency):
        c = np.mean(self.compute_wave_speed())
        omega = 2.0 * np.pi * frequency
        alpha = omega ** 2 * self.damping_ratio / (2.0 * c ** 3)
        return alpha


def compute_stress_wave_reflection_coefficient(E1, E2, rho1, rho2):
    c1 = np.sqrt(E1 / rho1)
    c2 = np.sqrt(E2 / rho2)
    Z1 = rho1 * c1
    Z2 = rho2 * c2
    R = (Z2 - Z1) / (Z2 + Z1 + 1e-12)
    T = 2.0 * Z2 / (Z1 + Z2 + 1e-12)
    return R, T
