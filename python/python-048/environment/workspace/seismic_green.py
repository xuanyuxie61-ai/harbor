
import numpy as np
from typing import Callable
from quadrature_rules import filon_cos_quad, filon_sin_quad


class SeismicGreen:

    def __init__(self, rho: float = 2650.0,
                 alpha: float = 4500.0,
                 beta: float = 2600.0):
        if alpha <= beta:
            raise ValueError("P 波速度必须大于 S 波速度")
        if rho <= 0 or alpha <= 0 or beta <= 0:
            raise ValueError("介质参数必须为正")
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

    def radiation_factors(self, x: np.ndarray, xi: np.ndarray) -> tuple:
        dx = x - xi
        r = np.linalg.norm(dx)
        if r < 1.0e-12:
            r = 1.0e-12
            gamma = np.array([0.0, 0.0, 1.0])
        else:
            gamma = dx / r
        return r, gamma

    def displacement_spectrum_farfield(self, x: np.ndarray, xi: np.ndarray,
                                        M: np.ndarray, omega: float) -> np.ndarray:





        raise NotImplementedError("Hole 1: 请实现远场位移谱公式")

    def time_domain_displacement_filon(self, x: np.ndarray, xi: np.ndarray,
                                        M: np.ndarray, t: float,
                                        omega_max: float = 200.0,
                                        n_omega: int = 401) -> np.ndarray:
        if n_omega % 2 == 0:
            n_omega += 1

        r, gamma = self.radiation_factors(x, xi)


        u = np.zeros(3, dtype=float)

        for comp in range(3):

            def amplitude(omega_arr):
                if np.isscalar(omega_arr):
                    omega_arr = np.array([omega_arr])
                else:
                    omega_arr = np.asarray(omega_arr)
                out = np.zeros_like(omega_arr, dtype=complex)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        out[idx] = 0.0
                    else:
                        us = self.displacement_spectrum_farfield(x, xi, M, w)
                        out[idx] = us[comp]
                return out



            def integrand_p(omega_arr):
                omega_arr = np.asarray(omega_arr, dtype=float)
                out = np.zeros_like(omega_arr, dtype=float)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        continue
                    coeff = 1.0j * w / (4.0 * np.pi * self.rho * r * self.alpha ** 3)
                    M_proj = np.sum(M * np.outer(gamma, gamma))
                    val = coeff * M_proj * np.exp(1.0j * w * (r / self.alpha - t))
                    out[idx] = np.real(val)
                return out

            def integrand_s(omega_arr):
                omega_arr = np.asarray(omega_arr, dtype=float)
                out = np.zeros_like(omega_arr, dtype=float)
                for idx, w in enumerate(omega_arr):
                    if w == 0:
                        continue
                    coeff = 1.0j * w / (4.0 * np.pi * self.rho * r * self.beta ** 3)
                    identity = np.eye(3)
                    Mgk = (identity - np.outer(gamma, gamma)) @ (M @ gamma)
                    val = coeff * Mgk[comp] * np.exp(1.0j * w * (r / self.beta - t))
                    out[idx] = np.real(val)
                return out



            omega_vals = np.linspace(0.0, omega_max, n_omega)
            d_omega = omega_vals[1] - omega_vals[0]

            f_p = integrand_p(omega_vals)
            f_s = integrand_s(omega_vals)


            def simpson_integral(y, h):
                if y.size < 3 or y.size % 2 == 0:
                    return np.trapezoid(y, dx=h)
                return h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2]))

            u[comp] = simpson_integral(f_p, d_omega) + simpson_integral(f_s, d_omega)


        return u

    def travel_time_p(self, x: np.ndarray, xi: np.ndarray) -> float:
        r, _ = self.radiation_factors(x, xi)
        return r / self.alpha

    def travel_time_s(self, x: np.ndarray, xi: np.ndarray) -> float:
        r, _ = self.radiation_factors(x, xi)
        return r / self.beta
