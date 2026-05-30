
import numpy as np
from typing import Tuple, Optional


class ShiftedLegendreBasis:

    def __init__(self, domain_length: float, max_degree: int):
        if domain_length <= 0:
            raise ValueError("域长度必须为正")
        if max_degree < 0:
            raise ValueError("最高阶数必须非负")
        self.L = domain_length
        self.N = max_degree

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < -1e-12) or np.any(x > self.L + 1e-12):

            x = np.clip(x, 0.0, self.L)

        m = len(x)
        n = self.N
        v = np.zeros((m, n + 1))

        v[:, 0] = 1.0
        if n >= 1:
            y = 2.0 * x / self.L - 1.0
            v[:, 1] = y
            for i in range(1, n):
                v[:, i + 1] = (
                    (2.0 * i + 1.0) * y * v[:, i]
                    - i * v[:, i - 1]
                ) / (i + 1.0)

        return v

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0.0, self.L)
        m = len(x)
        n = self.N

        dv = np.zeros((m, n + 1))
        if n >= 1:
            y = 2.0 * x / self.L - 1.0

            dp = np.zeros((m, n + 1))
            if n >= 1:
                dp[:, 1] = 1.0
            for i in range(1, n):
                dp[:, i + 1] = (
                    (2.0 * i + 1.0) * (y * dp[:, i] + v[:, i])
                    - i * dp[:, i - 1]
                ) / (i + 1.0)

            v = self.evaluate(x)
            for i in range(1, n):
                dp[:, i + 1] = (
                    (2.0 * i + 1.0) * (y * dp[:, i] + v[:, i])
                    - i * dp[:, i - 1]
                ) / (i + 1.0)
            dv = (2.0 / self.L) * dp

        return dv

    def inner_product(self, f: np.ndarray, g: np.ndarray) -> float:

        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(self.N + 5)

        x_gl = 0.5 * self.L * (nodes + 1.0)
        w_gl = 0.5 * self.L * weights
        return float(np.sum(f(x_gl) * g(x_gl) * w_gl))


class MajoranaWavefunctionSolver:

    def __init__(self, length: float, n_sites: int,
                 mu: float, t: float, delta: float):
        self.L = length
        self.n = n_sites
        self.mu = mu
        self.t = t
        self.delta = delta
        self.dx = length / (n_sites - 1)

    def analytical_zero_mode_profile(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xi = 2.0 * abs(self.t) / (abs(self.delta) + 1e-15)

        psi = np.exp(-x / xi)

        norm = np.sqrt(np.trapezoid(psi ** 2, x))
        if norm > 1e-15:
            psi /= norm
        return psi

    def spectral_expansion_coefficients(self, wavefunction: np.ndarray,
                                        max_degree: int = 20) -> np.ndarray:
        basis = ShiftedLegendreBasis(self.L, max_degree)
        x = np.linspace(0.0, self.L, self.n)
        v = basis.evaluate(x)

        coeffs = np.zeros(max_degree + 1)
        for n in range(max_degree + 1):
            integrand = wavefunction * v[:, n]
            coeffs[n] = ((2.0 * n + 1.0) / self.L
                         * np.trapezoid(integrand, x))

        return coeffs

    def reconstruct_from_spectral(self, coeffs: np.ndarray,
                                   x: np.ndarray) -> np.ndarray:
        max_degree = len(coeffs) - 1
        basis = ShiftedLegendreBasis(self.L, max_degree)
        v = basis.evaluate(x)
        return v @ coeffs

    def finite_difference_time_evolution(self,
                                          initial_wave: np.ndarray,
                                          num_steps: int,
                                          dt: float,
                                          alpha: float) -> np.ndarray:
        if alpha < 0 or alpha > 1.0:
            raise ValueError("CFL参数alpha必须在[0,1]范围内")
        if len(initial_wave) != self.n:
            raise ValueError("初始波函数长度必须与格点数匹配")

        u = np.zeros((num_steps + 1, self.n))

        for j in range(num_steps + 1):
            if j == 0:
                u[0, 0] = 0.0
                u[0, 1:self.n - 1] = initial_wave[1:self.n - 1]
                u[0, self.n - 1] = 0.0
            elif j == 1:
                u[1, 0] = 0.0
                for i in range(1, self.n - 1):

                    u[1, i] = (
                        0.5 * alpha * u[0, i - 1]
                        + (1.0 - alpha) * u[0, i]
                        + 0.5 * alpha * u[0, i + 1]
                    )
                u[1, self.n - 1] = 0.0
            else:
                u[j, 0] = 0.0
                for i in range(1, self.n - 1):
                    u[j, i] = (
                        alpha * u[j - 1, i - 1]
                        + 2.0 * (1.0 - alpha) * u[j - 1, i]
                        + alpha * u[j - 1, i + 1]
                        - u[j - 2, i]
                    )
                u[j, self.n - 1] = 0.0

        return u

    def compute_probability_current(self, wavefunction: np.ndarray) -> np.ndarray:
        psi = np.asarray(wavefunction, dtype=np.complex128)
        dpsi = np.zeros_like(psi)
        dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * self.dx)
        dpsi[0] = (psi[1] - psi[0]) / self.dx
        dpsi[-1] = (psi[-1] - psi[-2]) / self.dx

        j = np.imag(np.conj(psi) * dpsi)
        return j

    def overlap_integral(self, psi1: np.ndarray,
                         psi2: np.ndarray) -> complex:
        if len(psi1) != len(psi2):
            raise ValueError("波函数长度必须相同")
        x = np.linspace(0.0, self.L, len(psi1))
        integrand = np.conj(psi1) * psi2
        return complex(np.trapezoid(integrand, x))


def demo():
    solver = MajoranaWavefunctionSolver(
        length=100.0, n_sites=100, mu=0.5, t=1.0, delta=0.8
    )
    x = np.linspace(0.0, solver.L, solver.n)
    psi = solver.analytical_zero_mode_profile(x)
    print("Analytical MZM profile norm:", np.trapezoid(psi ** 2, x))

    coeffs = solver.spectral_expansion_coefficients(psi, max_degree=15)
    print("Spectral coefficients (first 5):", coeffs[:5])

    psi_recon = solver.reconstruct_from_spectral(coeffs, x)
    error = np.max(np.abs(psi - psi_recon))
    print("Spectral reconstruction max error:", error)


    u = solver.finite_difference_time_evolution(
        initial_wave=psi, num_steps=50, dt=0.01, alpha=0.25
    )
    print("FD evolution final wave energy:", np.trapezoid(u[-1] ** 2, x))


if __name__ == "__main__":
    demo()
