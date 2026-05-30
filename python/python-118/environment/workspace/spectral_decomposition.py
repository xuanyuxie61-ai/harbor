
import numpy as np
from scipy.special import jv, spherical_jn
from utils_numeric import (
    laguerre_polynomial_alpha, gegenbauer_polynomial,
    bessel_zero_newton, safe_sqrt
)


class RDFSpectralExpansion:

    def __init__(self, alpha=2.0, beta=1.0, n_modes=10):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_modes = int(n_modes)

    def expand(self, r_grid, g_r):
        coeffs = np.zeros(self.n_modes, dtype=np.float64)
        r = r_grid
        dr = np.gradient(r)

        for n in range(self.n_modes):
            Ln = laguerre_polynomial_alpha(self.beta * r, n, self.alpha)

            weight = (self.beta * r) ** self.alpha * np.exp(-self.beta * r)

            from scipy.special import gamma as gamma_func
            norm = gamma_func(n + self.alpha + 1) / np.math.factorial(n)
            numerator = np.sum(g_r * Ln * weight * dr)
            coeffs[n] = numerator / (norm + 1e-15)


        g_reconstructed = np.zeros_like(r)
        for n in range(self.n_modes):
            Ln = laguerre_polynomial_alpha(self.beta * r, n, self.alpha)
            g_reconstructed += coeffs[n] * Ln

        return coeffs, g_reconstructed

    def compute_structure_index(self, coeffs):
        n = np.arange(len(coeffs))
        S = np.sum((-1.0) ** n * np.abs(coeffs) / (n + 1.0))
        return S


class BesselModeAnalysis:

    def __int__(self, max_n=3, max_k=5, R=10.0):
        self.max_n = int(max_n)
        self.max_k = int(max_k)
        self.R = float(R)
        self._precompute_zeros()

    def _precompute_zeros(self):
        self.zeros = {}
        for n in range(self.max_n + 1):
            for k in range(1, self.max_k + 1):
                self.zeros[(n, k)] = bessel_zero_newton(float(n), k, kind=1)

    def compute_mode_amplitudes(self, h_field, r_grid, theta_grid):

        amplitudes = {}
        for n in range(self.max_n + 1):
            for k in range(1, self.max_k + 1):
                alpha_nk = self.zeros[(n, k)]

                amp = 0.0
                amplitudes[(n, k)] = amp
        return amplitudes


class GegenbauerAngularExpansion:

    def __init__(self, lambda_param=0.5, max_degree=8):
        self.lambda_param = float(lambda_param)
        self.max_degree = int(max_degree)

    def expand_angular_distribution(self, theta, f_theta):
        x = np.cos(theta)
        dx = np.gradient(x)

        weight = (1.0 - x ** 2) ** (self.lambda_param - 0.5)
        weight = np.where(x ** 2 < 1.0, weight, 0.0)

        coeffs = np.zeros(self.max_degree + 1, dtype=np.float64)
        for l in range(self.max_degree + 1):
            Cl = gegenbauer_polynomial(x, l, self.lambda_param)

            from scipy.special import gamma as gamma_func
            norm = (np.pi * 2.0 ** (1.0 - 2.0 * self.lambda_param) *
                    gamma_func(l + 2.0 * self.lambda_param) /
                    (gamma_func(self.lambda_param) ** 2 * (l + self.lambda_param) *
                     np.math.factorial(l)))
            coeffs[l] = np.sum(f_theta * Cl * weight * dx) / (norm + 1e-15)


        f_reconstructed = np.zeros_like(theta)
        for l in range(self.max_degree + 1):
            Cl = gegenbauer_polynomial(x, l, self.lambda_param)
            f_reconstructed += coeffs[l] * Cl

        return coeffs, f_reconstructed


class SpectralEntropy:

    @staticmethod
    def shannon_entropy(coeffs):
        probs = np.abs(coeffs) ** 2
        probs /= (np.sum(probs) + 1e-15)
        return -np.sum(probs * np.log(probs + 1e-15))

    @staticmethod
    def participation_ratio(coeffs):
        p2 = np.sum(np.abs(coeffs) ** 2)
        p4 = np.sum(np.abs(coeffs) ** 4)
        return p2 ** 2 / (p4 + 1e-15)
