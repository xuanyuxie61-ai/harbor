
import numpy as np
from typing import Tuple, Optional


class QuadratureRule:

    def __init__(self, x: np.ndarray, w: np.ndarray, a: float = -1.0, b: float = 1.0):
        self.x = np.array(x, dtype=float)
        self.w = np.array(w, dtype=float)
        self.a = a
        self.b = b
        self.n = len(x)

    def integrate(self, f) -> float:
        return float(np.sum(self.w * f(self.x)))

    def scale_to(self, a_new: float, b_new: float) -> "QuadratureRule":
        if self.b <= self.a:
            raise ValueError("原始区间无效")
        scale = (b_new - a_new) / (self.b - self.a)
        shift = (a_new + b_new - (self.a + self.b) * scale) / 2.0
        x_new = self.x * scale + shift
        w_new = self.w * scale
        return QuadratureRule(x_new, w_new, a_new, b_new)


def jacobi_gw(n: int, alpha: float, beta: float) -> QuadratureRule:
    if n < 1:
        return QuadratureRule(np.array([]), np.array([]))
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Jacobi 参数必须满足 alpha > -1, beta > -1")

    ab = alpha + beta


    diag = np.zeros(n)
    diag[0] = (beta - alpha) / (ab + 2.0)
    if n > 1:
        a2b2 = beta**2 - alpha**2
        for i in range(1, n):
            idx = i + 1
            abi = 2.0 * idx + ab
            diag[i] = a2b2 / ((abi - 2.0) * abi)


    sub = np.zeros(n - 1)
    if n > 1:
        sub[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta)
                         / ((ab + 3.0) * (ab + 2.0)**2))
        for i in range(1, n - 1):
            idx = i + 1
            abi = 2.0 * idx + ab
            numer = 4.0 * idx * (idx + alpha) * (idx + beta) * (idx + ab)
            denom = (abi**2 - 1.0) * abi**2
            sub[i] = np.sqrt(numer / denom)


    J = np.diag(diag) + np.diag(sub, k=1) + np.diag(sub, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)


    log_mu0 = (ab + 1.0) * np.log(2.0) \
        + np.math.lgamma(alpha + 1.0) + np.math.lgamma(beta + 1.0) - np.math.lgamma(ab + 2.0)
    mu0 = np.exp(log_mu0)

    weights = mu0 * eigenvectors[0, :]**2
    nodes = eigenvalues

    return QuadratureRule(nodes, weights, -1.0, 1.0)


def legendre_gauss(n: int) -> QuadratureRule:
    return jacobi_gw(n, 0.0, 0.0)


def chebyshev_gauss_first(n: int) -> QuadratureRule:
    return jacobi_gw(n, -0.5, -0.5)


def compute_fermi_dirac_integral(k: int, eta: float, n_quad: int = 64) -> float:
    if n_quad < 1:
        return 0.0

    quad = legendre_gauss(n_quad)

    def integrand(t):
        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2

        exp_arg = np.clip(x - eta, -700.0, 700.0)
        denom = np.exp(exp_arg) + 1.0
        return (x**k) / denom * dx_dt

    return quad.integrate(integrand)


def planck_mean_opacity_integral(T: float, n_quad: int = 32) -> float:
    quad = legendre_gauss(n_quad)

    def integrand_weight(t):

        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2
        exp_x = np.clip(x, 1.0e-10, 700.0)
        planck_weight = x**3 / (np.exp(exp_x) - 1.0)
        return planck_weight * dx_dt


    norm = quad.integrate(integrand_weight)
    if norm < 1.0e-30:
        return 0.0
    return norm


def rosseland_mean_weight(n_quad: int = 32) -> float:
    quad = legendre_gauss(n_quad)

    def integrand(t):
        x = (1.0 + t) / (1.0 - t + 1.0e-15)
        dx_dt = 2.0 / (1.0 - t + 1.0e-15)**2
        exp_x = np.clip(x, 1.0e-10, 700.0)
        weight = x**4 * np.exp(exp_x) / (np.exp(exp_x) - 1.0)**2
        return weight * dx_dt

    return quad.integrate(integrand)
