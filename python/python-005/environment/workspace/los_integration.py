# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple
from utils import clip_to_unit, ensure_positive





def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    i = np.arange(1.0, n, dtype=float)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * eigvecs[0, :] ** 2
    return x, w





def fejer1_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(1, n + 1)
    x = np.cos((2.0 * i - 1.0) * np.pi / (2.0 * n))

    N = 2 * n
    v = np.zeros(N)
    v[0] = 2.0

    idx = np.arange(1, N, 2)
    v[idx] = 2.0 / (idx * (idx + 2))

    v_tilde = np.fft.ifft(v).real
    w = 2.0 * v_tilde[:n]
    return x, w


def fejer2_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(1, n + 1)
    x = np.cos(i * np.pi / (n + 1.0))

    k = np.arange(1, n + 1)
    w = np.zeros(n)
    for m in range(n):
        s = 0.0
        for j in range(1, (n + 1) // 2 + 1):
            s += np.sin((2.0 * j - 1.0) * k[m] * np.pi / (n + 1.0)) / (2.0 * j - 1.0)
        w[m] = 4.0 * np.sin(k[m] * np.pi / (n + 1.0)) * s / (n + 1.0)
    return x, w





def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n 必须 ≥ 1")
    i = np.arange(n + 1)
    x = np.cos(i * np.pi / n)

    w = np.zeros(n + 1)
    theta = i * np.pi / n

    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n ** 2 - 1.0)
        for k in range(1, n):
            s = 0.0
            for j in range(n // 2):
                s += np.cos(2.0 * j * theta[k]) / (4.0 * j ** 2 - 1.0)
            w[k] = 2.0 * (1.0 - s) / (n - 1.0)
            if k == 0 or k == n:
                w[k] *= 0.5
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
        for k in range(1, n):
            s = 0.0
            for j in range((n - 1) // 2 + 1):
                s += np.cos(2.0 * j * theta[k]) / (4.0 * j ** 2 - 1.0)
            w[k] = 2.0 * s / n

    w[0] *= 0.5
    w[n] *= 0.5
    return x, w





class FastQuadrature:

    RULES = ["gauss_legendre", "clenshaw_curtis", "fejer1", "fejer2"]

    def __init__(self, rule: str = "gauss_legendre", n: int = 64):
        if rule not in self.RULES:
            raise ValueError(f"不支持的求积规则: {rule}")
        self.rule = rule
        self.n = n
        self._precompute()

    def _precompute(self):
        if self.rule == "gauss_legendre":
            self.x_ref, self.w_ref = gauss_legendre_nodes_weights(self.n)
        elif self.rule == "clenshaw_curtis":
            self.x_ref, self.w_ref = clenshaw_curtis_nodes_weights(self.n)
        elif self.rule == "fejer1":
            self.x_ref, self.w_ref = fejer1_nodes_weights(self.n)
        elif self.rule == "fejer2":
            self.x_ref, self.w_ref = fejer2_nodes_weights(self.n)

    def integrate(self, f: callable, a: float, b: float) -> float:
        if b <= a:
            raise ValueError("积分上限必须大于下限")
        t = self.x_ref
        x = 0.5 * (b - a) * t + 0.5 * (a + b)
        fx = np.array([f(xi) for xi in x])
        return 0.5 * (b - a) * np.dot(self.w_ref, fx)





def los_integral_power_spectrum(l: int,
                                 transfer_l: callable,
                                 primordial_power: callable,
                                 k_min: float, k_max: float,
                                 n_quad: int = 128,
                                 rule: str = "gauss_legendre") -> float:


    raise NotImplementedError("Hole_3: 请补全 los_integral_power_spectrum 的实现")


def compute_sachs_wolfe_integral(l: int, k: float, eta_grid: np.ndarray,
                                  Delta0_grid: np.ndarray,
                                  Phi_grid: np.ndarray) -> float:
    eta0 = eta_grid[-1]

    eta_rec = 280.0
    sigma_rec = 30.0
    g = np.exp(-0.5 * ((eta_grid - eta_rec) / sigma_rec) ** 2) / (sigma_rec * np.sqrt(2.0 * np.pi))
    source = g * (Delta0_grid / 4.0 + Phi_grid)
    arg = k * (eta0 - eta_grid)
    from utils import spherical_bessel_j
    jvals = np.array([spherical_bessel_j(l, a) for a in arg])
    integrand = source * jvals
    return np.trapezoid(integrand, eta_grid)
