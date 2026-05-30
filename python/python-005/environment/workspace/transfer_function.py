# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable
from utils import spherical_bessel_j, clip_to_unit


class ChebyshevInterpolator:

    def __init__(self, a: float, b: float, n: int, f: Callable[[float], float]):
        if b <= a:
            raise ValueError("插值区间必须满足 b > a")
        if n < 2:
            raise ValueError("Chebyshev 节点数 n 必须 ≥ 2")
        self.a = a
        self.b = b
        self.n = n
        self.coeffs = self._compute_coefficients(f)

    def _xt(self, x: float) -> float:
        return (2.0 * x - self.a - self.b) / (self.b - self.a)

    def _compute_coefficients(self, f: Callable[[float], float]) -> np.ndarray:
        i = np.arange(1, self.n + 1)
        x_tilde = np.cos((2.0 * i - 1.0) * np.pi / (2.0 * self.n))
        x_nodes = 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * x_tilde
        f_vals = np.array([f(x) for x in x_nodes])

        c = np.zeros(self.n)
        for j in range(self.n):
            Tj = np.cos(j * np.arccos(clip_to_unit(x_tilde)))
            c[j] = (2.0 / self.n) * np.sum(f_vals * Tj)
        return c

    def evaluate(self, x: float) -> float:
        xt = clip_to_unit(self._xt(x))

        b2 = 0.0
        b1 = 0.0
        for j in range(self.n - 1, 0, -1):
            b0 = 2.0 * xt * b1 - b2 + self.coeffs[j]
            b2 = b1
            b1 = b0
        return 0.5 * (self.coeffs[0] + b1 * xt - b2)

    def evaluate_array(self, x_arr: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(x) for x in x_arr])


class TransferFunctionComputer:

    def __init__(self, lmax: int = 100, k_min: float = 1e-4,
                 k_max: float = 1.0, n_cheb: int = 32):
        self.lmax = lmax
        self.k_min = k_min
        self.k_max = k_max
        self.n_cheb = n_cheb

        self.eta0 = 14000.0
        self.eta_rec = 280.0
        self.k_D = 0.14
        self.k_p = 0.05

        self.interpolators = {}

    def _transfer_analytic(self, l: int, k: float) -> float:


        raise NotImplementedError("Hole_2: 请补全 _transfer_analytic 的实现")

    def build_interpolator(self, l: int) -> ChebyshevInterpolator:
        def Tl_of_k(k: float) -> float:
            return self._transfer_analytic(l, k)
        return ChebyshevInterpolator(self.k_min, self.k_max, self.n_cheb, Tl_of_k)

    def precompute_all(self):
        for l in range(2, self.lmax + 1):
            self.interpolators[l] = self.build_interpolator(l)

    def get_transfer(self, l: int, k: float) -> float:
        if l < 2:
            return 1.0
        if l not in self.interpolators:

            self.interpolators[l] = self.build_interpolator(l)
        k_clipped = max(self.k_min, min(self.k_max, k))
        return self.interpolators[l].evaluate(k_clipped)
