# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma as Gamma
from typing import Tuple, Optional


class SphereQuadrature:

    @staticmethod
    def sphere01_area() -> float:
        return 4.0 * np.pi

    @staticmethod
    def monomial_integral_exact(e: Tuple[int, int, int]) -> float:
        e1, e2, e3 = e
        if e1 < 0 or e2 < 0 or e3 < 0:
            raise ValueError("指数必须非负")
        if (e1 % 2 == 1) or (e2 % 2 == 1) or (e3 % 2 == 1):
            return 0.0
        num = 2.0 * Gamma(0.5 * (e1 + 1)) * Gamma(0.5 * (e2 + 1)) * Gamma(0.5 * (e3 + 1))
        den = Gamma(0.5 * (e1 + e2 + e3 + 3))
        return float(num / den)

    @staticmethod
    def evaluate_monomial(points: np.ndarray, e: Tuple[int, int, int]) -> np.ndarray:
        e1, e2, e3 = e
        return (points[:, 0] ** e1) * (points[:, 1] ** e2) * (points[:, 2] ** e3)

    @staticmethod
    def uniform_sample(n: int, seed: Optional[int] = None) -> np.ndarray:
        if n <= 0:
            raise ValueError("n 必须 > 0")
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(size=(n, 3))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return g / norms

    def monte_carlo_integral(self, n_samples: int, e: Tuple[int, int, int], seed: Optional[int] = None) -> Tuple[float, float]:
        pts = self.uniform_sample(n_samples, seed)
        f_vals = self.evaluate_monomial(pts, e)
        mean_f = np.mean(f_vals)
        std_f = np.std(f_vals, ddof=1)
        A = self.sphere01_area()
        I_est = A * mean_f
        I_err = A * std_f / np.sqrt(n_samples)
        return I_est, I_err


class Vandermonde2DQuadrature:

    @staticmethod
    def compute_weights(nodes_x: np.ndarray,
                        nodes_y: np.ndarray,
                        total_degree: int,
                        rect_a: float = -1.0,
                        rect_b: float = 1.0,
                        rect_c: float = -1.0,
                        rect_d: float = 1.0,
                        rcond: float = 1e-12) -> np.ndarray:
        n_expected = (total_degree + 1) * (total_degree + 2) // 2
        n = len(nodes_x)
        if n != n_expected:
            raise ValueError(f"节点数 {n} 与期望 {n_expected} 不符，t={total_degree}")


        exponents = []
        for i in range(total_degree + 1):
            for j in range(total_degree + 1 - i):
                exponents.append((i, j))


        V = np.zeros((n, n), dtype=float)
        for k, (i, j) in enumerate(exponents):
            V[k, :] = (nodes_x ** i) * (nodes_y ** j)


        rhs = np.zeros(n, dtype=float)
        for k, (i, j) in enumerate(exponents):
            int_x = (rect_b ** (i + 1) - rect_a ** (i + 1)) / (i + 1)
            int_y = (rect_d ** (j + 1) - rect_c ** (j + 1)) / (j + 1)
            rhs[k] = int_x * int_y


        w = np.linalg.lstsq(V, rhs, rcond=rcond)[0]
        return w

    @staticmethod
    def integrate(values: np.ndarray, weights: np.ndarray) -> float:
        if len(values) != len(weights):
            raise ValueError("values 与 weights 长度不一致")
        return float(np.dot(weights, values))


class GaussLegendreTensor:

    @staticmethod
    def tensor_quad_2d(f, a: float, b: float, c: float, d: float, m: int = 8):
        from numpy.polynomial.legendre import leggauss
        xi, wi = leggauss(m)

        x_phys = 0.5 * (b - a) * xi + 0.5 * (b + a)
        y_phys = 0.5 * (d - c) * xi + 0.5 * (d + c)
        J = 0.25 * (b - a) * (d - c)
        total = 0.0
        for p in range(m):
            for q in range(m):
                total += wi[p] * wi[q] * f(x_phys[p], y_phys[q])
        return total * J


