# -*- coding: utf-8 -*-

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.hermite import hermgauss
from typing import Callable, Tuple, Optional


class HermiteQuadrature:

    @staticmethod
    def probabilist_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("n must be >= 1.")
        nodes, weights = hermegauss(n)

        weights = weights / np.sqrt(2.0 * np.pi)
        return nodes, weights

    @staticmethod
    def physicist_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("n must be >= 1.")
        nodes, weights = hermgauss(n)
        return nodes, weights

    @staticmethod
    def exactness_test(n: int, max_degree: Optional[int] = None) -> bool:
        if max_degree is None:
            max_degree = 2 * n - 1
        nodes, weights = HermiteQuadrature.probabilist_nodes_weights(n)
        for k in range(max_degree + 1):
            exact_moment = 0.0 if k % 2 == 1 else np.prod(np.arange(1, k, 2), dtype=float)
            approx_moment = np.sum(weights * (nodes ** k))
            if not np.isclose(approx_moment, exact_moment, atol=1e-12):
                print(f"Exactness failed for degree {k}: exact={exact_moment}, approx={approx_moment}")
                return False
        return True


class HermitePolynomials:

    @staticmethod
    def probabilist_hermite(n: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return x.copy()
        H_prev2 = np.ones_like(x)
        H_prev1 = x.copy()
        H_curr = np.zeros_like(x)
        for k in range(1, n):
            H_curr = x * H_prev1 - k * H_prev2
            H_prev2, H_prev1 = H_prev1, H_curr
        return H_curr

    @staticmethod
    def physicist_hermite(n: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return 2.0 * x
        H_prev2 = np.ones_like(x)
        H_prev1 = 2.0 * x
        H_curr = np.zeros_like(x)
        for k in range(1, n):
            H_curr = 2.0 * x * H_prev1 - 2.0 * k * H_prev2
            H_prev2, H_prev1 = H_prev1, H_curr
        return H_curr


class PolynomialChaosExpansion:

    def __init__(self, max_order: int, num_quad_points: Optional[int] = None):
        self.P = max_order
        self.Q = num_quad_points if num_quad_points is not None else max_order + 2
        if self.Q < (self.P + 1):
            self.Q = self.P + 2
        self.nodes, self.weights = HermiteQuadrature.probabilist_nodes_weights(self.Q)

    def compute_coefficients(self, f: Callable) -> np.ndarray:
        coeffs = np.zeros(self.P + 1)
        f_vals = np.array([f(xi) for xi in self.nodes])
        for k in range(self.P + 1):
            He_k = HermitePolynomials.probabilist_hermite(k, self.nodes)
            factorial_k = np.math.factorial(k)
            coeffs[k] = np.sum(self.weights * f_vals * He_k) / factorial_k
        return coeffs

    def evaluate(self, coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi)
        result = np.zeros_like(xi)
        for k, c in enumerate(coeffs):
            result += c * HermitePolynomials.probabilist_hermite(k, xi)
        return result

    def mean(self, coeffs: np.ndarray) -> float:
        return coeffs[0]

    def variance(self, coeffs: np.ndarray) -> float:
        var = 0.0
        for k in range(1, len(coeffs)):
            var += coeffs[k] ** 2 * np.math.factorial(k)
        return var

    def standard_deviation(self, coeffs: np.ndarray) -> float:
        return np.sqrt(self.variance(coeffs))

    def skewness(self, coeffs: np.ndarray) -> float:
        sigma = self.standard_deviation(coeffs)
        if sigma < 1e-30:
            return 0.0
        mu3 = 0.0

        for k in range(len(coeffs)):
            for m in range(len(coeffs)):
                for n in range(len(coeffs)):
                    if (k + m + n) % 2 == 0 and self._triangle_inequality(k, m, n):
                        E_triple = self._expectation_triple_product(k, m, n)
                        mu3 += coeffs[k] * coeffs[m] * coeffs[n] * E_triple
        return mu3 / (sigma ** 3 + 1e-30)

    @staticmethod
    def _triangle_inequality(a: int, b: int, c: int) -> bool:
        return abs(a - b) <= c <= a + b

    @staticmethod
    def _expectation_triple_product(k: int, m: int, n: int) -> float:
        total = k + m + n
        if total % 2 != 0:
            return 0.0
        s = total // 2
        if s < max(k, m, n):
            return 0.0

        log_num = (np.math.lgamma(k + 1) + np.math.lgamma(m + 1) + np.math.lgamma(n + 1))
        log_den = (np.math.lgamma(s + 1) + np.math.lgamma(s - k + 1)
                   + np.math.lgamma(s - m + 1) + np.math.lgamma(s - n + 1))
        return np.exp(log_num - log_den)


class CompositeRandomProperties:

    @staticmethod
    def fiber_volume_fraction_to_modulus(Vf: float,
                                          Ef: float = 230e9,
                                          Em: float = 3.5e9) -> float:
        Vf_clip = np.clip(Vf, 0.0, 1.0)
        return Vf_clip * Ef + (1.0 - Vf_clip) * Em

    @staticmethod
    def random_Vf_pce_analysis(mu_Vf: float = 0.60, sigma_Vf: float = 0.03,
                                max_order: int = 4) -> dict:
        Ef = 230e9
        Em = 3.5e9
        pce = PolynomialChaosExpansion(max_order=max_order)

        def E1_func(xi):
            Vf = mu_Vf + sigma_Vf * xi
            return CompositeRandomProperties.fiber_volume_fraction_to_modulus(Vf, Ef, Em)

        coeffs = pce.compute_coefficients(E1_func)
        mean_E1 = pce.mean(coeffs)
        std_E1 = pce.standard_deviation(coeffs)
        skew_E1 = pce.skewness(coeffs)

        return {
            "E1_mean": mean_E1,
            "E1_std": std_E1,
            "E1_cov": std_E1 / (mean_E1 + 1e-30),
            "E1_skewness": skew_E1,
            "pce_coefficients": coeffs,
        }

    @staticmethod
    def lognormal_property_pce_analysis(mu_ln: float, sigma_ln: float,
                                        max_order: int = 5) -> dict:
        pce = PolynomialChaosExpansion(max_order=max_order)

        def X_func(xi):
            return np.exp(mu_ln + sigma_ln * xi)

        coeffs = pce.compute_coefficients(X_func)
        mean_X = pce.mean(coeffs)
        std_X = pce.standard_deviation(coeffs)


        exact_mean = np.exp(mu_ln + 0.5 * sigma_ln ** 2)
        exact_var = (np.exp(sigma_ln ** 2) - 1.0) * np.exp(2 * mu_ln + sigma_ln ** 2)

        return {
            "mean_pce": mean_X,
            "mean_exact": exact_mean,
            "std_pce": std_X,
            "std_exact": np.sqrt(exact_var),
            "relative_error_mean": abs(mean_X - exact_mean) / (exact_mean + 1e-30),
        }


if __name__ == "__main__":

    assert HermiteQuadrature.exactness_test(n=5, max_degree=9)
    print("Hermite exactness test PASSED.")


    pce = PolynomialChaosExpansion(max_order=3)
    coeffs = pce.compute_coefficients(lambda xi: 2.0 + 3.0 * xi)
    print("PCE coeffs for linear func:", coeffs)
    assert np.isclose(coeffs[0], 2.0, atol=1e-12)
    assert np.isclose(coeffs[1], 3.0, atol=1e-12)
    assert np.isclose(pce.variance(coeffs), 9.0, atol=1e-12)


    result = CompositeRandomProperties.random_Vf_pce_analysis()
    print("Composite UQ result:", result)
