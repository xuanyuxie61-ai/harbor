# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List, Callable


class GegenbauerQuadrature:

    def __init__(self, order: int, alpha: float, a: float = -1.0, b: float = 1.0):
        if order < 1:
            raise ValueError("order 必须为正整数")
        if alpha <= -1.0:
            raise ValueError("alpha 必须大于 -1")

        self.order = order
        self.alpha = alpha
        self.a = a
        self.b = b
        self.nodes, self.weights = self._compute_rule()

    def _compute_rule(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.order
        alpha = self.alpha



        diag = np.zeros(n)
        offdiag = np.zeros(n - 1)

        for i in range(1, n):
            num = i * (i + 2.0 * alpha)
            den = (2.0 * i + 2.0 * alpha - 1.0) * (2.0 * i + 2.0 * alpha + 1.0)
            if den > 0 and num > 0:
                offdiag[i - 1] = np.sqrt(num / den)



        J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
        eigenvalues, eigenvectors = np.linalg.eigh(J)


        nodes = eigenvalues




        from math import gamma
        mu0 = (2.0 ** (2.0 * alpha + 1.0)) * (gamma(alpha + 1.0) ** 2) / gamma(2.0 * alpha + 2.0)

        weights = mu0 * (eigenvectors[0, :] ** 2)


        mid = (self.a + self.b) / 2.0
        scale = (self.b - self.a) / 2.0
        nodes_scaled = mid + scale * nodes
        weights_scaled = scale * weights

        return nodes_scaled, weights_scaled

    def integrate(self, f: Callable[[np.ndarray], np.ndarray]) -> float:
        fx = f(self.nodes)
        return float(np.dot(self.weights, fx))

    def integrate_parabolic_profile(self, dp_dx: float, b: float,
                                     mu: float = 1.0e-3) -> float:
        if b <= 0 or mu <= 0:
            raise ValueError("b 和 mu 必须为正")

        def velocity(z):
            zeta = 2.0 * z / b - 1.0
            return -(b ** 2 / (8.0 * mu)) * dp_dx * (1.0 - zeta ** 2)

        return self.integrate(velocity)


class FlowIntegrator:

    @staticmethod
    def breakthrough_curve_moments(times: np.ndarray,
                                    concentrations: np.ndarray) -> dict:
        if len(times) != len(concentrations):
            raise ValueError("times 和 concentrations 长度必须相同")
        if len(times) < 2:
            raise ValueError("至少需要 2 个数据点")


        dt = np.diff(times)
        C_mid = 0.5 * (concentrations[:-1] + concentrations[1:])

        M0 = np.sum(C_mid * dt)
        if M0 < 1e-20:
            return {'M0': 0.0, 't_mean': 0.0, 'variance': 0.0, 'skewness': 0.0}

        t_mid = 0.5 * (times[:-1] + times[1:])
        M1 = np.sum(t_mid * C_mid * dt)
        M2 = np.sum(t_mid ** 2 * C_mid * dt)
        M3 = np.sum(t_mid ** 3 * C_mid * dt)

        t_mean = M1 / M0
        variance = M2 / M0 - t_mean ** 2
        std = np.sqrt(max(variance, 0.0))

        skewness = 0.0
        if std > 1e-12:
            skewness = (M3 / M0 - 3.0 * t_mean * variance - t_mean ** 3) / (std ** 3)

        return {
            'M0': float(M0),
            't_mean': float(t_mean),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(skewness)
        }

    @staticmethod
    def dispersivity_from_moments(t_mean: float, variance: float,
                                   L: float, v: float) -> float:
        if t_mean <= 0 or L <= 0 or v <= 0:
            raise ValueError("参数必须为正")

        alpha_L = variance * v ** 3 / (2.0 * L)
        return max(alpha_L, 0.0)

    @staticmethod
    def recovery_ratio(C_out: np.ndarray, Q: np.ndarray,
                         dt: float, M_injected: float) -> float:
        if M_injected <= 0:
            raise ValueError("M_injected 必须为正")

        recovered = np.sum(Q * C_out) * dt
        return float(recovered / M_injected)

    @staticmethod
    def peclet_number(v: float, L: float, D: float) -> float:
        if D <= 0:
            raise ValueError("D 必须为正")
        return v * L / D

    @staticmethod
    def reynolds_number(v: float, b: float, rho: float = 1000.0,
                        mu: float = 1.0e-3) -> float:
        if mu <= 0 or b <= 0:
            raise ValueError("mu 和 b 必须为正")
        return rho * v * b / mu
