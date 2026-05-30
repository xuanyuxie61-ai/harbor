# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Tuple, Optional


class StellarIntegrator:

    @staticmethod
    def newton_cotes_weights(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if n < 2 or n > 7:
            raise ValueError("当前仅支持 2~7 点 Newton-Cotes 规则")
        x = np.linspace(a, b, n)
        h = (b - a) / (n - 1)

        coeff_map = {
            2: np.array([1.0, 1.0]) / 2.0,
            3: np.array([1.0, 4.0, 1.0]) / 3.0,
            4: np.array([3.0, 9.0, 9.0, 3.0]) / 8.0,
            5: np.array([14.0, 64.0, 24.0, 64.0, 14.0]) / 45.0,
            6: np.array([95.0, 375.0, 250.0, 250.0, 375.0, 95.0]) / 288.0,
            7: np.array([41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0]) / 140.0,
        }
        w = coeff_map[n] * h * (n - 1)
        return x, w

    @staticmethod
    def integrate_ncc(f: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                      n: int = 5) -> float:
        x, w = StellarIntegrator.newton_cotes_weights(n, a, b)
        fx = f(x)
        return float(np.dot(w, fx))

    @staticmethod
    def triangle_gauss_rule(degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:

        if degree <= 1:

            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0]])
            weights = np.array([0.5])
        elif degree <= 2:

            nodes = np.array([[2.0 / 3.0, 1.0 / 6.0],
                              [1.0 / 6.0, 2.0 / 3.0],
                              [1.0 / 6.0, 1.0 / 6.0]])
            weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        elif degree <= 3:

            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0],
                              [0.6, 0.2],
                              [0.2, 0.6],
                              [0.2, 0.2]])
            weights = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        elif degree <= 4:

            a1 = 0.108103018168070
            b1 = 0.445948490915965
            a2 = 0.816847572980459
            b2 = 0.091576213509771
            nodes = np.array([[a1, b1], [b1, a1], [b1, b1],
                              [a2, b2], [b2, a2], [b2, b2]])
            w1 = 0.223381589678011
            w2 = 0.109951743655322
            weights = np.array([w1, w1, w1, w2, w2, w2])
        elif degree <= 5:

            a1 = 0.059715871789770
            b1 = 0.470142064105115
            a2 = 0.797426985353087
            b2 = 0.101286507323456
            nodes = np.array([[1.0 / 3.0, 1.0 / 3.0],
                              [a1, b1], [b1, a1], [b1, b1],
                              [a2, b2], [b2, a2], [b2, b2]])
            w1 = 0.225000000000000
            w2 = 0.132394152788506
            w3 = 0.125939180544827
            weights = np.array([w1, w2, w2, w2, w3, w3, w3])
        elif degree <= 6:

            a1 = 0.501426509658179
            b1 = 0.249286745170910
            a2 = 0.873821971016996
            b2 = 0.063089014491502
            a3 = 0.053145049844817
            b3 = 0.310352451033784
            c3 = 1.0 - a3 - b3
            nodes = np.array([
                [a1, b1], [b1, a1], [b1, b1],
                [a2, b2], [b2, a2], [b2, b2],
                [a3, b3], [b3, a3], [c3, a3],
                [a3, c3], [b3, c3], [c3, b3]
            ])
            w1 = 0.116786275726379
            w2 = 0.050844906370207
            w3 = 0.082851075618374
            weights = np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])
        else:

            a1 = 0.479308067841920
            b1 = 0.260345966079040
            a2 = 0.869739794195568
            b2 = 0.065130102902216
            a3 = 0.048690315425316
            b3 = 0.312865496004874
            c3 = 1.0 - a3 - b3
            nodes = np.array([
                [1.0 / 3.0, 1.0 / 3.0],
                [a1, b1], [b1, a1], [b1, b1],
                [a2, b2], [b2, a2], [b2, b2],
                [a3, b3], [b3, a3], [c3, a3],
                [a3, c3], [b3, c3], [c3, b3]
            ])
            w1 = -0.149570044467671
            w2 = 0.175615257433204
            w3 = 0.053347235608838
            w4 = 0.077113760890257
            weights = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w4, w4, w4])
        return nodes, weights * 0.5

    @staticmethod
    def integrate_triangle(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           v1: Tuple[float, float],
                           v2: Tuple[float, float],
                           v3: Tuple[float, float],
                           degree: int = 5) -> float:
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        v3 = np.array(v3, dtype=np.float64)

        area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
        nodes_ref, weights_ref = StellarIntegrator.triangle_gauss_rule(degree)

        x_phys = v1[0] + (v2[0] - v1[0]) * nodes_ref[:, 0] + (v3[0] - v1[0]) * nodes_ref[:, 1]
        y_phys = v1[1] + (v2[1] - v1[1]) * nodes_ref[:, 0] + (v3[1] - v1[1]) * nodes_ref[:, 1]
        fx = f(x_phys, y_phys)
        return float(np.sum(weights_ref * fx) * 2.0 * area)

    @staticmethod
    def integrate_shell(f: np.ndarray, mass: np.ndarray, method: str = 'simpson') -> float:
        f = np.asarray(f, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        if len(f) != len(mass):
            raise ValueError("f 与 mass 长度必须相同")
        n = len(mass)
        if n < 2:
            return 0.0
        if method == 'trapezoidal':
            return np.trapz(f, mass)
        elif method == 'simpson':
            if n < 3:
                return np.trapz(f, mass)


            dm = np.diff(mass)
            if np.allclose(dm, dm[0], rtol=0.1):
                h = dm[0]
                result = f[0] + f[-1]
                result += 4.0 * np.sum(f[1:-1:2])
                result += 2.0 * np.sum(f[2:-1:2])
                return result * h / 3.0
            else:
                return np.trapz(f, mass)
        else:
            return np.trapz(f, mass)

    @staticmethod
    def moment_of_inertia(radius: np.ndarray, density: np.ndarray, dm: np.ndarray) -> float:
        r = np.asarray(radius, dtype=np.float64)
        dm_arr = np.asarray(dm, dtype=np.float64)
        integrand = (2.0 / 3.0) * r ** 2
        return float(np.trapz(integrand, np.cumsum(dm_arr)))

    @staticmethod
    def gravitational_binding_energy(mass: np.ndarray, radius: np.ndarray) -> float:
        G = 6.67430e-8
        m = np.asarray(mass, dtype=np.float64)
        r = np.asarray(radius, dtype=np.float64)
        r = np.maximum(r, 1e-3)
        integrand = -G * m / r
        return float(np.trapz(integrand, m))
