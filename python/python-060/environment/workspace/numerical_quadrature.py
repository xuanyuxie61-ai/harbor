
import numpy as np
from typing import Tuple, Optional, Callable


class SquareQuadrature:

    def __init__(self):
        pass

    def square01_area(self) -> float:
        return 1.0

    def monomial_integral(self, e: Tuple[int, int]) -> float:
        e1, e2 = e
        if e1 < 0 or e2 < 0:
            raise ValueError("指数必须非负")
        return 1.0 / ((e1 + 1) * (e2 + 1))

    def sample_uniform(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.rand(n)
        y = np.random.rand(n)
        return x, y

    def monte_carlo_integrate(self, f: Callable,
                               n_samples: int = 10000) -> Tuple[float, float]:
        x, y = self.sample_uniform(n_samples)
        values = f(x, y)
        mean = np.mean(values)
        std = np.std(values)

        error = std / np.sqrt(n_samples)
        return mean, error

    def gauss_legendre_2d(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_1d, w_1d = np.polynomial.legendre.leggauss(n)

        x_1d = 0.5 * (x_1d + 1.0)
        w_1d = 0.5 * w_1d

        x = np.zeros(n * n)
        y = np.zeros(n * n)
        w = np.zeros(n * n)

        idx = 0
        for i in range(n):
            for j in range(n):
                x[idx] = x_1d[i]
                y[idx] = x_1d[j]
                w[idx] = w_1d[i] * w_1d[j]
                idx += 1

        return x, y, w

    def integrate_gauss(self, f: Callable, n: int = 5) -> float:
        x, y, w = self.gauss_legendre_2d(n)
        values = f(x, y)
        return np.sum(w * values)


class HexagonQuadrature:

    def __init__(self):
        pass

    def hexagon_area(self, side_length: float = 1.0) -> float:
        if side_length <= 0:
            raise ValueError("边长必须为正")
        return 3.0 * np.sqrt(3.0) / 2.0 * side_length ** 2

    def lyness_rule(self, rule_id: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rule_id == 1:

            xi = np.array([0.0])
            eta = np.array([0.0])
            w = np.array([1.0])
        elif rule_id == 2:

            xi = np.array([0.0, 1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
            eta = np.array([0.0, 0.0, np.sqrt(3)/2, np.sqrt(3)/2,
                            0.0, -np.sqrt(3)/2, -np.sqrt(3)/2])
            w = np.array([0.5, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])
        elif rule_id == 3:

            s3 = np.sqrt(3.0) / 2.0
            xi = np.array([0.0, 0.75, 0.0, -0.75, -0.75, 0.0, 0.75])
            eta = np.array([0.0, 0.375, 0.75, 0.375, -0.375, -0.75, -0.375])
            w_center = 0.25
            w_edge = 0.125
            w = np.array([w_center, w_edge, w_edge, w_edge,
                          w_edge, w_edge, w_edge])
        elif rule_id == 4:


            r1 = 0.5
            r2 = 0.9
            angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 13, endpoint=False)

            xi = [0.0]
            eta = [0.0]
            w = [0.2]

            for a in angles1[:-1]:
                xi.append(r1 * np.cos(a))
                eta.append(r1 * np.sin(a))
                w.append(0.1 / 6.0)

            for a in angles2[:-1]:
                xi.append(r2 * np.cos(a))
                eta.append(r2 * np.sin(a))
                w.append(0.1 / 12.0)

            xi = np.array(xi)
            eta = np.array(eta)
            w = np.array(w)

            w = w / np.sum(w)
        else:
            xi = np.array([0.0])
            eta = np.array([0.0])
            w = np.array([1.0])

        return xi, eta, w

    def integrate(self, f: Callable, side_length: float = 1.0,
                  rule_id: int = 3) -> float:
        xi, eta, w = self.lyness_rule(rule_id)
        area = self.hexagon_area(side_length)


        x = xi * side_length
        y = eta * side_length

        values = f(x, y)
        return area * np.sum(w * values)


class VerticalIntegrator:

    def __init__(self, z: np.ndarray):
        if len(z) < 2:
            raise ValueError("至少需要两个网格点")
        if not np.all(np.diff(z) > 0):
            raise ValueError("高度网格必须严格单调递增")
        self.z = z.copy()
        self.nz = len(z)

    def trapezoid(self, f: np.ndarray) -> float:
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        return np.trapezoid(f, self.z)

    def simpson(self, f: np.ndarray) -> float:
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        if self.nz % 2 == 0:

            return np.trapezoid(f, self.z)

        dz = self.z[1] - self.z[0]
        if not np.allclose(np.diff(self.z), dz, rtol=1e-3):

            return np.trapezoid(f, self.z)

        result = f[0] + f[-1]
        result += 4.0 * np.sum(f[1:-1:2])
        result += 2.0 * np.sum(f[2:-1:2])
        return result * dz / 3.0

    def integrate_product(self, f1: np.ndarray, f2: np.ndarray) -> float:
        if len(f1) != self.nz or len(f2) != self.nz:
            raise ValueError("数组长度与网格不匹配")
        return np.trapezoid(f1 * f2, self.z)

    def cumulative_integral(self, f: np.ndarray) -> np.ndarray:
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        F = np.zeros(self.nz)
        for i in range(1, self.nz):
            F[i] = F[i - 1] + 0.5 * (f[i] + f[i - 1]) * (self.z[i] - self.z[i - 1])
        return F

    def optical_depth_integral(self, sigma: np.ndarray,
                                n: np.ndarray) -> np.ndarray:
        if len(sigma) != self.nz or len(n) != self.nz:
            raise ValueError("数组长度与网格不匹配")

        tau = np.zeros(self.nz)
        integrand = sigma * n
        for i in range(self.nz - 2, -1, -1):
            tau[i] = tau[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * \
                     (self.z[i + 1] - self.z[i])
        return tau


class AtmosphericColumnIntegrator:

    def __init__(self, z: np.ndarray,
                 horizontal_area: float = 1.0e10):
        self.vertical = VerticalIntegrator(z)
        self.horizontal_area = horizontal_area
        self.square_quad = SquareQuadrature()

    def column_density(self, n_z: np.ndarray) -> float:
        return self.vertical.trapezoid(n_z)

    def total_moles(self, n_z: np.ndarray) -> float:
        N_A = 6.022e23
        column = self.column_density(n_z)
        return column * self.horizontal_area / N_A

    def dobson_unit(self, n_z: np.ndarray) -> float:
        column = self.column_density(n_z)

        du = column / 2.69e20
        return du

    def horizontal_average(self, field_3d: np.ndarray) -> np.ndarray:
        if field_3d.ndim != 3:
            raise ValueError("field_3d 必须为三维数组")
        return np.mean(np.mean(field_3d, axis=0), axis=0)
