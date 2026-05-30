
import numpy as np
from typing import Callable, Tuple, List
from scipy.special import gamma as Gamma_func


def en_r2_monomial_integral(exponents: Tuple[int, ...]) -> float:
    result = 1.0
    for alpha in exponents:
        if alpha < 0:
            raise ValueError("Exponents must be non-negative")
        if alpha % 2 == 1:
            return 0.0
        result *= Gamma_func((alpha + 1) / 2.0)
    return result


def cn_leg_monomial_integral(exponents: Tuple[int, ...]) -> float:
    result = 1.0
    for alpha in exponents:
        if alpha < 0:
            raise ValueError("Exponents must be non-negative")
        result *= (1.0 - (-1.0) ** (alpha + 1)) / (alpha + 1.0)
    return result


def stroud_cn_leg_03_1(n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_dim <= 0:
        raise ValueError("Dimension must be positive")

    n_points = 2 * n_dim
    nodes = np.zeros((n_points, n_dim))
    weights = np.zeros(n_points)

    r = np.sqrt(2.0 / 3.0)
    volume = 2.0 ** n_dim
    w = volume / n_points

    for i in range(n_dim):
        nodes[2 * i, i] = r
        nodes[2 * i + 1, i] = -r
        weights[2 * i] = w
        weights[2 * i + 1] = w

    return nodes, weights


def stroud_en_r2_03_1(n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_dim <= 0:
        raise ValueError("Dimension must be positive")

    n_points = 2 * n_dim
    nodes = np.zeros((n_points, n_dim))
    weights = np.zeros(n_points)

    r = np.sqrt((n_dim + 2.0) / 2.0)
    volume = np.pi ** (n_dim / 2.0)
    w = volume / n_points

    for i in range(n_dim):
        nodes[2 * i, i] = r
        nodes[2 * i + 1, i] = -r
        weights[2 * i] = w
        weights[2 * i + 1] = w

    return nodes, weights


def stroud_en_r2_05_1(n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_dim <= 0:
        raise ValueError("Dimension must be positive")


    if n_dim > 6:
        return stroud_en_r2_03_1(n_dim)

    n_points = (2 ** n_dim) + 2 * n_dim
    nodes = np.zeros((n_points, n_dim))
    weights = np.zeros(n_points)

    volume = np.pi ** (n_dim / 2.0)


    s = np.sqrt((n_dim + 2.0) / 4.0)
    idx = 0
    for i in range(2 ** n_dim):
        sign_pattern = [(i >> j) & 1 for j in range(n_dim)]
        for j in range(n_dim):
            nodes[idx, j] = s if sign_pattern[j] == 0 else -s
        idx += 1


    r = np.sqrt((n_dim + 2.0) / 2.0)
    for i in range(n_dim):
        nodes[idx, i] = r
        idx += 1
        nodes[idx, i] = -r
        idx += 1


    w_diag = volume * (4.0 - n_dim) / (2.0 ** (n_dim + 2) * (n_dim + 2.0))
    w_axis = volume * n_dim / (2.0 * n_dim * (n_dim + 2.0))

    for i in range(2 ** n_dim):
        weights[i] = w_diag
    for i in range(2 ** n_dim, n_points):
        weights[i] = w_axis

    return nodes, weights


class StroudIntegrator:

    def __init__(self, n_dim: int, rule_type: str = "en_r2_03"):
        if n_dim <= 0:
            raise ValueError("Dimension must be positive")
        self.n_dim = n_dim
        self.rule_type = rule_type

        if rule_type == "en_r2_03":
            self.nodes, self.weights = stroud_en_r2_03_1(n_dim)
        elif rule_type == "en_r2_05":
            self.nodes, self.weights = stroud_en_r2_05_1(n_dim)
        elif rule_type == "cn_leg_03":
            self.nodes, self.weights = stroud_cn_leg_03_1(n_dim)
        else:
            raise ValueError(f"Unknown rule_type: {rule_type}")

    def integrate(self, f: Callable[[np.ndarray], float]) -> float:
        if len(self.nodes) != len(self.weights):
            raise ValueError("Nodes and weights must have same length")

        result = 0.0
        for i in range(len(self.nodes)):
            result += self.weights[i] * f(self.nodes[i])
        return result

    def integrate_vectorized(self, f: Callable[[np.ndarray], np.ndarray]) -> float:
        values = f(self.nodes)
        return np.dot(self.weights, values)


def gaussian_quadrature_kernel_expectation(
    kernel_func: Callable[[np.ndarray, np.ndarray], float],
    x_point: np.ndarray,
    n_dim: int,
    rule_type: str = "en_r2_03"
) -> float:
    if len(x_point) != n_dim:
        raise ValueError("x_point dimension must match n_dim")

    integrator = StroudIntegrator(n_dim, rule_type)

    def integrand(y: np.ndarray) -> float:
        return kernel_func(x_point, y)

    return integrator.integrate(integrand)
