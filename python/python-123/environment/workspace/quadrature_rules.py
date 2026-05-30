
import numpy as np
from typing import Tuple, Callable


def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("clenshaw_curtis_nodes_weights: n >= 1")

    if n == 1:
        return np.array([0.0]), np.array([2.0])

    theta = np.linspace(0.0, np.pi, n)
    x = np.cos(theta)


    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0
        for j in range(1, (n - 1) // 2 + 1):
            b = 1.0 if 2 * j == n - 1 else 2.0
            w[i] -= b * np.cos(2.0 * j * theta[i]) / (4.0 * j * j - 1.0)

    w[0] = w[0] / (n - 1)
    w[1:n - 1] = 2.0 * w[1:n - 1] / (n - 1)
    w[n - 1] = w[n - 1] / (n - 1)

    return x, w


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("gauss_legendre_nodes_weights: n >= 1")


    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_1d(f: Callable[[np.ndarray], np.ndarray],
                 a: float, b: float,
                 rule: str = "gauss", n: int = 16) -> float:
    if b <= a:
        raise ValueError("integrate_1d: 需要 b > a")

    if rule == "gauss":
        t, w = gauss_legendre_nodes_weights(n)
    elif rule == "clenshaw_curtis":
        t, w = clenshaw_curtis_nodes_weights(n)
    else:
        raise ValueError("integrate_1d: rule 必须是 'gauss' 或 'clenshaw_curtis'")


    x = 0.5 * ((b - a) * t + (a + b))
    fx = f(x)
    fx = np.asarray(fx, dtype=float).ravel()

    if fx.shape[0] != n:
        raise ValueError("integrate_1d: f 输出维度与求积阶数不匹配")

    return float(np.sum(w * fx) * (b - a) / 2.0)


def compute_therapy_response_index(
    drug_concentration: np.ndarray,
    cell_density: np.ndarray,
    stress_field: np.ndarray,
    dx: float, dy: float,
    stress_sensitivity: float = 2.0
) -> float:
    if drug_concentration.shape != cell_density.shape or drug_concentration.shape != stress_field.shape:
        raise ValueError("compute_therapy_response_index: 输入场维度不匹配")


    penalty = np.exp(-stress_sensitivity * np.maximum(stress_field, 0.0))
    integrand = drug_concentration * cell_density * penalty
    tri = float(np.sum(integrand) * dx * dy)
    return tri


def compute_cumulative_oxygen_consumption(
    oxygen_field: np.ndarray,
    cell_density: np.ndarray,
    dx: float, dy: float,
    Vmax: float = 1.0, Km: float = 0.1
) -> float:


    raise NotImplementedError("Hole_3: compute_cumulative_oxygen_consumption 待实现")



def integrate_radial_profile(
    r_vals: np.ndarray, f_vals: np.ndarray, dim: int = 2
) -> float:
    if r_vals.shape[0] < 2:
        return 0.0
    if f_vals.shape != r_vals.shape:
        raise ValueError("integrate_radial_profile: r_vals 与 f_vals 形状不匹配")

    if dim == 2:
        integrand = f_vals * 2.0 * np.pi * r_vals
    elif dim == 3:
        integrand = f_vals * 4.0 * np.pi * r_vals ** 2
    else:
        raise ValueError("integrate_radial_profile: dim 必须是 2 或 3")

    return float(np.trapezoid(integrand, r_vals))


def estimate_quadrature_error(
    f: Callable[[np.ndarray], np.ndarray],
    a: float, b: float,
    rule: str = "gauss",
    n_coarse: int = 8, n_fine: int = 32
) -> float:
    q_coarse = integrate_1d(f, a, b, rule, n_coarse)
    q_fine = integrate_1d(f, a, b, rule, n_fine)
    return abs(q_fine - q_coarse)
