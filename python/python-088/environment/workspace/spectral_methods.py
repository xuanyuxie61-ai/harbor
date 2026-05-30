
import numpy as np
from typing import Tuple, Optional


def shifted_legendre_polynomial(
    x: np.ndarray, n_max: int
) -> np.ndarray:
    x = np.asarray(x)
    m = x.shape[0]

    if n_max < 0:
        return np.zeros((m, 0))

    v = np.zeros((m, n_max + 1))
    v[:, 0] = 1.0

    if n_max < 1:
        return v

    v[:, 1] = 2.0 * x - 1.0

    for i in range(2, n_max + 1):
        v[:, i] = (
            (2 * i - 1) * (2.0 * x - 1.0) * v[:, i - 1]
            - (i - 1) * v[:, i - 2]
        ) / i

    return v


def shifted_legendre_derivative(
    x: np.ndarray, n_max: int
) -> np.ndarray:
    x = np.asarray(x)
    m = x.shape[0]

    if n_max < 0:
        return np.zeros((m, 0))


    t = 2.0 * x - 1.0
    v = np.zeros((m, n_max + 1))
    dv = np.zeros((m, n_max + 1))

    v[:, 0] = 1.0
    dv[:, 0] = 0.0

    if n_max >= 1:
        v[:, 1] = t
        dv[:, 1] = 1.0

    for i in range(2, n_max + 1):
        v[:, i] = ((2 * i - 1) * t * v[:, i - 1] - (i - 1) * v[:, i - 2]) / i
        dv[:, i] = ((2 * i - 1) * (v[:, i - 1] + t * dv[:, i - 1]) - (i - 1) * dv[:, i - 2]) / i


    return 2.0 * dv


def gauss_legendre_nodes_weights(n: int, domain: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        return np.array([]), np.array([])



    nodes, weights = np.polynomial.legendre.leggauss(n)


    a, b = domain
    nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    weights = 0.5 * (b - a) * weights

    return nodes, weights


def spectral_projection(
    f, n_modes: int, n_quad: int = None
) -> np.ndarray:
    if n_quad is None:
        n_quad = 2 * n_modes

    nodes, weights = gauss_legendre_nodes_weights(n_quad, domain=(0.0, 1.0))
    poly_vals = shifted_legendre_polynomial(nodes, n_modes)

    f_vals = f(nodes)
    coeffs = np.zeros(n_modes + 1)
    for n in range(n_modes + 1):
        coeffs[n] = (2 * n + 1) * np.sum(weights * f_vals * poly_vals[:, n])

    return coeffs


def spectral_reconstruct(
    coeffs: np.ndarray, x: np.ndarray
) -> np.ndarray:
    n_max = len(coeffs) - 1
    poly_vals = shifted_legendre_polynomial(x, n_max)
    return poly_vals @ coeffs


def spectral_derivative_matrix(n: int, domain: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:



    x_cheb = np.cos(np.pi * np.arange(n + 1) / n)

    a, b = domain
    x_nodes = 0.5 * (b - a) * (x_cheb + 1) + a


    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:

                prod = 1.0
                for k in range(n + 1):
                    if k != i and k != j:
                        prod *= (x_nodes[i] - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
                D[i, j] = prod / (x_nodes[j] - x_nodes[i])
            else:

                s = 0.0
                for k in range(n + 1):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s

    return D


def integrate_with_legendre_expansion(
    coeffs: np.ndarray
) -> float:
    return float(coeffs[0]) if len(coeffs) > 0 else 0.0


def convolution_legendre_kernel(
    kernel_func, n_modes: int, t: float
) -> np.ndarray:
    n = n_modes + 1
    nodes, weights = gauss_legendre_nodes_weights(n, domain=(0.0, t))
    poly_vals = shifted_legendre_polynomial(nodes / t, n_modes)

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):

            s = nodes
            tau = t - s
            K_vals = kernel_func(tau)
            C[i, j] = np.sum(weights * K_vals * poly_vals[:, i] * poly_vals[:, j])

    return C
