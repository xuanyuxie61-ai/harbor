# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lu_factor, lu_solve


def legendre_polynomial(n: int, xi: np.ndarray) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative.")
    xi = np.asarray(xi)
    if n == 0:
        return np.ones_like(xi)
    if n == 1:
        return xi.copy()

    P_prev2 = np.ones_like(xi)
    P_prev1 = xi.copy()
    P_curr = np.zeros_like(xi)
    for k in range(1, n):

        P_curr = ((2.0 * k + 1.0) * xi * P_prev1 - k * P_prev2) / (k + 1.0)
        P_prev2, P_prev1 = P_prev1, P_curr
    return P_curr


def legendre_polynomial_derivative(n: int, xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi)
    if n == 0:
        return np.zeros_like(xi)
    if n == 1:
        return np.ones_like(xi)

    Pn = legendre_polynomial(n, xi)
    Pn_1 = legendre_polynomial(n - 1, xi)

    eps = 1e-14
    dP = np.zeros_like(xi)
    mask = np.abs(np.abs(xi) - 1.0) > eps

    dP[mask] = n * (Pn_1[mask] - xi[mask] * Pn[mask]) / (1.0 - xi[mask] ** 2)

    boundary_mask = ~mask
    sign = np.where(xi[boundary_mask] > 0, 1.0, -1.0)
    dP[boundary_mask] = sign ** (n + 1) * n * (n + 1.0) / 2.0
    return dP


def jacobi_gauss_lobatto_points(N: int) -> np.ndarray:
    if N < 1:
        raise ValueError("N must be >= 1.")
    if N == 1:
        return np.array([-1.0, 1.0])

    n_roots = N - 1

    x0 = -np.cos(np.pi * np.arange(1, n_roots + 1) / N)
    x = x0.copy()

    tol = 1e-14
    max_iter = 100
    for _ in range(max_iter):
        P = legendre_polynomial(N - 1, x)
        dP = legendre_polynomial_derivative(N - 1, x)



        mask = np.abs(1.0 - x ** 2) > 1e-14
        d2P = np.zeros_like(x)
        d2P[mask] = (2.0 * x[mask] * dP[mask] - N * (N - 1.0) * P[mask]) / (1.0 - x[mask] ** 2)

        d2P[~mask] = 1e6

        dx = dP / (d2P + 1e-30)
        x_new = x - dx
        if np.max(np.abs(dx)) < tol:
            break
        x = x_new

    nodes = np.empty(N + 1)
    nodes[0] = -1.0
    nodes[1:N] = np.sort(x)
    nodes[N] = 1.0
    return nodes


def jacobi_gauss_lobatto_weights(nodes: np.ndarray) -> np.ndarray:
    N = len(nodes) - 1
    if N < 1:
        raise ValueError("At least 2 nodes required.")
    PN = legendre_polynomial(N, nodes)
    weights = 2.0 / (N * (N + 1.0) * (PN ** 2))

    weights = weights / np.sum(weights) * 2.0
    return weights


def vandermonde_matrix_1d(N: int, nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes)
    V = np.zeros((len(nodes), N + 1))
    for j in range(N + 1):
        V[:, j] = legendre_polynomial(j, nodes)
    return V


def differentiation_matrix_1d(N: int, nodes: np.ndarray) -> np.ndarray:
    V = vandermonde_matrix_1d(N, nodes)
    Vr = np.zeros_like(V)
    for j in range(N + 1):
        Vr[:, j] = legendre_polynomial_derivative(j, nodes)

    V_inv = np.linalg.inv(V)
    D = Vr @ V_inv
    return D


def quadrature_weights_arbitrary_nodes(nodes: np.ndarray, a: float = -1.0, b: float = 1.0,
                                        max_degree: Optional[int] = None) -> np.ndarray:
    nodes = np.asarray(nodes)
    N = len(nodes)
    if max_degree is None:
        max_degree = N - 1
    if max_degree < 0 or max_degree >= N:
        raise ValueError("max_degree must be in [0, N-1].")


    rhs = np.zeros(N)
    for k in range(N):
        rhs[k] = (b ** (k + 1.0) - a ** (k + 1.0)) / (k + 1.0)


    V = np.vander(nodes, N=N, increasing=True)


    try:
        lu, piv = lu_factor(V)
        weights = lu_solve((lu, piv), rhs)
    except Exception:

        weights, *_ = np.linalg.lstsq(V, rhs, rcond=None)


    weights = np.where(np.abs(weights) < 1e-15, 0.0, weights)
    return weights


def vandermonde_matrix_2d_total_degree(degree: int, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2).")

    dim = (degree + 1) * (degree + 2) // 2
    V = np.ones((points.shape[0], dim))
    col = 1

    for m in range(1, degree + 1):
        for i in range(m + 1):
            j = m - i
            V[:, col] = (points[:, 0] ** i) * (points[:, 1] ** j)
            col += 1
    return V


def solve_2d_interpolation_coefficients(degree: int, points: np.ndarray, values: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    values = np.asarray(values)
    V = vandermonde_matrix_2d_total_degree(degree, points)
    dim = V.shape[1]
    if len(values) < dim:
        raise ValueError(f"Need at least {dim} points for degree {degree} interpolation.")
    if len(values) == dim:
        coeffs = np.linalg.solve(V, values)
    else:
        coeffs, *_ = np.linalg.lstsq(V, values, rcond=None)
    return coeffs


def evaluate_2d_polynomial(degree: int, coeffs: np.ndarray, points: np.ndarray) -> np.ndarray:
    V = vandermonde_matrix_2d_total_degree(degree, points)
    return V @ coeffs


if __name__ == "__main__":

    N = 5
    nodes = jacobi_gauss_lobatto_points(N)
    weights = jacobi_gauss_lobatto_weights(nodes)
    print("GLL nodes:", nodes)
    print("GLL weights:", weights)
    print("Sum of weights:", np.sum(weights))


    for p in range(2 * N - 1):
        exact = (1.0 ** (p + 1) - (-1.0) ** (p + 1)) / (p + 1.0)
        approx = np.sum(weights * (nodes ** p))
        if p <= 2 * N - 2:
            assert np.isclose(approx, exact, atol=1e-12), f"Failed for degree {p}"
    print("GLL quadrature exactness test PASSED.")


    arb_nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    arb_weights = quadrature_weights_arbitrary_nodes(arb_nodes)
    print("Arbitrary nodes weights:", arb_weights)

    approx4 = np.sum(arb_weights * (arb_nodes ** 4))
    exact4 = 2.0 / 5.0
    print(f"x^4 integral: exact={exact4:.6f}, approx={approx4:.6f}")
