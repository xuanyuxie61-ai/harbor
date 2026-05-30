
import numpy as np


def legendre_polynomial(n, x):
    x = np.atleast_1d(x)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()

    p0 = np.ones_like(x)
    p1 = x.copy()
    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2
    return p1


def legendre_polynomial_derivative(n, x):
    x = np.atleast_1d(x)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)


    dp0 = np.zeros_like(x)
    dp1 = np.ones_like(x)
    for k in range(1, n):
        dp2 = (2.0 * k + 1.0) * legendre_polynomial(k, x) + dp0
        dp0, dp1 = dp1, dp2
    return dp1


def lobatto_polynomial(n, x):
    x = np.atleast_1d(x)
    if n <= 0:
        return np.zeros_like(x)
    val = n * (legendre_polynomial(n - 1, x) - x * legendre_polynomial(n, x))
    return val


def lobatto_polynomial_derivative(n, x):
    x = np.atleast_1d(x)
    if n <= 0:
        return np.zeros_like(x)
    dp_nm1 = legendre_polynomial_derivative(n - 1, x)
    p_n = legendre_polynomial(n, x)
    dp_n = legendre_polynomial_derivative(n, x)
    return n * (dp_nm1 - p_n - x * dp_n)


def gll_nodes_weights(n):
    if n < 1:
        raise ValueError("gll_nodes_weights: 要求 n ≥ 1")


    k = np.arange(1, n)
    x_init = -np.cos(np.pi * k / n)

    x_inner = x_init.copy()
    for _ in range(100):
        f = lobatto_polynomial(n, x_inner)
        df = lobatto_polynomial_derivative(n, x_inner)
        dx = f / (df + 1.0e-30)
        x_inner -= dx
        if np.max(np.abs(dx)) < 1.0e-14:
            break

    nodes = np.concatenate(([-1.0], np.sort(x_inner), [1.0]))


    p_n_vals = legendre_polynomial(n, nodes)
    weights = 2.0 / (n * (n + 1.0) * p_n_vals ** 2)

    return nodes, weights


def lagrange_derivative_matrix(nodes):
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)


    w = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] *= (nodes[i] - nodes[j])
        w[i] = 1.0 / w[i]

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / (w[i] * (nodes[i] - nodes[j]))
    for i in range(n):
        D[i, i] = -np.sum(D[i, :])

    return D


def build_gll_time_operators(n, T=1.0):
    nodes_ref, weights_ref = gll_nodes_weights(n)


    nodes_t = 0.5 * T * (nodes_ref + 1.0)
    scale = 0.5 * T


    M_t = np.diag(weights_ref * scale)


    D_ref = lagrange_derivative_matrix(nodes_ref)

    D_t = D_ref / scale




    S_t = M_t @ D_t

    return nodes_t, M_t, S_t, D_t
