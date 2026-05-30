# -*- coding: utf-8 -*-

import numpy as np



_GLL_TABLES = {
    4: {
        'nodes': np.array([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0]),
        'weights': np.array([0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1])
    },
    6: {
        'nodes': np.array([-1.0, -0.830223896278567, -0.4688487934707142, 0.0,
                           0.4688487934707142, 0.830223896278567, 1.0]),
        'weights': np.array([0.047619047619047616, 0.2768260473615659, 0.4317453812098627,
                             0.4876190476190476, 0.4317453812098627, 0.2768260473615659,
                             0.047619047619047616])
    },
    8: {
        'nodes': np.array([-1.0, -0.8997579954114602, -0.6771862795107377, -0.36311746382617816,
                           0.0, 0.36311746382617816, 0.6771862795107377, 0.8997579954114602, 1.0]),
        'weights': np.array([0.027777777777777776, 0.1654953615608055, 0.2745387125006584,
                             0.3464533648026935, 0.3715192743764172, 0.3464533648026935,
                             0.2745387125006584, 0.1654953615608055, 0.027777777777777776])
    }
}


def legendre_polynomial(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x), np.zeros_like(x)
    if n == 1:
        return x, np.ones_like(x)

    p0 = np.ones_like(x)
    p1 = x.copy()

    for j in range(2, n + 1):
        p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
        p0 = p1
        p1 = p2


    denom = 1.0 - x ** 2
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    dp = n * (p0 - x * p1) / denom

    return p1, dp


def lobatto_polynomial_value(m, n, x):
    x = np.asarray(x, dtype=float).flatten()
    m = len(x)
    L = np.zeros((m, n), dtype=float)

    if n >= 1:
        L[:, 0] = 1.0 - x ** 2

        if n >= 2:
            P = np.zeros((m, n + 2), dtype=float)
            P[:, 0] = 1.0
            P[:, 1] = x

            for j in range(2, n + 2):
                P[:, j] = ((2 * j - 1) * x * P[:, j - 1] - (j - 1) * P[:, j - 2]) / j

            for j in range(2, n + 1):
                L[:, j - 1] = j * (P[:, j - 1] - x * P[:, j])

    return L


def gll_nodes_weights(n):
    if n in _GLL_TABLES:
        data = _GLL_TABLES[n]
        return data['nodes'].copy(), data['weights'].copy()


    data = _GLL_TABLES[6]
    return data['nodes'].copy(), data['weights'].copy()


def differentiation_matrix(nodes):
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)
    eps = 1e-15


    w = np.ones(n, dtype=float)
    for j in range(n):
        for k in range(n):
            if k != j:
                diff = nodes[j] - nodes[k]
                if abs(diff) > eps:
                    w[j] *= diff
                else:
                    w[j] *= eps
        if abs(w[j]) > eps:
            w[j] = 1.0 / w[j]
        else:
            w[j] = 0.0

    for i in range(n):
        for j in range(n):
            if i != j:
                diff = nodes[i] - nodes[j]
                denom = w[i] * diff
                if abs(denom) > eps:
                    D[i, j] = w[j] / denom
                else:
                    D[i, j] = 0.0
            else:
                D[i, i] = 0.0
                for k in range(n):
                    if k != i:
                        diff = nodes[i] - nodes[k]
                        if abs(diff) > eps:
                            D[i, i] += 1.0 / diff

    return D


def spectral_laplacian_1d(n):
    nodes, weights = gll_nodes_weights(n)
    D = differentiation_matrix(nodes)
    M = np.diag(weights)
    L = -D.T @ M @ D
    return L, nodes, weights


def spectral_derivative_2d(u, nx, ny):
    nodes_x, _ = gll_nodes_weights(nx)
    nodes_y, _ = gll_nodes_weights(ny)
    Dx = differentiation_matrix(nodes_x)
    Dy = differentiation_matrix(nodes_y)

    dudx = Dx @ u
    dudy = u @ Dy.T

    return dudx, dudy
