# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import eval_legendre


def mono_next_grlex(m, x):
    x = np.asarray(x, dtype=int)
    x_next = x.copy()

    for i in range(m - 1, -1, -1):
        if x_next[i] > 0:
            x_next[i] -= 1
            if i + 1 < m:
                x_next[i + 1] += np.sum(x[i:]) + 1
                x_next[i:] = 0
                x_next[i] = 0
            break
    else:

        x_next[-1] = 1
    return x_next


def mono_rank_grlex(m, x):
    x = np.asarray(x, dtype=int)
    total = np.sum(x)

    rank = 0
    for t in range(total):
        rank += int(np.math.comb(t + m - 1, m - 1))


    if m <= 3 and total <= 10:
        current = np.zeros(m, dtype=int)
        current[-1] = total
        idx = 0
        while not np.array_equal(current, x):
            current = mono_next_grlex_within_degree(m, current)
            idx += 1
        rank += idx
    else:

        rank += 0
    return rank


def mono_next_grlex_within_degree(m, x):
    x = np.asarray(x, dtype=int)
    x_next = x.copy()
    for i in range(m - 2, -1, -1):
        if x_next[i] > 0:
            x_next[i] -= 1
            x_next[i + 1] += 1
            return x_next
    return x_next


def enumerate_grlex_indices(d, p):
    indices = []

    def recurse(dim, remaining, current):
        if dim == d - 1:
            current.append(remaining)
            indices.append(current.copy())
            current.pop()
            return
        for k in range(remaining + 1):
            current.append(k)
            recurse(dim + 1, remaining - k, current)
            current.pop()
    for total in range(p + 1):
        recurse(0, total, [])
    return np.array(indices, dtype=int)


def legendre_polynomial_1d(n, xi):
    xi = np.asarray(xi, dtype=np.float64)
    P = np.zeros((len(xi), n + 1), dtype=np.float64)
    for k in range(n + 1):

        pk = eval_legendre(k, xi)

        norm = np.sqrt((2.0 * k + 1.0) / 2.0)
        P[:, k] = pk * norm
    return P


def gpc_basis_evaluation(d, p, xi_samples):
    xi_samples = np.asarray(xi_samples, dtype=np.float64)
    n_samples = xi_samples.shape[0]
    multi_indices = enumerate_grlex_indices(d, p)
    n_basis = len(multi_indices)


    max_deg = p
    P_1d = []
    for dim in range(d):
        P_1d.append(legendre_polynomial_1d(max_deg, xi_samples[:, dim]))

    Psi = np.ones((n_samples, n_basis), dtype=np.float64)
    for b in range(n_basis):
        alpha = multi_indices[b]
        for dim in range(d):
            Psi[:, b] *= P_1d[dim][:, alpha[dim]]
    return Psi, multi_indices


def gpc_coefficients_collocation(u_samples, Psi):
    u_samples = np.asarray(u_samples, dtype=np.float64)

    M = Psi.T @ Psi
    rhs = Psi.T @ u_samples

    M += 1e-12 * np.eye(M.shape[0])
    coeffs = np.linalg.solve(M, rhs)
    return coeffs


def gpc_mean_variance(coeffs, multi_indices):
    mean = coeffs[0]
    variance = np.sum(coeffs[1:] ** 2)
    return mean, variance


def gpc_sobol_indices(coeffs, multi_indices):
    _, var = gpc_mean_variance(coeffs, multi_indices)
    if var < 1e-15:
        return np.zeros(multi_indices.shape[1])
    d = multi_indices.shape[1]
    S1 = np.zeros(d)
    for i in range(d):
        mask = (multi_indices[:, i] > 0) & (np.sum(multi_indices, axis=1) == multi_indices[:, i])
        S1[i] = np.sum(coeffs[mask] ** 2) / var
    return S1
