# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import legendre
from constants import hbar2_over_2m


def gauss_lobatto_nodes(N, a=-1.0, b=1.0):
    if N < 2:
        raise ValueError("gauss_lobatto_nodes requires N >= 2.")

    P = legendre(N - 1)

    dp = np.polyder(P)
    inner = np.roots(dp)

    inner = np.real(inner[np.isreal(inner)])
    inner = inner[(inner > -1.0) & (inner < 1.0)]
    inner = np.sort(inner)

    if inner.size < N - 2:
        inner = -np.cos(np.pi * np.arange(1, N - 1) / (N - 1))
    x = np.concatenate([[-1.0], inner[:N - 2], [1.0]])
    x = np.sort(x)

    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    return x


def gauss_lobatto_weights(N):
    x = gauss_lobatto_nodes(N, -1.0, 1.0)
    P = legendre(N - 1)
    pval = np.polyval(P, x)
    w = 2.0 / (N * (N - 1) * (pval ** 2))

    w[0] = 2.0 / (N * (N - 1))
    w[-1] = 2.0 / (N * (N - 1))
    return w


def vandermonde_quadrature_weights(x, a, b):
    N = x.size
    V = np.vander(x, N, increasing=True).T
    rhs = np.empty(N)
    for k in range(1, N + 1):
        rhs[k - 1] = (b ** k - a ** k) / k

    w = np.linalg.solve(V.T, rhs)
    return w


def lagrange_derivative_matrix(x):
    N = x.size
    D = np.zeros((N, N))

    lam = np.ones(N)
    for i in range(N):
        for k in range(N):
            if k != i:
                lam[i] *= 1.0 / (x[i] - x[k])
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (lam[i] / lam[j]) / (x[i] - x[j])

    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D


def solve_radial_schroedinger(rmax, N, l, V_func, n_eig=5,
                              method='gll', mass_nucleon=None):
    if rmax <= 0:
        raise ValueError("rmax must be positive.")
    if N < 4:
        raise ValueError("N must be at least 4.")
    if l < 0:
        raise ValueError("l must be non-negative.")


    r = np.linspace(0.0, rmax, N)
    h = r[1] - r[0]

    if mass_nucleon is None:
        h2m = hbar2_over_2m()
    else:
        h2m = hbar2_over_2m(mass_nucleon)
















    raise NotImplementedError("HOLE 1: radial Schrödinger solver core is not implemented.")


    wavefunctions = np.zeros((N, n_int))
    wavefunctions[1:N - 1, :] = eigvecs


    for k in range(n_int):
        norm = np.sqrt(np.trapz(wavefunctions[:, k] ** 2, r))
        if norm > 0:
            wavefunctions[:, k] /= norm


    bound_mask = eigvals < 0.0
    bound_vals = eigvals[bound_mask]
    bound_wf = wavefunctions[:, bound_mask]

    n_return = min(n_eig, bound_vals.size)
    if n_return == 0:

        n_return = min(n_eig, eigvals.size)
        return eigvals[:n_return], wavefunctions[:, :n_return], r

    return bound_vals[:n_return], bound_wf[:, :n_return], r


def radial_matrix_element(r, u1, u2, operator=None):
    if operator is None:
        integrand = u1 * u2
    else:
        integrand = u1 * operator(r) * u2
    return np.trapz(integrand, r)


def kinetic_energy_matrix_element(r, u1, u2, l, mass_nucleon=None):
    h2m = hbar2_over_2m(mass_nucleon)

    d2u2 = np.zeros_like(u2)
    h = np.diff(r)

    h_avg = np.mean(h)
    if np.max(np.abs(h - h_avg)) < 0.1 * h_avg:

        h2 = h_avg ** 2
        d2u2[2:-2] = (-u2[:-4] + 16.0 * u2[1:-3]
                      - 30.0 * u2[2:-2]
                      + 16.0 * u2[3:-1] - u2[4:]) / (12.0 * h2)

        d2u2[0] = (2.0 * u2[0] - 5.0 * u2[1] + 4.0 * u2[2] - u2[3]) / h2
        d2u2[1] = (u2[0] - 2.0 * u2[1] + u2[2]) / h2
        d2u2[-2] = (u2[-3] - 2.0 * u2[-2] + u2[-1]) / h2
        d2u2[-1] = (2.0 * u2[-1] - 5.0 * u2[-2]
                    + 4.0 * u2[-3] - u2[-4]) / h2
    else:

        for i in range(1, r.size - 1):
            hp = r[i + 1] - r[i]
            hm = r[i] - r[i - 1]
            d2u2[i] = 2.0 * (hm * u2[i + 1] - (hp + hm) * u2[i]
                             + hp * u2[i - 1]) / (hp * hm * (hp + hm))
        d2u2[0] = d2u2[1]
        d2u2[-1] = d2u2[-2]

    r_safe = np.where(r < 1e-6, 1e-6, r)
    centrifugal = h2m * l * (l + 1) / (r_safe ** 2)
    integrand = u1 * (-h2m * d2u2 + centrifugal * u2)
    return np.trapz(integrand, r)
