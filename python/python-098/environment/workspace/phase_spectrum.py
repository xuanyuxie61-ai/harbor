# -*- coding: utf-8 -*-

import numpy as np


def horner_eval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    d = coeffs.shape[0] - 1
    if d < 0:
        return np.zeros_like(x)
    value = np.full_like(x, coeffs[d])
    for i in range(d - 1, -1, -1):
        value = value * x + coeffs[i]
    return value


def legendre_polynomials(n, x):
    x = float(x)
    if n < 0:
        return np.array([]), np.array([])
    cx = np.zeros(n + 1)
    cpx = np.zeros(n + 1)
    cx[0] = 1.0
    cpx[0] = 0.0
    if n < 1:
        return cx, cpx
    cx[1] = x
    cpx[1] = 1.0
    for i in range(2, n + 1):
        cx[i] = ((2.0 * i - 1.0) * x * cx[i - 1] - (i - 1.0) * cx[i - 2]) / i
        cpx[i] = ((2.0 * i - 1.0) * (cx[i - 1] + x * cpx[i - 1]) -
                  (i - 1.0) * cpx[i - 2]) / i
    return cx, cpx


def legendre_polynomials_array(n, x_arr):
    x_arr = np.asarray(x_arr, dtype=float)
    m = x_arr.shape[0]
    cx = np.zeros((m, n + 1))
    if n >= 0:
        cx[:, 0] = 1.0
    if n >= 1:
        cx[:, 1] = x_arr
    for i in range(2, n + 1):
        cx[:, i] = ((2.0 * i - 1.0) * x_arr * cx[:, i - 1] -
                    (i - 1.0) * cx[:, i - 2]) / i
    return cx


def chebyshev_nodes(n):
    k = np.arange(n, dtype=float)
    return np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))


def chebyshev_polynomials(n, x):
    x = float(x)
    if n < 0:
        return np.array([])
    T = np.zeros(n + 1)
    T[0] = 1.0
    if n >= 1:
        T[1] = x
    for i in range(2, n + 1):
        T[i] = 2.0 * x * T[i - 1] - T[i - 2]
    return T


def design_hologram_phase_2d(x_grid, y_grid, legendre_coeffs, Lx=1.0, Ly=1.0):
    legendre_coeffs = np.asarray(legendre_coeffs, dtype=float)
    M, N = legendre_coeffs.shape
    M -= 1
    N -= 1
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)




    raise NotImplementedError("Hole 1: design_hologram_phase_2d 需要实现")


def reconstruct_phase_from_spectrum(phase_samples, x_nodes, y_nodes, max_degree):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    phase_samples = np.asarray(phase_samples, dtype=float)
    nx = x_nodes.shape[0]
    ny = y_nodes.shape[0]
    if phase_samples.shape != (ny, nx):
        raise ValueError("phase_samples shape must match (len(y_nodes), len(x_nodes))")

    x_tilde = np.clip(x_nodes, -1.0, 1.0)
    y_tilde = np.clip(y_nodes, -1.0, 1.0)

    Px = legendre_polynomials_array(max_degree, x_tilde)
    Py = legendre_polynomials_array(max_degree, y_tilde)


    A = np.kron(Px, Py)
    b = phase_samples.ravel()

    ATA = A.T @ A
    ATb = A.T @ b

    reg = 1e-10 * np.eye(ATA.shape[0])
    coeffs_flat = np.linalg.solve(ATA + reg, ATb)
    coeffs = coeffs_flat.reshape(max_degree + 1, max_degree + 1)
    return coeffs


def phase_gradient_2d(x_grid, y_grid, legendre_coeffs, Lx=1.0, Ly=1.0):
    legendre_coeffs = np.asarray(legendre_coeffs, dtype=float)
    M, N = legendre_coeffs.shape
    M -= 1
    N -= 1
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    x_tilde = np.clip(2.0 * x_grid / Lx, -1.0, 1.0)
    y_tilde = np.clip(2.0 * y_grid / Ly, -1.0, 1.0)


    Px = np.zeros((len(x_grid), M + 1))
    Py = np.zeros((len(y_grid), N + 1))
    for i, xt in enumerate(x_tilde):
        _, dPx = legendre_polynomials(M, xt)
        Px[i, :] = dPx
    for j, yt in enumerate(y_tilde):
        _, dPy = legendre_polynomials(N, yt)
        Py[j, :] = dPy

    dphi_dx = (2.0 / Lx) * np.einsum('ij,ni,mj->mn', legendre_coeffs, Py, Px)


    Px_val = legendre_polynomials_array(M, x_tilde)
    Py_val = legendre_polynomials_array(N, y_tilde)

    Px_d = np.zeros_like(Px_val)
    Py_d = np.zeros_like(Py_val)
    for i, xt in enumerate(x_tilde):
        _, dPx = legendre_polynomials(M, xt)
        Px_d[i, :] = dPx
    for j, yt in enumerate(y_tilde):
        _, dPy = legendre_polynomials(N, yt)
        Py_d[j, :] = dPy

    dphi_dy = (2.0 / Ly) * np.einsum('ij,ni,mj->mn', legendre_coeffs, Py_d, Px_val)
    return dphi_dx, dphi_dy
