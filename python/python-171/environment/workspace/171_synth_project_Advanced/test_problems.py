# -*- coding: utf-8 -*-

import numpy as np
import math
from sparse_matrix import dif2_r8ge, SparseMatrixOperator
from special_functions import steinerberger_function
from random_tools import random_spd_matrix, random_spd_with_clustered_spectrum






def hankel_spd_cholesky_lower(n, lii, liim1):
    if len(lii) != n:
        raise ValueError("lii length must equal n.")
    if len(liim1) != n - 1:
        raise ValueError("liim1 length must equal n-1.")
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = lii[i]
    for i in range(n - 1):
        L[i + 1, i] = liim1[i]

    for i in range(2, n):
        for j in range(i - 1):
            if (i + j) % 2 == 0:
                q = (i + j) // 2
                r = q
            else:
                q = (i + j - 1) // 2
                r = q + 1

            alpha = 0.0
            for s in range(q):
                alpha += L[q, s] * L[r, s]

            beta = 0.0
            for t in range(j):
                beta += L[i, t] * L[j, t]

            if abs(L[j, j]) < 1e-30:
                L[i, j] = 0.0
            else:
                L[i, j] = (alpha - beta) / L[j, j]

    H = L @ L.T
    return L, H


def hankel_spd_from_moments(n, moments):
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(moments):
                H[i, j] = moments[idx]
            else:
                H[i, j] = 0.0

    lam = np.linalg.eigvalsh(H)
    if lam[0] <= 0:
        H += (-lam[0] + 1e-6) * np.eye(n)
    return H






def anisotropic_diffusion_2d(nx, ny, epsilon_x, epsilon_y, hx=None, hy=None):
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = epsilon_x / (hx ** 2)
    cy = epsilon_y / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A


def helmholtz_2d(nx, ny, k, hx=None, hy=None):
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = 1.0 / (hx ** 2)
    cy = 1.0 / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy + k ** 2

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A






def rosenbrock_rhs(m, x_grid):
    x = np.asarray(x_grid, dtype=float)
    if x.ndim == 1 and x.size == m:
        val = 0.0
        for i in range(m - 1):
            val += 100.0 * (x[i] - x[i + 1]) ** 2 + (x[i] - 1.0) ** 2
        return val
    else:

        n = x.shape[1] if x.ndim > 1 else 1
        val = np.zeros(n, dtype=float)
        for i in range(m - 1):
            val += 100.0 * (x[i, :] - x[i + 1, :]) ** 2 + (x[i, :] - 1.0) ** 2
        return val


def camel_rhs(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x1, x2 = x[0], x[1]
    else:
        x1, x2 = x[0, :], x[1, :]
    return (4.0 * x1 ** 2
            - 2.1 * x1 ** 4
            + (1.0 / 3.0) * x1 ** 6
            + x1 * x2
            - 4.0 * x2 ** 2
            + 4.0 * x2 ** 4)






def cavity_flow_stokes_matrix(nx, ny, nu=1.0, hx=None, hy=None):
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = nu / (hx ** 2)
    cy = nu / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A


def cavity_flow_rhs(nx, ny):
    N = nx * ny
    b = np.zeros(N, dtype=float)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            y = (j + 1) / (ny + 1)
            b[idx] = math.exp(-10.0 * (1.0 - y))
    return b






def build_test_problem(problem_id, n, extra=None):
    extra = extra if extra is not None else {}
    rng = np.random.default_rng(extra.get('seed', 42))

    if problem_id == 'dif2':
        A = dif2_r8ge(n)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'hankel':
        lii = np.ones(n, dtype=float)
        liim1 = 0.5 * np.ones(n - 1, dtype=float)
        _, A = hankel_spd_cholesky_lower(n, lii, liim1)
        x_exact = np.sin(np.linspace(0, math.pi, n))
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'aniso2d':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        eps_x = extra.get('eps_x', 1.0)
        eps_y = extra.get('eps_y', 0.01)
        A = anisotropic_diffusion_2d(nx, ny, eps_x, eps_y)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'helmholtz2d':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        k = extra.get('k', 10.0)
        A = helmholtz_2d(nx, ny, k)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'random_spd':
        A, _, _ = random_spd_matrix(n, seed=extra.get('seed', 42))
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'clustered':
        clusters = extra.get('clusters', [(0.1, 0.05, n // 4), (10.0, 1.0, n // 4), (100.0, 5.0, n // 2)])
        A, _, _ = random_spd_with_clustered_spectrum(n, clusters, seed=extra.get('seed', 42))
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'cavity':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        A = cavity_flow_stokes_matrix(nx, ny, nu=extra.get('nu', 1.0))
        b = cavity_flow_rhs(nx, ny)
        x_exact = np.linalg.solve(A, b)
        return A, b, x_exact

    elif problem_id == 'steinerberger':
        A = dif2_r8ge(n)
        x_grid = np.linspace(0.0, 1.0, n)
        sb_param = extra.get('sb_param', 20)
        b = steinerberger_function(sb_param, x_grid)

        h = 1.0 / (n + 1)
        b = b * (h ** 2)
        x_exact = np.linalg.solve(A, b)
        return A, b, x_exact

    else:
        raise ValueError(f"Unknown problem_id: {problem_id}")
