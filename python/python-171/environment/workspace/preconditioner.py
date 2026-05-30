# -*- coding: utf-8 -*-

import numpy as np
import math
from orthogonal_polynomials import build_polynomial_preconditioner_spectrum






def jacobi_preconditioner(A):
    A = np.asarray(A, dtype=float)
    diag = np.diag(A).copy()
    diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)
    inv_diag = 1.0 / diag

    def apply(r):
        return inv_diag * np.asarray(r, dtype=float).flatten()

    return apply


def ssor_preconditioner(A, omega=1.5):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    D = np.diag(A).copy()
    D = np.where(np.abs(D) < 1e-30, 1.0, D)

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()

        y = np.zeros(n, dtype=float)
        for i in range(n):
            sum_val = r[i]
            for j in range(i):
                sum_val -= omega * A[i, j] * y[j]
            y[i] = sum_val / D[i]

        z = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            sum_val = D[i] * y[i]
            for j in range(i + 1, n):
                sum_val -= omega * A[j, i] * z[j]
            z[i] = sum_val / D[i]
        return z * omega * (2.0 - omega)

    return apply


def incomplete_cholesky_ic0(A, drop_tol=1e-12):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    nz_mask = np.abs(A) > drop_tol

    for i in range(n):
        for j in range(i + 1):
            if not nz_mask[i, j]:
                continue
            sum_val = A[i, j]
            for k in range(j):
                if nz_mask[i, k] and nz_mask[j, k]:
                    sum_val -= L[i, k] * L[j, k]
            if i == j:
                if sum_val <= 0:
                    sum_val = abs(sum_val) + 1e-10
                L[i, j] = math.sqrt(sum_val)
            else:
                if abs(L[j, j]) > 1e-30:
                    L[i, j] = sum_val / L[j, j]

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()

        y = np.zeros(n, dtype=float)
        for i in range(n):
            sum_val = r[i]
            for j in range(i):
                sum_val -= L[i, j] * y[j]
            if abs(L[i, i]) > 1e-30:
                y[i] = sum_val / L[i, i]

        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            sum_val = y[i]
            for j in range(i + 1, n):
                sum_val -= L[j, i] * x[j]
            if abs(L[i, i]) > 1e-30:
                x[i] = sum_val / L[i, i]
        return x

    return apply






def polynomial_spectral_preconditioner(A, poly_type='laguerre', n_nodes=16):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    D = np.diag(A).copy()
    D = np.where(np.abs(D) < 1e-30, 1.0, D)
    D_inv_sqrt = 1.0 / np.sqrt(D)


    lam_max, lam_min = _estimate_extreme_eigenvalues(A, max_iter=20)
    kappa = lam_max / max(lam_min, 1e-30)


    nodes, weights = build_polynomial_preconditioner_spectrum(n_nodes, poly_type)







    def apply(r):
        r = np.asarray(r, dtype=float).flatten()



        pass

    return apply


def _estimate_extreme_eigenvalues(A, max_iter=30):
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    for _ in range(max_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-30:
            break
        x = y / norm_y
    lam_max = float(x @ (A @ x))


    x2 = np.random.randn(n)
    x2 = x2 / np.linalg.norm(x2)
    shift = lam_max * 1.01
    for _ in range(max_iter):
        try:
            y2 = np.linalg.solve(A - shift * np.eye(n), x2)
        except np.linalg.LinAlgError:
            y2 = x2
        norm_y2 = np.linalg.norm(y2)
        if norm_y2 < 1e-30:
            break
        x2 = y2 / norm_y2
    lam_min = float(x2 @ (A @ x2))
    if lam_min <= 0:
        lam_min = 1e-6
    return lam_max, lam_min






def two_grid_preconditioner(A_coarse, prolongation, restriction):
    P = np.asarray(prolongation, dtype=float)
    R = np.asarray(restriction, dtype=float)
    A_c = np.asarray(A_coarse, dtype=float)
    n_f = P.shape[0]


    try:
        A_c_inv = np.linalg.inv(A_c)
    except np.linalg.LinAlgError:
        A_c_inv = np.eye(A_c.shape[0])

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        x = np.zeros(n_f, dtype=float)

        diag = np.diag(A_coarse if False else np.eye(n_f))

        r_c = R @ r
        e_c = A_c_inv @ r_c
        x = P @ e_c
        return x

    return apply






def block_diagonal_preconditioner(blocks):
    inverses = []
    for B in blocks:
        B = np.asarray(B, dtype=float)
        try:
            invB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            invB = np.eye(B.shape[0])
        inverses.append(invB)

    def apply(r):
        r = np.asarray(r, dtype=float).flatten()
        x = np.array([])
        offset = 0
        for invB in inverses:
            m = invB.shape[0]
            x = np.concatenate([x, invB @ r[offset:offset + m]])
            offset += m
        return x

    return apply
