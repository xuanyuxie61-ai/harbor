#!/usr/bin/env python3

import numpy as np
import time






def r8pbl_zeros(n, mu):
    a = np.zeros((mu + 1, n), dtype=float)
    return a


def r8pbl_dif2(n, mu):
    a = r8pbl_zeros(n, mu)
    for j in range(n):
        a[0, j] = 2.0
        if mu >= 1 and j + 1 < n:
            a[1, j + 1] = -1.0
    return a


def r8pbl_to_r8ge(a, n, mu):
    a_dense = np.zeros((n, n), dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            a_dense[i, j] = a[j - i, j]
            a_dense[j, i] = a[j - i, j]
    return a_dense


def r8pbl_mv(a, n, mu, x):
    b = np.zeros(n, dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            bij = a[j - i, j]
            b[i] += bij * x[j]
            if i != j:
                b[j] += bij * x[i]
    return b






def dgefa(a, n):
    a = a.copy()
    ipvt = np.arange(n)
    info = 0

    for k in range(n - 1):

        max_idx = np.argmax(np.abs(a[k:n, k])) + k
        if abs(a[max_idx, k]) < 1e-15:
            info = k
            continue
        if max_idx != k:
            a[[k, max_idx]] = a[[max_idx, k]]
            ipvt[[k, max_idx]] = ipvt[[max_idx, k]]


        for i in range(k + 1, n):
            a[i, k] /= a[k, k]
            a[i, k + 1:n] -= a[i, k] * a[k, k + 1:n]

    if abs(a[n - 1, n - 1]) < 1e-15:
        info = n - 1

    return a, ipvt, info


def dgesl(a, n, ipvt, b, job=0):
    x = b.copy()

    if job == 0:

        for k in range(n - 1):

            if ipvt[k] != k:
                x[k], x[ipvt[k]] = x[ipvt[k]], x[k]
            for i in range(k + 1, n):
                x[i] -= a[i, k] * x[k]


        for k in range(n - 1, -1, -1):
            x[k] /= a[k, k]
            for i in range(k):
                x[i] -= a[i, k] * x[k]
    else:

        for k in range(n):
            x[k] /= a[k, k]
            for i in range(k + 1, n):
                x[i] -= a[k, i] * x[k]
        for k in range(n - 1, -1, -1):
            for i in range(k):
                x[i] -= a[i, k] * x[i]
            if ipvt[k] != k:
                x[k], x[ipvt[k]] = x[ipvt[k]], x[k]

    return x






def hankel_cholesky_upper(h_first_row):
    h_first_row = np.asarray(h_first_row, dtype=float)
    n = len(h_first_row)

    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                H[i, j] = h_first_row[idx]
            else:
                H[i, j] = 0.0


    H += np.eye(n) * 1e-8


    R = np.zeros((n, n), dtype=float)
    for j in range(n):
        s = H[j, j]
        for k in range(j):
            s -= R[k, j] ** 2
        if s <= 1e-15:
            s = 1e-15
        R[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            s = H[j, i]
            for k in range(j):
                s -= R[k, j] * R[k, i]
            if abs(R[j, j]) > 1e-15:
                R[j, i] = s / R[j, j]
            else:
                R[j, i] = 0.0

    return R


def hankel_spd_cholesky_lower(diag, subdiag):
    n = len(diag)
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = diag[i]
    for i in range(1, n):
        L[i, i - 1] = subdiag[i - 1]


    for j in range(n):
        for i in range(j + 1, n):
            if i > 0 and j > 0:
                L[i, j] = L[i - 1, j - 1]
            elif j == 0 and i > 1:

                L[i, j] = L[i - 1, j + 1] if j + 1 < i else 0.0

    return L


def hankel_covariance_factor(signal):
    signal = np.asarray(signal, dtype=float)
    n = min(len(signal), 25)
    sig = signal[:n]


    sig_c = sig - np.mean(sig)
    autocorr = np.correlate(sig_c, sig_c, mode='full')
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / max(autocorr[0], 1e-15)

    for i in range(len(autocorr)):
        autocorr[i] *= np.exp(-0.1 * i)
    autocorr = np.clip(autocorr, -0.99, 0.99)


    from scipy.linalg import toeplitz
    cov = toeplitz(autocorr)
    cov += np.eye(n) * 1e-4


    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    return L






def solve_banded_linear_system(params):
    n = max(20, params.get('Nx', 81))
    mu = 1


    a_pbl = r8pbl_dif2(n, mu)
    A_dense = r8pbl_to_r8ge(a_pbl, n, mu)


    b = np.sin(np.linspace(0.0, np.pi, n))


    t0 = time.perf_counter()
    a_lu, ipvt, info = dgefa(A_dense, n)
    if info != 0:

        A_dense += np.eye(n) * 1e-6
        a_lu, ipvt, info = dgefa(A_dense, n)
    x = dgesl(a_lu, n, ipvt, b, job=0)
    t_solve = time.perf_counter() - t0


    b_recon = A_dense @ x
    resid = np.linalg.norm(b_recon - b) / np.linalg.norm(b)

    return resid, t_solve


if __name__ == '__main__':
    p = {'Nx': 81}
    r, t = solve_banded_linear_system(p)
    print("Residual:", r, "Time:", t)
