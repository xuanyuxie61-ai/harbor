# -*- coding: utf-8 -*-

import numpy as np
import math


def vec_norm(v, ord=2):
    v = np.asarray(v, dtype=float).flatten()
    if v.size == 0:
        return 0.0
    if ord == 2:
        return np.linalg.norm(v, 2)
    elif ord == 1:
        return np.linalg.norm(v, 1)
    elif ord == np.inf:
        return np.linalg.norm(v, np.inf)
    else:
        return np.linalg.norm(v, ord)


def mat_norm(A, ord=2):
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0.0
    return np.linalg.norm(A, ord)


def residual_dense(A, x, b):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if A.shape[0] != b.size or A.shape[1] != x.size:
        raise ValueError("Dimension mismatch in residual_dense.")
    return b - A @ x


def residual_tridiag(a, x, b):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    if a.shape != (3, n):
        raise ValueError("R83 matrix must have shape (3, n).")
    ax = np.zeros(n, dtype=float)
    ax[0] = a[1, 0] * x[0] + a[0, 1] * x[1] if n > 1 else a[1, 0] * x[0]
    for i in range(1, n - 1):
        ax[i] = a[2, i - 1] * x[i - 1] + a[1, i] * x[i] + a[0, i + 1] * x[i + 1]
    if n > 1:
        ax[n - 1] = a[2, n - 2] * x[n - 2] + a[1, n - 1] * x[n - 1]
    return b - ax


def residual_banded(mu, a, x, b):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    if a.shape != (mu + 1, n):
        raise ValueError("R8PBU matrix shape mismatch.")
    ax = np.zeros(n, dtype=float)
    for i in range(n):
        ax[i] = a[mu, i] * x[i]
    for k in range(1, mu + 1):
        for j in range(mu + 1 - k, n):
            i_eq = k + j - mu - 1
            ax[i_eq] += a[mu - k, j] * x[j]
            ax[j] += a[mu - k, j] * x[i_eq]
    return b - ax


def residual_sd(ndiag, offset, a, x, b):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    ax = np.zeros(n, dtype=float)
    for i in range(n):
        for jd in range(ndiag):
            off = offset[jd]
            if off >= 0:
                j = i + off
                if 0 <= j < n:
                    ax[i] += a[i, jd] * x[j]
                    if off != 0:
                        ax[j] += a[i, jd] * x[i]
    return b - ax


def residual_sparse(nz_num, row, col, a_val, x, b):
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    ax = np.zeros(n, dtype=float)
    for k in range(nz_num):
        i = row[k]
        j = col[k]
        ax[i] += a_val[k] * x[j]
    return b - ax


def checksum_vector(x, base=11):
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return 0
    n = x.size
    weights = np.arange(n, 0, -1, dtype=float)
    s = float(np.dot(weights, x))

    cs = ((s % base) + base) % base
    return int(cs)


def verify_checksum(x, expected, base=11, tol=1e-6):
    cs = checksum_vector(x, base)
    return abs(cs - expected) < tol or abs(cs - expected - base) < tol or abs(cs - expected + base) < tol


def safe_divide(a, b, default=0.0):
    if abs(b) < 1e-30:
        return default
    return a / b


def condition_number_estimate(A, method='power', max_iter=50):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    x = np.random.randn(n)
    x = x / vec_norm(x)
    for _ in range(max_iter):
        y = A @ x
        norm_y = vec_norm(y)
        if norm_y < 1e-30:
            break
        x = y / norm_y
    lam_max = float(x @ (A @ x))


    x2 = np.random.randn(n)
    x2 = x2 / vec_norm(x2)


    eps_reg = 1e-12
    for _ in range(max_iter):
        try:
            y2 = np.linalg.solve(A + eps_reg * np.eye(n), x2)
        except np.linalg.LinAlgError:
            y2 = x2
        norm_y2 = vec_norm(y2)
        if norm_y2 < 1e-30:
            break
        x2 = y2 / norm_y2
    lam_min = float(x2 @ (A @ x2))
    if lam_min <= 0:
        lam_min = eps_reg
    kappa = lam_max / lam_min
    return kappa, lam_max, lam_min


def print_header(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_vec(name, v, max_show=8):
    v = np.asarray(v, dtype=float).flatten()
    n = v.size
    if n <= max_show:
        s = ", ".join(f"{vi:.6e}" for vi in v)
    else:
        s = ", ".join(f"{vi:.6e}" for vi in v[:max_show // 2])
        s += ", ... , "
        s += ", ".join(f"{vi:.6e}" for vi in v[-max_show // 2:])
    print(f"  {name}[{n}] = [{s}]")


def is_spd(A, tol=1e-10):
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
