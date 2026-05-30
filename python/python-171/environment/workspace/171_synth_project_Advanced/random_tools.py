# -*- coding: utf-8 -*-

import numpy as np
import math






_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def halton_value(i, m):
    if m < 1 or m > _PRIMES.size:
        raise ValueError(f"Dimension m must be in [1, {_PRIMES.size}]")
    i = abs(int(math.floor(i)))
    r = np.zeros(m, dtype=float)
    t = np.full(m, i, dtype=int)
    prime_inv = 1.0 / _PRIMES[:m].astype(float)

    while np.any(t != 0):
        for j in range(m):
            d = t[j] % _PRIMES[j]
            r[j] += d * prime_inv[j]
            prime_inv[j] /= _PRIMES[j]
            t[j] = t[j] // _PRIMES[j]
    return r


def halton_sequence(i1, i2, m):
    if m < 1 or m > _PRIMES.size:
        raise ValueError(f"Dimension m must be in [1, {_PRIMES.size}]")
    if i1 <= i2:
        i3 = 1
    else:
        i3 = -1
    n = abs(i2 - i1) + 1
    r = np.zeros((m, n), dtype=float)
    k = 0
    for i in range(i1, i2 + i3, i3):
        t = np.full(m, i, dtype=int)
        prime_inv = 1.0 / _PRIMES[:m].astype(float)
        while np.any(t != 0):
            for j in range(m):
                d = t[j] % _PRIMES[j]
                r[j, k] += d * prime_inv[j]
                prime_inv[j] /= _PRIMES[j]
                t[j] = t[j] // _PRIMES[j]
        k += 1
    return r


def halton_scrambled(i, m, seed=0):
    rng = np.random.default_rng(seed)
    base = halton_value(i, m)

    shift = rng.random(m)
    return np.mod(base + shift, 1.0)






def householder_column(n, a_vec, k):
    a_vec = np.asarray(a_vec, dtype=float).flatten()
    v = np.zeros(n, dtype=float)
    if k < 1 or k >= n:
        return v
    s = math.sqrt(np.sum(a_vec[k:] ** 2))
    if s < 1e-30:
        return v
    if a_vec[k] < 0:
        v[k] = a_vec[k] - abs(s)
    else:
        v[k] = a_vec[k] + abs(s)
    v[k + 1:] = a_vec[k + 1:]
    norm_v = math.sqrt(np.sum(v[k:] ** 2))
    v[k:] /= norm_v
    return v


def apply_householder_right(n, A, v):
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float).flatten()
    vv = float(v @ v)
    if vv < 1e-30:
        return A.copy()
    return A - 2.0 * (A @ v[:, None]) @ v[None, :] / vv


def random_orthogonal_matrix(n, seed=None):
    rng = np.random.default_rng(seed)
    A = np.eye(n, dtype=float)
    for j in range(n - 1):
        x = np.zeros(n, dtype=float)
        x[j:] = rng.standard_normal(n - j)
        v = householder_column(n, x, j)
        A = apply_householder_right(n, A, v)

        if rng.random() > 0.5:
            k = rng.integers(0, n)
            A[k, :] *= -1.0
    return A


def random_spd_matrix(n, seed=None, eigenvalue_min=1e-3, eigenvalue_max=1.0):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(eigenvalue_min, eigenvalue_max, size=n)
    Q = random_orthogonal_matrix(n, seed=rng.integers(0, 2 ** 31))
    A = Q @ np.diag(lam) @ Q.T

    A = 0.5 * (A + A.T)
    return A, lam, Q


def random_spd_with_clustered_spectrum(n, clusters, seed=None):
    rng = np.random.default_rng(seed)
    lam = []
    for c, w, count in clusters:
        lam.extend(rng.uniform(c - w, c + w, size=count).tolist())
    lam = np.array(lam[:n], dtype=float)
    if lam.size < n:
        lam = np.concatenate([lam, rng.uniform(0.01, 1.0, size=n - lam.size)])
    lam = np.clip(np.sort(lam)[::-1], 1e-6, None)
    Q = random_orthogonal_matrix(n, seed=rng.integers(0, 2 ** 31))
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    return A, lam, Q






def random_probe_vector(n, distribution='rademacher', seed=None):
    rng = np.random.default_rng(seed)
    if distribution == 'rademacher':
        return rng.choice([-1.0, 1.0], size=n)
    elif distribution == 'gaussian':
        return rng.standard_normal(n)
    elif distribution == 'halton':

        i = rng.integers(0, 100000)
        return halton_value(i, 1) * 2.0 - 1.0
    else:
        return rng.standard_normal(n)


def hutchinson_trace_estimator(matvec, n, num_samples=30, seed=None):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(num_samples):
        v = rng.choice([-1.0, 1.0], size=n)
        Av = matvec(v)
        total += float(v @ Av)
    return total / num_samples


def randomized_svd_approx(matvec, n, rank, power_iterations=2, seed=None):
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((n, rank))
    Y = np.zeros((n, rank), dtype=float)
    for j in range(rank):
        Y[:, j] = matvec(Omega[:, j])


    for _ in range(power_iterations):
        Z = np.zeros((n, rank), dtype=float)
        for j in range(rank):
            Z[:, j] = matvec(Y[:, j])
        Y = Z

    Q, _ = np.linalg.qr(Y)

    B = np.zeros((rank, rank), dtype=float)
    for j in range(rank):
        AQj = matvec(Q[:, j])
        for i in range(rank):
            B[i, j] = float(Q[:, i] @ AQj)

    B = 0.5 * (B + B.T)
    lam, V = np.linalg.eigh(B)

    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V = V[:, idx]
    U = Q @ V
    return U, lam
