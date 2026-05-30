
import numpy as np


def assemble_hilbert_interaction_matrix(n, m=None):
    if m is None:
        m = n
    if n < 1 or m < 1:
        raise ValueError("Matrix dimensions must be positive.")
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    H = 1.0 / (i + j)
    return H


def r8gb_fa(n, ml, mu, ab):
    ab = np.array(ab, dtype=float, copy=True)
    pivot = np.zeros(n, dtype=int)
    info = 0
    m = ml + mu + 1

    for j in range(n - 1):

        pivot[j] = j
        pivot_val = abs(ab[mu, j])
        max_row = j
        for i in range(j + 1, min(j + ml + 1, n)):
            row_in_ab = mu + j - i
            if row_in_ab < 0:
                continue
            val = abs(ab[row_in_ab, i])
            if val > pivot_val:
                pivot_val = val
                max_row = i
                pivot[j] = i

        if pivot_val < 1e-15:
            info = j + 1
            continue


        if max_row != j:
            for col in range(max(0, j - mu), min(n, j + ml + 1)):
                idx_j = mu + j - col
                idx_m = mu + max_row - col
                if 0 <= idx_j < ab.shape[0] and 0 <= idx_m < ab.shape[0]:
                    ab[idx_j, col], ab[idx_m, col] = ab[idx_m, col], ab[idx_j, col]


        for i in range(j + 1, min(j + ml + 1, n)):
            row_in_ab = mu + j - i
            if row_in_ab < 0:
                continue
            factor = ab[row_in_ab, i] / ab[mu, j]
            ab[row_in_ab, i] = factor
            for col in range(j + 1, min(j + mu + 1, n)):
                idx_i = mu + i - col
                idx_j = mu + j - col
                if 0 <= idx_i < ab.shape[0] and 0 <= idx_j < ab.shape[0]:
                    ab[idx_i, col] -= factor * ab[idx_j, col]

    pivot[n - 1] = n - 1
    if abs(ab[mu, n - 1]) < 1e-15:
        info = n

    return ab, pivot, info


def r8gb_sl(n, ml, mu, ab, pivot, b):
    x = np.array(b, dtype=float, copy=True)
    m = ml + mu + 1


    for j in range(n - 1):
        if pivot[j] != j:
            x[j], x[pivot[j]] = x[pivot[j]], x[j]
        for i in range(j + 1, min(j + ml + 1, n)):
            row_in_ab = mu + j - i
            if row_in_ab < 0:
                continue
            factor = ab[row_in_ab, i]
            x[i] -= factor * x[j]


    for j in range(n - 1, -1, -1):
        if abs(ab[mu, j]) < 1e-15:
            x[j] = 0.0
        else:
            x[j] /= ab[mu, j]
        for i in range(max(0, j - mu), j):
            row_in_ab = mu + i - j
            if row_in_ab < 0:
                continue
            x[i] -= ab[row_in_ab, j] * x[j]

    return x


def banded_solve_tridiagonal(lower, diag, upper, rhs):
    n = len(diag)
    if n < 1:
        return np.array([])
    if n == 1:
        return np.array([rhs[0] / diag[0]])

    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = upper[i] / denom
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom

    denom = diag[n - 1] - lower[n - 2] * cp[n - 2]
    if abs(denom) < 1e-14:
        denom = 1e-14
    dp[n - 1] = (rhs[n - 1] - lower[n - 2] * dp[n - 2]) / denom

    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def matrix_chain_optimal_order(dims):
    dims = np.asarray(dims)
    if np.any(dims <= 0):
        raise ValueError("All dimensions must be positive.")
    n = len(dims) - 1
    if n < 2:
        return 0, np.array([])

    INF = float("inf")
    m = np.full((n, n), INF)
    s = np.zeros((n, n), dtype=int)

    for i in range(n):
        m[i, i] = 0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i, j]:
                    m[i, j] = cost
                    s[i, j] = k

    return int(m[0, n - 1]), s


def reconstruct_optimal_order(s, i, j):
    if i == j:
        return f"A{i}"
    k = s[i, j]
    left = reconstruct_optimal_order(s, i, k)
    right = reconstruct_optimal_order(s, k + 1, j)
    return f"({left} * {right})"


def build_sparse_interaction_matrix(n_sites, coupling_range=3, base_coupling=0.5):
    if n_sites < 1:
        raise ValueError("n_sites must be positive.")
    A = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        A[i, i] = 2.0
        for r in range(1, coupling_range + 1):
            if i + r < n_sites:
                val = base_coupling / r
                A[i, i + r] = val
                A[i + r, i] = val
    return A
