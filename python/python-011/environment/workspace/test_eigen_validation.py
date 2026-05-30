# -*- coding: utf-8 -*-

import numpy as np


def householder_column(a_vec, k):
    n = a_vec.size
    v = np.zeros(n)
    v[k:] = a_vec[k:].copy()
    norm_x = np.linalg.norm(v[k:])
    if norm_x < 1e-15:
        return v
    if v[k] >= 0:
        s = norm_x
    else:
        s = -norm_x
    v[k] += s

    norm_v = np.linalg.norm(v[k:])
    if norm_v > 1e-15:
        v[k:] /= norm_v
    return v


def apply_householder_right(a, v):
    av = a @ v
    return a - 2.0 * np.outer(av, v)


def random_orthogonal_matrix(n):
    if n <= 0:
        return np.zeros((0, 0))
    X = np.random.randn(n, n)
    Q, R = np.linalg.qr(X)

    diag_r = np.diag(R)
    signs = np.where(diag_r >= 0, 1.0, -1.0)
    Q = Q * signs[np.newaxis, :]
    return Q


def generate_symmetric_test_matrix(n, lambda_mean=0.0, lambda_std=1.0):
    if n <= 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    lam = np.random.normal(lambda_mean, lambda_std, size=n)
    Q = random_orthogonal_matrix(n)
    A = Q @ np.diag(lam) @ Q.T
    return A, Q, lam


def generate_nonsymmetric_test_matrix(n, lambda_mean=0.0, lambda_std=1.0):
    if n <= 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    lam = np.random.normal(lambda_mean, lambda_std, size=n)
    T = np.triu(np.random.randn(n, n) * 0.5, k=1)
    np.fill_diagonal(T, lam)
    Q = random_orthogonal_matrix(n)
    A = Q.T @ T @ Q
    return A, Q, lam


def validate_eigensolver(A, lam_exact, method='numpy'):
    if A.size == 0:
        return {'max_abs_err': 0.0, 'mean_rel_err': 0.0, 'residual_norm': 0.0}
    if method == 'numpy':
        lam_computed = np.linalg.eigvals(A)
    else:
        raise ValueError("不支持的求解器方法。")

    lam_exact_s = np.sort(np.real(lam_exact))
    lam_comp_s = np.sort(np.real(lam_computed))
    abs_err = np.abs(lam_exact_s - lam_comp_s)
    max_abs_err = np.max(abs_err)
    rel_err = safe_divide(abs_err, np.abs(lam_exact_s) + 1e-15)
    mean_rel_err = np.mean(rel_err)

    if A.shape[0] <= 200:
        w, v = np.linalg.eig(A)
        try:
            v_inv = np.linalg.inv(v)
            recon = v @ np.diag(w) @ v_inv
            res_norm = np.linalg.norm(A - recon, ord='fro')
        except np.linalg.LinAlgError:
            res_norm = np.inf
    else:
        res_norm = None
    return {
        'max_abs_err': max_abs_err,
        'mean_rel_err': mean_rel_err,
        'residual_norm': res_norm
    }


def r8vec_bin(data, nbin, a_min=None, a_max=None):
    data = np.asarray(data, dtype=float)
    if a_min is None:
        a_min = np.min(data)
    if a_max is None:
        a_max = np.max(data)
    if a_max <= a_min:
        a_max = a_min + 1.0
    counts = np.zeros(nbin, dtype=int)
    width = (a_max - a_min) / nbin
    idx = np.floor((data - a_min) / width).astype(int)
    idx = np.clip(idx, 0, nbin - 1)
    for i in idx:
        counts[i] += 1
    centers = a_min + (np.arange(nbin) + 0.5) * width
    return centers, counts


def safe_divide(a, b, eps=1e-15):
    return np.where(np.abs(b) > eps, a / b, 0.0)
