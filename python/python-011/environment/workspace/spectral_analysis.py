# -*- coding: utf-8 -*-

import numpy as np


def haar_step_1d(u):
    u = np.asarray(u, dtype=float).flatten()
    n = u.size
    if n < 2:
        return u.copy()
    n_pair = n // 2
    a = (u[0:2 * n_pair:2] + u[1:2 * n_pair:2]) / np.sqrt(2.0)
    d = (u[0:2 * n_pair:2] - u[1:2 * n_pair:2]) / np.sqrt(2.0)
    if n % 2 == 1:
        a = np.append(a, u[-1])
    return np.concatenate([a, d])


def haar_step_1d_inverse(v):
    v = np.asarray(v, dtype=float).flatten()
    n = v.size
    if n < 2:
        return v.copy()



    half = n // 2
    a = v[:half]
    d = v[half:]
    n_pair = min(a.size, d.size)
    u = np.zeros(2 * n_pair, dtype=float)
    u[0::2] = (a[:n_pair] + d[:n_pair]) / np.sqrt(2.0)
    u[1::2] = (a[:n_pair] - d[:n_pair]) / np.sqrt(2.0)
    return u


def haar_1d(u):
    u = np.asarray(u, dtype=float).flatten()
    n = u.size
    if n < 2:
        return u.copy()
    result = u.copy()
    current_len = n
    while current_len >= 2:
        step_res = haar_step_1d(result[:current_len])
        result[:current_len] = step_res
        current_len = (current_len + 1) // 2
    return result


def haar_1d_inverse(u):
    u = np.asarray(u, dtype=float).flatten()
    n = u.size
    if n < 2:
        return u.copy()
    result = u.copy()

    lengths = [n]
    current = n
    while current > 1:
        current = (current + 1) // 2
        lengths.append(current)

    for i in range(len(lengths) - 2, -1, -1):
        L = lengths[i]
        half = lengths[i + 1]
        a = result[:half]
        d = result[half:L]
        n_pair = min(a.size, d.size)
        recon = np.zeros(2 * n_pair, dtype=float)
        recon[0::2] = (a[:n_pair] + d[:n_pair]) / np.sqrt(2.0)
        recon[1::2] = (a[:n_pair] - d[:n_pair]) / np.sqrt(2.0)
        result[:2 * n_pair] = recon
    return result


def haar_step_2d(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    W = np.zeros_like(A)
    for j in range(n):
        W[:, j] = haar_step_1d(A[:, j])

    B = np.zeros_like(W)
    for i in range(m):
        B[i, :] = haar_step_1d(W[i, :])
    return B


def haar_step_2d_inverse(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    W = np.zeros_like(A)
    for i in range(m):
        W[i, :] = haar_step_1d_inverse(A[i, :])
    B = np.zeros_like(W)
    for j in range(n):
        B[:, j] = haar_step_1d_inverse(W[:, j])
    return B


def haar_2d(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m < 2 and n < 2:
        return A.copy()
    result = A.copy()
    cm, cn = m, n
    while cm >= 2 or cn >= 2:
        sub = result[:cm, :cn]
        sub_t = haar_step_2d(sub)
        result[:cm, :cn] = sub_t
        cm = (cm + 1) // 2
        cn = (cn + 1) // 2
    return result


def haar_2d_inverse(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m < 2 and n < 2:
        return A.copy()

    ms = [m]
    ns = [n]
    while ms[-1] > 1 or ns[-1] > 1:
        ms.append((ms[-1] + 1) // 2)
        ns.append((ns[-1] + 1) // 2)
    result = A.copy()
    for idx in range(len(ms) - 2, -1, -1):
        cm, cn = ms[idx], ns[idx]
        sub = result[:cm, :cn]
        sub_t = haar_step_2d_inverse(sub)
        result[:cm, :cn] = sub_t
    return result


def wavelet_energy_spectrum(coeffs_2d):
    A = np.asarray(coeffs_2d, dtype=float)
    m, n = A.shape
    energies = []
    cm, cn = m, n
    while cm >= 2 or cn >= 2:
        hm = (cm + 1) // 2
        hn = (cn + 1) // 2

        hl = A[:hm, hn:cn]
        lh = A[hm:cm, :hn]
        hh = A[hm:cm, hn:cn]
        e = np.sum(hl ** 2) + np.sum(lh ** 2) + np.sum(hh ** 2)
        energies.append(e)
        cm = hm
        cn = hn

    energies.append(np.sum(A[:cm, :cn] ** 2))
    return np.array(energies[::-1])


def analyze_superconducting_fluctuations(order_parameter_field):
    field = np.real(np.asarray(order_parameter_field, dtype=complex))
    coeffs = haar_2d(field)
    spectrum = wavelet_energy_spectrum(coeffs)
    scales = [2 ** i for i in range(len(spectrum))]
    return spectrum, scales
