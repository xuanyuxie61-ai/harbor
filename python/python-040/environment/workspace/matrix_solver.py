#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def c8mat_fss(n: int, a: np.ndarray, nb: int, b: np.ndarray) -> np.ndarray:
    if n < 1:
        raise ValueError("矩阵阶数 N 必须 >= 1")
    if a.shape != (n, n):
        raise ValueError(f"A 的形状 {a.shape} 与 N={n} 不匹配")
    if b.shape[0] != n:
        raise ValueError(f"B 的行数 {b.shape[0]} 与 N={n} 不匹配")


    a_work = np.array(a, dtype=complex, copy=True)
    b_work = np.array(b, dtype=complex, copy=True)

    for jcol in range(n):

        piv = abs(a_work[jcol, jcol])
        ipiv = jcol
        for i in range(jcol + 1, n):
            ai = abs(a_work[i, jcol])
            if ai > piv:
                piv = ai
                ipiv = i

        if piv < 1e-15:
            raise ValueError(f"C8MAT_FSS: 在第 {jcol} 步遇到零 pivot")


        if ipiv != jcol:
            a_work[[jcol, ipiv], :] = a_work[[ipiv, jcol], :]
            b_work[[jcol, ipiv], :] = b_work[[ipiv, jcol], :]


        temp = a_work[jcol, jcol]
        a_work[jcol, jcol] = 1.0 + 0.0j
        if jcol + 1 < n:
            a_work[jcol, jcol + 1:] /= temp
        b_work[jcol, :] /= temp


        for i in range(jcol + 1, n):
            if abs(a_work[i, jcol]) > 1e-18:
                temp = -a_work[i, jcol]
                a_work[i, jcol] = 0.0 + 0.0j
                if jcol + 1 < n:
                    a_work[i, jcol + 1:] += temp * a_work[jcol, jcol + 1:]
                b_work[i, :] += temp * b_work[jcol, :]


    for j in range(nb):
        for jcol in range(n - 1, 0, -1):
            b_work[0:jcol, j] -= a_work[0:jcol, jcol] * b_work[jcol, j]

    return b_work


def r8row_part_quick_a(m: int, n: int, a: np.ndarray) -> Tuple[np.ndarray, int, int]:
    if m < 1:
        raise ValueError("M < 1")
    if m == 1:
        return a, 0, 2

    key = a[0, :].copy()


    def row_cmp(row, key_row):
        for col in range(n):
            if row[col] > key_row[col]:
                return 1
            elif row[col] < key_row[col]:
                return -1
        return 0


    lt_end = 0
    gt_start = m
    i = 1

    while i < gt_start:
        cmp = row_cmp(a[i, :], key)
        if cmp > 0:
            gt_start -= 1
            tmp = a[gt_start, :].copy()
            a[gt_start, :] = a[i, :]
            a[i, :] = tmp

        elif cmp < 0:
            lt_end += 1
            tmp = a[lt_end, :].copy()
            a[lt_end, :] = a[i, :]
            a[i, :] = tmp
            i += 1
        else:
            i += 1




    if lt_end >= 0:
        tmp = a[lt_end, :].copy()
        a[lt_end, :] = a[0, :]
        a[0, :] = tmp

    return a, lt_end, gt_start


def r8row_sort_quick_a(m: int, n: int, a: np.ndarray) -> np.ndarray:
    if m <= 1:
        return a

    a, l, r = r8row_part_quick_a(m, n, a)

    if 1 < l:
        a[0:l, :] = r8row_sort_quick_a(l, n, a[0:l, :])
    if r < m:
        a[r:m, :] = r8row_sort_quick_a(m - r, n, a[r:m, :])

    return a


def r8utt_sl(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if n < 1:
        raise ValueError("N 必须 >= 1")
    if a.size < n:
        raise ValueError("Toeplitz 向量长度不足")
    if b.size < n:
        raise ValueError("右端项长度不足")
    if abs(a[0]) < 1e-15:
        raise ValueError("对角元 a[0] 接近零，矩阵奇异")

    x = np.array(b[:n], dtype=float, copy=True)

    for j in range(n - 1, -1, -1):
        x[j] /= a[0]
        for i in range(j):
            x[i] -= a[j - i] * x[j]

    return x


def r8utt_solve_batch(a_row: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    n = b_matrix.shape[0]
    k = b_matrix.shape[1] if b_matrix.ndim > 1 else 1
    x = np.zeros((n, k), dtype=float)
    for col in range(k):
        b_vec = b_matrix[:, col] if k > 1 else b_matrix
        x[:, col] = r8utt_sl(n, a_row, b_vec)
    return x


def detector_deconvolution_toeplitz(
    observed: np.ndarray,
    psf: np.ndarray,
    regularization: float = 1e-6
) -> np.ndarray:
    n = observed.size

    ktk_row = np.zeros(n)
    for shift in range(n):
        val = 0.0
        for i in range(n):
            j = i + shift
            if j < n:
                ki = psf[i] if i < psf.size else 0.0
                kj = psf[j] if j < psf.size else 0.0
                val += ki * kj
        ktk_row[shift] = val
    ktk_row[0] += regularization


    kty = np.zeros(n)
    for i in range(n):
        for j in range(n):
            kj = psf[j] if j < psf.size else 0.0
            if abs(i - j) < psf.size:
                kty[i] += kj * observed[j]

    return r8utt_sl(n, ktk_row, kty)
