#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matrix_solver.py
特殊结构矩阵求解与排序模块

融合原项目:
- 131_c8lib: 复数矩阵运算与线性方程组求解
- 993_r8row: 矩阵行快速排序与分区
- 1003_r8utt: 上三角Toeplitz矩阵快速求解

在BSM信号分析中用于:
- 复数散射振幅矩阵的LU分解与求逆
- 事例数据矩阵的行排序（按能量/动量）
- 探测器响应Toeplitz矩阵的快速反卷积
"""

import numpy as np
from typing import Tuple, Optional


def c8mat_fss(n: int, a: np.ndarray, nb: int, b: np.ndarray) -> np.ndarray:
    """
    复数矩阵因子分解并求解多右端项线性系统 A X = B。

    采用部分主元高斯消去法（无需显式存储 pivot 向量）。

    算法步骤:
        1. 对 j = 1, ..., N:
           a) 在第 j 列中从第 j 行开始寻找模最大元 a(ipiv, jcol)
           b) 若 pivot = 0, 系统奇异，返回错误
           c) 交换第 jcol 行与第 ipiv 行（同时交换 B 的对应行）
           d) 归一化 pivot 行
           e) 消去第 jcol 列中下方所有元素
        2. 回代求解 X

    Parameters
    ----------
    n : int
        矩阵阶数
    a : np.ndarray
        复数系数矩阵 A(N, N)，会被覆盖为消去后的上三角形式
    nb : int
        右端项个数
    b : np.ndarray
        右端项矩阵 B(N, NB)

    Returns
    -------
    np.ndarray
        解矩阵 X(N, NB)

    Raises
    ------
    ValueError
        若遇到零 pivot
    """
    if n < 1:
        raise ValueError("矩阵阶数 N 必须 >= 1")
    if a.shape != (n, n):
        raise ValueError(f"A 的形状 {a.shape} 与 N={n} 不匹配")
    if b.shape[0] != n:
        raise ValueError(f"B 的行数 {b.shape[0]} 与 N={n} 不匹配")

    # 工作副本
    a_work = np.array(a, dtype=complex, copy=True)
    b_work = np.array(b, dtype=complex, copy=True)

    for jcol in range(n):
        # 部分主元搜索
        piv = abs(a_work[jcol, jcol])
        ipiv = jcol
        for i in range(jcol + 1, n):
            ai = abs(a_work[i, jcol])
            if ai > piv:
                piv = ai
                ipiv = i

        if piv < 1e-15:
            raise ValueError(f"C8MAT_FSS: 在第 {jcol} 步遇到零 pivot")

        # 行交换
        if ipiv != jcol:
            a_work[[jcol, ipiv], :] = a_work[[ipiv, jcol], :]
            b_work[[jcol, ipiv], :] = b_work[[ipiv, jcol], :]

        # 归一化 pivot 行
        temp = a_work[jcol, jcol]
        a_work[jcol, jcol] = 1.0 + 0.0j
        if jcol + 1 < n:
            a_work[jcol, jcol + 1:] /= temp
        b_work[jcol, :] /= temp

        # 消去下方元素
        for i in range(jcol + 1, n):
            if abs(a_work[i, jcol]) > 1e-18:
                temp = -a_work[i, jcol]
                a_work[i, jcol] = 0.0 + 0.0j
                if jcol + 1 < n:
                    a_work[i, jcol + 1:] += temp * a_work[jcol, jcol + 1:]
                b_work[i, :] += temp * b_work[jcol, :]

    # 回代
    for j in range(nb):
        for jcol in range(n - 1, 0, -1):
            b_work[0:jcol, j] -= a_work[0:jcol, jcol] * b_work[jcol, j]

    return b_work


def r8row_part_quick_a(m: int, n: int, a: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    R8ROW 快速分区排序（QuickSort partition）。

    以 A(0, 0:N) 为 pivot key，将所有行分为三部分：
        [0 : l]     : 小于 key 的行
        [l : r]     : 等于 key 的行
        [r : m]     : 大于 key 的行

    这是快速排序算法的核心子程序，用于对事例数据按物理量排序。

    Parameters
    ----------
    m, n : int
        矩阵维度 (m 行, n 列)
    a : np.ndarray
        实数矩阵，形状 (M, N)

    Returns
    -------
    a : np.ndarray
        重排后的矩阵
    l : int
        左边界索引（小于 pivot 的行数）
    r : int
        右边界索引（大于 pivot 的起始行号）
    """
    if m < 1:
        raise ValueError("M < 1")
    if m == 1:
        return a, 0, 2

    key = a[0, :].copy()

    # 使用字典序比较
    def row_cmp(row, key_row):
        for col in range(n):
            if row[col] > key_row[col]:
                return 1   # greater
            elif row[col] < key_row[col]:
                return -1  # less
        return 0  # equal

    # 三向分区 (Dutch National Flag)
    lt_end = 0      # 小于 key 的区域的末尾（开区间）
    gt_start = m    # 大于 key 的区域的起始（闭区间）
    i = 1

    while i < gt_start:
        cmp = row_cmp(a[i, :], key)
        if cmp > 0:  # greater: 放到右边
            gt_start -= 1
            tmp = a[gt_start, :].copy()
            a[gt_start, :] = a[i, :]
            a[i, :] = tmp
            # i 不递增，因为新换入的行还未检查
        elif cmp < 0:  # less: 放到左边
            lt_end += 1
            tmp = a[lt_end, :].copy()
            a[lt_end, :] = a[i, :]
            a[i, :] = tmp
            i += 1
        else:  # equal
            i += 1

    # 将 key (第0行) 移到等于 key 的区域中间
    # 等于 key 的区域是 [lt_end+1 : gt_start]
    # 将第0行与 lt_end 行交换
    if lt_end >= 0:
        tmp = a[lt_end, :].copy()
        a[lt_end, :] = a[0, :]
        a[0, :] = tmp

    return a, lt_end, gt_start


def r8row_sort_quick_a(m: int, n: int, a: np.ndarray) -> np.ndarray:
    """
    R8ROW 快速排序完整实现。

    递归地对矩阵行按字典序排序（首列优先，次列次之）。
    用于将探测器击中事件按能量、动量等物理量排序。

    Parameters
    ----------
    m, n : int
        矩阵维度
    a : np.ndarray
        待排序矩阵

    Returns
    -------
    np.ndarray
        排序后的矩阵
    """
    if m <= 1:
        return a

    a, l, r = r8row_part_quick_a(m, n, a)

    if 1 < l:
        a[0:l, :] = r8row_sort_quick_a(l, n, a[0:l, :])
    if r < m:
        a[r:m, :] = r8row_sort_quick_a(m - r, n, a[r:m, :])

    return a


def r8utt_sl(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解上三角 Toeplitz 线性系统 A x = b。

    R8UTT 存储格式: 上三角 Toeplitz 矩阵仅由第一行决定：
        A = [ a0  a1  a2  ...  a_{N-1}
              0   a0  a1  ...  a_{N-2}
              :   :   :  ...    :
              0   0   0  ...    a0   ]

    利用 Toeplitz 结构，无需显式存储完整矩阵，
    求解复杂度为 O(N^2) 而非一般高斯消去的 O(N^3)。

    算法:
        x(j) = b(j) / a0
        for i = 0, ..., j-1:
            b(i) -= a(j-i) * x(j)

    Parameters
    ----------
    n : int
        矩阵阶数
    a : np.ndarray
        Toeplitz 第一行，长度 n，a[0] 为对角元
    b : np.ndarray
        右端项，长度 n

    Returns
    -------
    np.ndarray
        解向量 x，长度 n
    """
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
    """
    批量求解多个具有相同 Toeplitz 结构的右端项。

    在探测器反卷积中，所有像素共享相同的点扩散函数（PSF），
    其离散化后形成相同的 Toeplitz 矩阵。

    Parameters
    ----------
    a_row : np.ndarray
        Toeplitz 第一行
    b_matrix : np.ndarray
        右端项矩阵，形状 (N, K)

    Returns
    -------
    np.ndarray
        解矩阵，形状 (N, K)
    """
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
    """
    使用 Toeplitz 矩阵快速求解进行探测器响应反卷积。

    探测器观测信号为真实信号与点扩散函数（PSF）的卷积：
        y_{obs} = K * y_{true} + noise

    其中 K 是 Toeplitz 矩阵（由 PSF 离散化得到）。
    通过 Tikhonov 正则化求解：
        (K^T K + λ I) y_{true} = K^T y_{obs}

    为利用快速 Toeplitz 求解器，对 K^T K + λI 做近似处理。

    Parameters
    ----------
    observed : np.ndarray
        观测信号
    psf : np.ndarray
        点扩散函数（归一化）
    regularization : float
        Tikhonov 正则化参数 λ

    Returns
    -------
    np.ndarray
        反卷积后的信号估计
    """
    n = observed.size
    # 构建 K^T K + λI 的第一行（近似 Toeplitz）
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

    # 计算 K^T y
    kty = np.zeros(n)
    for i in range(n):
        for j in range(n):
            kj = psf[j] if j < psf.size else 0.0
            if abs(i - j) < psf.size:
                kty[i] += kj * observed[j]

    return r8utt_sl(n, ktk_row, kty)
