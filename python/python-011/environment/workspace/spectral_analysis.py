# -*- coding: utf-8 -*-
"""
spectral_analysis.py
--------------------
基于 Haar 小波变换的多分辨率谱分析模块。

对应种子项目：
  - 496_haar_transform：Haar 小波变换（1D/2D、正交归一）
  - 607_jacobi_polynomial：Jacobi 多项式用于谱展开

物理背景：
  高温超导体中，电荷密度波 (CDW) 和配对密度波 (PDW)
  在空间上呈现多尺度涨落特征。Haar 小波提供了一种
  计算高效的多分辨率分解工具，可将实空间或动量空间的
  序参量场分解为不同尺度的成分：
      Δ(x) = Σ_{j,l} d_{j,l} ψ_{j,l}(x) + 近似项
  其中 ψ_{j,l} 为 Haar 小波基函数。

核心公式：
  - 单步 Haar 变换：
      a = (u_0 + u_1) / sqrt(2)
      d = (u_0 - u_1) / sqrt(2)
  - 逆变换：
      u_0 = (a + d) / sqrt(2)
      u_1 = (a - d) / sqrt(2)
"""

import numpy as np


def haar_step_1d(u):
    """
    对一维向量做单步 Haar 变换。

    将 u 分解为近似系数 a（前半）和细节系数 d（后半）。
    若长度为奇数，最后一个元素保留在 a 末尾。
    """
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
    """
    单步 Haar 逆变换。
    要求 v 的长度为偶数（或由 haar_step_1d 产生）。
    """
    v = np.asarray(v, dtype=float).flatten()
    n = v.size
    if n < 2:
        return v.copy()
    # 判断 a 的长度：若原长度为奇数，a 比 d 多一个
    # 这里假设输入来自 haar_step_1d
    # 简化：假设 n 为偶数
    half = n // 2
    a = v[:half]
    d = v[half:]
    n_pair = min(a.size, d.size)
    u = np.zeros(2 * n_pair, dtype=float)
    u[0::2] = (a[:n_pair] + d[:n_pair]) / np.sqrt(2.0)
    u[1::2] = (a[:n_pair] - d[:n_pair]) / np.sqrt(2.0)
    return u


def haar_1d(u):
    """
    完全一维 Haar 变换。
    递归对近似部分做分解直到长度为 1。
    """
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
    """
    完全一维 Haar 逆变换。
    要求 u 的长度为 2 的幂次（或至少可重构）。
    """
    u = np.asarray(u, dtype=float).flatten()
    n = u.size
    if n < 2:
        return u.copy()
    result = u.copy()
    # 重构长度序列：找到各层近似长度
    lengths = [n]
    current = n
    while current > 1:
        current = (current + 1) // 2
        lengths.append(current)
    # 逆序重构
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
    """
    对二维矩阵做单步 Haar 变换（先列后行）。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    # 对每列做 1D 变换
    W = np.zeros_like(A)
    for j in range(n):
        W[:, j] = haar_step_1d(A[:, j])
    # 对每行做 1D 变换
    B = np.zeros_like(W)
    for i in range(m):
        B[i, :] = haar_step_1d(W[i, :])
    return B


def haar_step_2d_inverse(A):
    """
    二维单步 Haar 逆变换（先行逆后列逆）。
    """
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
    """
    完全二维 Haar 变换。
    递归对左上近似子矩阵做分解。
    """
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
    """
    完全二维 Haar 逆变换。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m < 2 and n < 2:
        return A.copy()
    # 收集各级尺寸
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
    """
    计算二维 Haar 小波系数的能量谱（各尺度能量）。

    对于 2D Haar，分解后矩阵分为：
      - LL：近似（最低频）
      - HL, LH, HH：水平、垂直、对角细节
    能量谱定义为各层细节系数的 L2 范数。
    """
    A = np.asarray(coeffs_2d, dtype=float)
    m, n = A.shape
    energies = []
    cm, cn = m, n
    while cm >= 2 or cn >= 2:
        hm = (cm + 1) // 2
        hn = (cn + 1) // 2
        # 细节块：HL, LH, HH
        hl = A[:hm, hn:cn]
        lh = A[hm:cm, :hn]
        hh = A[hm:cm, hn:cn]
        e = np.sum(hl ** 2) + np.sum(lh ** 2) + np.sum(hh ** 2)
        energies.append(e)
        cm = hm
        cn = hn
    # 近似块能量
    energies.append(np.sum(A[:cm, :cn] ** 2))
    return np.array(energies[::-1])  # 从低频到高频


def analyze_superconducting_fluctuations(order_parameter_field):
    """
    对实空间超导序参量场 Δ(r) 做 Haar 小波多分辨率分析，
    返回各尺度能量占比，用于识别 CDW/PDW 的主导波长。

    Parameters
    ----------
    order_parameter_field : ndarray, shape (Nx, Ny)
        实空间序参量复数场（取模或实部）。

    Returns
    -------
    spectrum : ndarray
        各尺度能量。
    scales : list of float
        对应的空间尺度（以格点数为单位）。
    """
    field = np.real(np.asarray(order_parameter_field, dtype=complex))
    coeffs = haar_2d(field)
    spectrum = wavelet_energy_spectrum(coeffs)
    scales = [2 ** i for i in range(len(spectrum))]
    return spectrum, scales
