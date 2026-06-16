# -*- coding: utf-8 -*-
"""
spectral_tools.py
=================

谱分析工具: FFT 与 Hankel 矩阵方法.

来源种子项目:
  - 426_fft_serial     -> 复数 FFT (Petersen-Arbenz 实现)
  - 505_hankel_inverse -> Hankel 矩阵与 Toeplitz 矩阵

物理背景:
  撕裂模不稳定性具有特征频率和增长率, 可以通过对时间信号进行
  谱分析来提取. 常用方法:

  1. FFT 功率谱:
     对磁通量扰动 psi(t) 做 FFT, 找到 dominant 频率:
       psi_hat(omega) = integral psi(t) exp(-i omega t) dt
     增长率 gamma = Re(omega), 振荡频率 omega_r = Im(omega)

  2. Hankel 矩阵方法 (Prony 方法):
     构造 Hankel 矩阵 H_{ij} = s_{i+j}, 其中 s_k 为时间序列.
     通过求解 H 的特征值问题, 可以直接提取阻尼/增长振荡模式:
       s_k = sum_j A_j z_j^k,  z_j = exp((gamma_j + i omega_j) dt)

  3. Toeplitz 矩阵:
     用于自相关函数的谱估计 (Wiener-Khinchin 定理).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------
#  FFT 工具 (参考 426_fft_serial)
# -------------------------------------------------------------------
def cffti(n: int) -> np.ndarray:
    """
    初始化 FFT 所需的正弦/余弦表.

    返回:
        w: 长度 2n 的数组, 存储 cos 和 sin 值
    """
    n2 = n // 2
    w = np.zeros(2 * n2)
    for i in range(n2):
        arg = 2.0 * np.pi * i / n
        w[2 * i] = np.cos(arg)
        w[2 * i + 1] = np.sin(arg)
    return w


def complex_fft(x: np.ndarray, sgn: int = 1) -> np.ndarray:
    """
    复数 FFT (Cooley-Tukey 基 2 算法).

    参数:
        x: 复数输入数组 (长度为 2 的幂)
        sgn: +1 表示正变换, -1 表示逆变换

    返回:
        复数 FFT 结果
    """
    n = x.size
    if n == 0:
        return x
    if n & (n - 1):
        raise ValueError(f"FFT 长度必须是 2 的幂, 当前 n = {n}")
    if n == 1:
        return x.copy()

    # 分离偶数和奇数
    even = complex_fft(x[0::2], sgn)
    odd = complex_fft(x[1::2], sgn)

    # 蝶形运算
    k = np.arange(n // 2)
    twiddle = np.exp(sgn * 2j * np.pi * k / n)
    result = np.zeros(n, dtype=complex)
    result[:n // 2] = even + twiddle * odd
    result[n // 2:] = even - twiddle * odd
    return result


def power_spectrum(signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算时间信号的功率谱密度.

    参数:
        signal: 实数时间序列
        dt: 采样间隔

    返回:
        (freq, psd): 频率数组和功率谱密度
    """
    n = signal.size
    # 补零到 2 的幂
    n_pad = 1
    while n_pad < n:
        n_pad *= 2
    signal_pad = np.zeros(n_pad)
    signal_pad[:n] = signal

    # FFT
    signal_complex = signal_pad.astype(complex)
    fft_result = complex_fft(signal_complex, sgn=-1)
    fft_result = fft_result[:n_pad // 2]

    # 功率谱
    psd = np.abs(fft_result) ** 2 / n
    freq = np.arange(n_pad // 2) / (n_pad * dt)

    return freq, psd


def find_dominant_frequency(
    freq: np.ndarray,
    psd: np.ndarray,
    n_peaks: int = 3,
) -> np.ndarray:
    """
    找到功率谱中最强的 n_peaks 个频率.

    返回:
        频率数组 (按功率降序)
    """
    if n_peaks > freq.size:
        n_peaks = freq.size
    idx_sorted = np.argsort(psd)[::-1]
    return freq[idx_sorted[:n_peaks]]


# -------------------------------------------------------------------
#  Hankel 矩阵构造与求逆 (来自 505_hankel_inverse)
# -------------------------------------------------------------------
def hankel_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """
    构造 Hankel 矩阵.

    Hankel 矩阵 H 的元素 H_{ij} = x_{i+j-2},
    其中 x 长度为 2n-1.

    例: x = [1, 2, 3, 4, 5], n = 3
        H = [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5]]
    """
    if x.size != 2 * n - 1:
        raise ValueError(f"x 长度必须为 2n-1 = {2*n-1}, 当前为 {x.size}")
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = x[i + j]
    return H


def toeplitz_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """
    构造 Toeplitz 矩阵.

    Toeplitz 矩阵 T 的元素 T_{ij} = x_{|i-j|}.
    """
    if x.size != 2 * n - 1:
        raise ValueError(f"x 长度必须为 2n-1 = {2*n-1}, 当前为 {x.size}")
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = x[n - 1 + i - j]
    return T


def hankel_inverse(n: int, x: np.ndarray) -> np.ndarray:
    """
    Hankel 矩阵求逆 (来自 505_hankel_inverse).

    使用 Fiedler 公式:
        H^{-1} = M1 * M2 - M3 * M4
    其中 M1, M2, M3, M4 由 Hankel 和 Toeplitz 矩阵构成.
    """
    A = hankel_matrix(n, x)
    # 求解两个线性系统
    p = np.zeros(2 * n - 1)
    p[n:] = x[n:]
    p = np.append(x[n:], 0.0)
    u = np.linalg.solve(A, p)

    q = np.zeros(n)
    q[n - 1] = 1.0
    v = np.linalg.solve(A, q)

    # 构造四个矩阵
    w1 = np.append(v[1:], 0.0)
    M1 = hankel_matrix(n, w1)

    w2 = np.append(np.zeros(n - 1), u)
    M2 = toeplitz_matrix(n, w2)

    w3 = np.append(u[1:], -1.0)
    M3 = hankel_matrix(n, w3)

    w4 = np.append(np.zeros(n - 1), v)
    M4 = toeplitz_matrix(n, w4)

    return M1 @ M2 - M3 @ M4


# -------------------------------------------------------------------
#  Prony 方法: 从 Hankel 矩阵提取振荡模式
# -------------------------------------------------------------------
def prony_analysis(
    signal: np.ndarray,
    n_modes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prony 方法 (Hankel SVD) 提取阻尼/增长振荡模式.

    模型: s_k = sum_j A_j z_j^k,  z_j = exp((gamma_j + i omega_j) dt)

    算法:
        1. 构造 Hankel 矩阵 H_{ij} = s_{i+j}
        2. SVD 分解 H = U Sigma V^T
        3. 截取前 n_modes 个奇异值
        4. 构造移位矩阵并求解广义特征值问题

    参数:
        signal: 时间序列 (长度 N)
        n_modes: 提取的模式数

    返回:
        (gamma, omega, amplitude):
            gamma: 增长率数组 (n_modes,)
            omega: 角频率数组 (n_modes,)
            amplitude: 振幅数组 (n_modes,)
    """
    N = signal.size
    if n_modes > N // 2:
        n_modes = max(1, N // 2)

    # 构造 Hankel 矩阵
    n_row = N - n_modes
    n_col = n_modes + 1
    if n_row < n_col:
        n_row = n_col
    H = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            if i + j < N:
                H[i, j] = signal[i + j]

    # SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    # 截取前 n_modes 个奇异值
    U_r = U[:, :n_modes]
    S_r = S[:n_modes]
    V_r = Vt[:n_modes, :]

    # 构造移位矩阵
    H1 = H[:, :-1]
    H2 = H[:, 1:]
    # 最小二乘: H2 ≈ H1 * A
    A_shift = np.linalg.lstsq(H1, H2, rcond=None)[0]

    # 特征值
    eigenvalues = np.linalg.eigvals(A_shift)

    # 提取增长率和频率 (假设 dt = 1)
    gamma = np.real(np.log(eigenvalues + 1e-30))
    omega = np.imag(np.log(eigenvalues + 1e-30))

    # 振幅 (从 V 矩阵)
    amplitude = np.abs(V_r[0, :n_modes])

    return gamma, omega, amplitude


# -------------------------------------------------------------------
#  自相关函数与 Toeplitz 谱估计
# -------------------------------------------------------------------
def autocorrelation(signal: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    计算时间信号的自相关函数.

    R(k) = (1/N) sum_{n=0}^{N-1-k} s_n s_{n+k}

    自相关矩阵是 Toeplitz 矩阵.
    """
    N = signal.size
    if max_lag is None:
        max_lag = N // 2
    max_lag = min(max_lag, N - 1)

    R = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        R[k] = np.sum(signal[:N - k] * signal[k:]) / N
    return R


def yule_walker_psd(
    signal: np.ndarray,
    order: int = 16,
    n_freq: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yule-Walker AR 模型谱估计 (基于 Toeplitz 系统).

    求解 Toeplitz 系统: R a = r, 其中 R 是自相关矩阵.
    """
    N = signal.size
    R = autocorrelation(signal, max_lag=order)

    # 构造 Toeplitz 矩阵
    R_mat = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            R_mat[i, j] = R[abs(i - j)]

    # 求解
    r_vec = R[1:order + 1]
    try:
        a = np.linalg.solve(R_mat, r_vec)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R_mat, r_vec, rcond=None)[0]

    # 计算 PSD
    freq = np.linspace(0, np.pi, n_freq)
    psd = np.zeros(n_freq)
    for k, omega in enumerate(freq):
        A_omega = 1.0 - np.sum(a * np.exp(-1j * omega * np.arange(1, order + 1)))
        psd[k] = R[0] / np.abs(A_omega) ** 2

    return freq, psd
