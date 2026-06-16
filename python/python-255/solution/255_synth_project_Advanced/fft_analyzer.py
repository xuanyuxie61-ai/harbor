# -*- coding: utf-8 -*-
"""
fft_analyzer.py
======================================================================
光谱数据的快速 Fourier 变换分析

物理背景:
    系外行星透射光谱中的周期性特征可能暗示:
    (1) 大气中的重力波 (gravity waves)
    (2) 云层的准周期结构
    (3) 行星自转调制
    (4) 轨道相位变化的谐波成分

    通过 FFT 将光谱从波长域变换到频率域 (空间频率),
    可以识别这些隐藏周期信号。

    本模块实现 Cooley-Tukey FFT (移植自 426_fft_serial),
    并计算功率谱密度、自相关函数。

数学公式:
    离散 Fourier 变换:
        X_k = sum_{n=0}^{N-1} x_n exp(-2 pi i k n / N)      (1)
    逆变换:
        x_n = (1/N) sum_{k=0}^{N-1} X_k exp(2 pi i k n / N) (2)
    功率谱密度:
        P_k = |X_k|^2 / N                                     (3)

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict


# ============================================================
# Cooley-Tukey FFT (移植自 426_fft_serial)
# ============================================================
def _fft_recursive(x_real: np.ndarray, x_imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    递归 Cooley-Tukey FFT (仅支持 2 的幂次长度)。
    对应 426_fft_serial 中的核心循环。

    复数存储为 (实部, 虚部) 两个实数组。
    """
    n = len(x_real)
    if n <= 1:
        return x_real.copy(), x_imag.copy()

    if n % 2 != 0:
        raise ValueError(f"FFT 长度必须为 2 的幂, 当前 n={n}")

    even_real, even_imag = _fft_recursive(x_real[0::2], x_imag[0::2])
    odd_real, odd_imag = _fft_recursive(x_real[1::2], x_imag[1::2])

    T_real = np.zeros(n // 2, dtype=np.float64)
    T_imag = np.zeros(n // 2, dtype=np.float64)

    for k in range(n // 2):
        angle = -2.0 * np.pi * k / n
        w_real = np.cos(angle)
        w_imag = np.sin(angle)
        T_real[k] = w_real * odd_real[k] - w_imag * odd_imag[k]
        T_imag[k] = w_real * odd_imag[k] + w_imag * odd_real[k]

    y_real = np.zeros(n, dtype=np.float64)
    y_imag = np.zeros(n, dtype=np.float64)
    y_real[: n // 2] = even_real + T_real
    y_imag[: n // 2] = even_imag + T_imag
    y_real[n // 2 :] = even_real - T_real
    y_imag[n // 2 :] = even_imag - T_imag

    return y_real, y_imag


def fft_forward(data: np.ndarray) -> np.ndarray:
    """
    正向 FFT。
    输入: 实数数组 (自动补零到 2 的幂次)
    输出: 复数数组 (频域)
    """
    n_orig = len(data)
    n = 1
    while n < n_orig:
        n = n * 2
    x = np.zeros(n, dtype=np.float64)
    x[:n_orig] = data

    x_real, x_imag = _fft_recursive(x, np.zeros(n))
    return x_real + 1j * x_imag


def fft_inverse(X: np.ndarray) -> np.ndarray:
    """
    逆向 FFT。
    输入: 复数频域数组
    输出: 实数时域数组
    """
    n = len(X)
    X_conj = np.conj(X)
    x_real = np.real(X_conj)
    x_imag = np.imag(X_conj)
    y_real, y_imag = _fft_recursive(x_real, x_imag)
    return (y_real + 1j * y_imag).real / n


def spectral_power_density(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算功率谱密度。
    P_k = |FFT(x)|^2 / N

    返回频率轴和功率谱。
    """
    n_orig = len(data)
    n = 1
    while n < n_orig:
        n = n * 2

    data_mean = np.mean(data)
    data_centered = data - data_mean

    X = fft_forward(data_centered)
    psd = np.abs(X[: n // 2]) ** 2 / n
    freqs = np.arange(n // 2) / n
    return freqs, psd


def autocorrelation(data: np.ndarray) -> np.ndarray:
    """
    通过 FFT 计算自相关函数。
    R(tau) = IFFT( |FFT(x)|^2 )
    """
    X = fft_forward(data)
    S = np.abs(X) ** 2
    R = fft_inverse(S)
    return R[: len(data)] / max(np.abs(R[0]), 1.0e-30)


def detect_periodicities(
    wavelength_grid: np.ndarray,
    spectrum: np.ndarray,
    threshold_factor: float = 3.0,
) -> Dict:
    """
    检测光谱中的周期性成分。

    物理应用: 识别重力波、云层周期结构等。
    """
    freqs, psd = spectral_power_density(spectrum)
    psd = np.maximum(psd, 1.0e-30)
    psd_db = 10.0 * np.log10(psd)

    threshold = np.mean(psd_db) + threshold_factor * np.std(psd_db)
    peak_mask = psd_db > threshold
    peak_indices = np.where(peak_mask)[0]

    dwl = wavelength_grid[1] - wavelength_grid[0] if len(wavelength_grid) > 1 else 1.0
    peak_periods = []
    for idx in peak_indices:
        if freqs[idx] > 1.0e-10:
            period = 1.0 / (freqs[idx] / max(dwl, 1.0e-30))
            peak_periods.append(period)

    acf = autocorrelation(spectrum)

    return {
        "frequencies": freqs,
        "psd": psd,
        "psd_db": psd_db,
        "peak_indices": peak_indices,
        "peak_periods": peak_periods,
        "autocorrelation": acf,
        "n_peaks": len(peak_indices),
    }
