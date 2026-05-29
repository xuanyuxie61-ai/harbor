"""
spectral_analysis.py

基于 911_prime_factors 核心算法的谱分析与 FFT 优化模块。

原项目 prime_factors 实现了整数的质因数分解。

在本气候归因框架中，质因数分解用于：
1. 优化 FFT 长度（Cooley-Tukey 算法要求长度为 2 的幂次，但混合基 FFT
   可利用质因数分解实现任意合数长度的快速变换）
2. 气候时间序列的周期分析（通过质因数分解识别主导周期）
3. 空间滤波器设计中的频率网格优化

核心公式：
- 质因数分解：N = p_1^{e_1} * p_2^{e_2} * ... * p_k^{e_k}
- 离散傅里叶变换（DFT）：
    X_k = Σ_{n=0}^{N-1} x_n * exp(-2πi * k * n / N)
- 功率谱密度：
    S_k = |X_k|^2 / N
- 通过质因数分解的混合基 FFT 复杂度：
    O(N * Σ e_i * p_i) 对于 N = Π p_i^{e_i}
"""

import numpy as np


def prime_factors(n):
    """
    整数的质因数分解（基于 911_prime_factors）。

    Parameters
    ----------
    n : int
        待分解的整数（>=1）。

    Returns
    -------
    factors : list
        质因数列表（重复出现）。
    """
    if n != int(n):
        raise ValueError("输入必须是整数")
    n = int(n)
    if n < 1:
        raise ValueError("输入必须 >= 1")

    i = 2
    factors = []
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n = n // i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def optimal_fft_length(min_length, max_search=1000):
    """
    寻找不小于 min_length 的最优 FFT 长度。

    最优标准：质因数仅包含 2, 3, 5（混合基 FFT 高度优化）。
    """
    for length in range(min_length, min_length + max_search):
        factors = prime_factors(length)
        if all(f in (2, 3, 5) for f in factors):
            return length
    return min_length


def mixed_radix_fft_complexity(n):
    """
    基于质因数分解的混合基 FFT 复杂度估计。

    复杂度 ≈ N * Σ e_i * p_i
    其中 n = Π p_i^{e_i}
    """
    factors = prime_factors(n)
    if not factors:
        return 0
    from collections import Counter
    cnt = Counter(factors)
    complexity = n * sum(e * p for p, e in cnt.items())
    return complexity


def power_spectrum_1d(signal, sample_rate=1.0):
    """
    计算一维信号的功率谱。

    Parameters
    ----------
    signal : ndarray
        输入信号。
    sample_rate : float
        采样率。

    Returns
    -------
    freqs : ndarray
        频率轴。
    power : ndarray
        功率谱密度。
    """
    n = len(signal)
    n_fft = optimal_fft_length(n)
    # 零填充到最优长度
    padded = np.zeros(n_fft)
    padded[:n] = signal
    fft_vals = np.fft.rfft(padded)
    power = np.abs(fft_vals) ** 2 / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, power


def dominant_periods(signal, sample_rate=1.0, n_peaks=3):
    """
    识别信号中的主导周期。

    通过质因数分解分析 FFT 长度与信号周期的关系。
    """
    freqs, power = power_spectrum_1d(signal, sample_rate)
    # 排除直流分量
    power[0] = 0.0
    peak_indices = np.argsort(power)[-n_peaks:][::-1]
    periods = []
    for idx in peak_indices:
        if freqs[idx] > 1e-14:
            periods.append(1.0 / freqs[idx])
    return periods, freqs, power


def spectral_coherence(series1, series2, sample_rate=1.0):
    """
    计算两个气候时间序列的谱相干性。

    公式：
        C_{xy}(f) = |S_{xy}(f)|^2 / (S_{xx}(f) * S_{yy}(f))
    其中 S_{xy} 为互谱密度。
    """
    n = max(len(series1), len(series2))
    n_fft = optimal_fft_length(n)
    s1 = np.zeros(n_fft)
    s2 = np.zeros(n_fft)
    s1[:len(series1)] = series1
    s2[:len(series2)] = series2

    f1 = np.fft.rfft(s1)
    f2 = np.fft.rfft(s2)

    sxx = np.abs(f1) ** 2
    syy = np.abs(f2) ** 2
    sxy = f1 * np.conj(f2)

    coherence = np.abs(sxy) ** 2 / (sxx * syy + 1e-14)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, np.real(coherence)


def test_spectral():
    # 测试质因数分解
    assert prime_factors(75) == [3, 5, 5]
    # 测试 FFT 优化长度
    opt = optimal_fft_length(100)
    factors = prime_factors(opt)
    assert all(f in (2, 3, 5) for f in factors)
    # 测试功率谱
    t = np.linspace(0, 10, 256)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)
    freqs, power = power_spectrum_1d(signal)
    assert len(freqs) == len(power)
    print("spectral_analysis 自测试通过")


if __name__ == "__main__":
    test_spectral()
