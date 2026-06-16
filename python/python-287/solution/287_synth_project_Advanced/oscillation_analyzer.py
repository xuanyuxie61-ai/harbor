# -*- coding: utf-8 -*-
"""
oscillation_analyzer.py
=======================

磁振荡相位-频率分析模块.

来源种子项目:
  - 1188_joaqgonzar_Gamma_Oscillations_PCx -> 神经信号相位-振幅耦合分析

物理背景:
  撕裂模非线性演化过程中, 磁岛合并会产生准周期振荡 (QPO).
  类似于神经科学中的 gamma 振荡, 我们可以分析:

  1. 相位-振幅耦合 (PAC):
     低频磁振荡的相位是否调制高频扰动的振幅?
     这可以揭示磁岛合并过程中不同尺度模式之间的非线性耦合.

  2. 瞬时频率和增长率:
     使用 Hilbert 变换提取解析信号:
       z(t) = ψ(t) + i H[ψ](t) = A(t) exp(i φ(t))
     瞬时增长率: γ(t) = d/dt ln A(t)
     瞬时频率: ω(t) = dφ/dt

  3. 调制指数 (Modulation Index):
     MI = (log(N) - H(p)) / log(N)
     其中 H(p) 为相位分布的熵, N 为相位 bin 数.

  4. Granger 因果:
     检验 B_x 和 B_y 扰动之间是否存在因果驱动关系.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict


# -------------------------------------------------------------------
#  Hilbert 变换
# -------------------------------------------------------------------
def hilbert_transform(signal: np.ndarray) -> np.ndarray:
    """
    计算实数信号的 Hilbert 变换.

    H[s](t) = (1/π) P ∫ s(τ)/(t-τ) dτ

    在频域: H[s](ω) = -i sgn(ω) s(ω)

    返回:
        解析信号 z(t) = s(t) + i H[s](t)
    """
    n = signal.size
    # FFT
    S = np.fft.fft(signal)
    # 构造单边谱
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(S * h)
    return analytic


def instantaneous_amplitude_phase(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从解析信号提取瞬时振幅和相位.

    z(t) = A(t) exp(i φ(t))
    A(t) = |z(t)|,  φ(t) = arg(z(t))

    返回:
        (amplitude, phase)
    """
    z = hilbert_transform(signal)
    amplitude = np.abs(z)
    phase = np.angle(z)
    return amplitude, phase


def instantaneous_frequency(
    phase: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    从相位计算瞬时频率: ω(t) = dφ/dt.

    需要处理相位卷绕 (phase wrapping).
    """
    # 解卷绕
    phase_unwrapped = np.unwrap(phase)
    omega = np.gradient(phase_unwrapped, dt)
    return omega


def instantaneous_growth_rate(
    amplitude: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    从振幅计算瞬时增长率: γ(t) = d/dt ln A(t).
    """
    amp_safe = np.maximum(amplitude, 1.0e-30)
    log_amp = np.log(amp_safe)
    gamma = np.gradient(log_amp, dt)
    return gamma


# -------------------------------------------------------------------
#  相位-振幅耦合 (PAC) 分析
# -------------------------------------------------------------------
def modulation_index(
    amplitude: np.ndarray,
    phase_slow: np.ndarray,
    num_bins: int = 18,
) -> float:
    """
    调制指数 (Modulation Index, MI).

    MI = (log(N) - H(p)) / log(N)

    其中:
        N: 相位 bin 数
        p: 振幅加权相位分布
        H(p) = -sum(p * log(p)): Shannon 熵

    MI = 0 表示无耦合, MI = 1 表示完全耦合.

    来源: Canolty & Knight (2010), NeuroImage.
    """
    if amplitude.size != phase_slow.size:
        raise ValueError("振幅和相位长度不一致")

    # 相位归一化到 [-pi, pi]
    phase_wrapped = np.angle(np.exp(1j * phase_slow))

    # 相位 bins
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    mean_amp = np.zeros(num_bins)

    for b in range(num_bins):
        mask = (phase_wrapped >= bin_edges[b]) & (phase_wrapped < bin_edges[b + 1])
        if np.any(mask):
            mean_amp[b] = np.mean(amplitude[mask])
        else:
            mean_amp[b] = 0.0

    # 归一化为概率分布
    total = np.sum(mean_amp)
    if total < 1.0e-30:
        return 0.0
    p = mean_amp / total
    p = p[p > 0]  # 去除零

    # 熵
    entropy = -np.sum(p * np.log(p))
    mi = (np.log(num_bins) - entropy) / np.log(num_bins)
    return float(mi)


def compute_comodulogram(
    signal: np.ndarray,
    dt: float,
    freq_slow: np.ndarray,
    freq_fast: np.ndarray,
    bw_slow: float = 2.0,
    bw_fast: float = 10.0,
    num_bins: int = 18,
) -> np.ndarray:
    """
    计算调制图 (comodulogram): 慢频相位对快频振幅的耦合.

    参数:
        signal: 输入信号
        dt: 采样间隔
        freq_slow: 慢频中心频率数组
        freq_fast: 快频中心频率数组
        bw_slow: 慢频带宽
        bw_fast: 快频带宽
        num_bins: 相位 bin 数

    返回:
        comod: 调制图 (len(freq_slow), len(freq_fast))
    """
    ns = freq_slow.size
    nf = freq_fast.size
    comod = np.zeros((ns, nf))

    for i, fs in enumerate(freq_slow):
        # 慢频带通滤波
        slow = bandpass_filter(signal, dt, fs - bw_slow / 2, fs + bw_slow / 2)
        _, phase_slow = instantaneous_amplitude_phase(slow)

        for j, ff in enumerate(freq_fast):
            # 快频带通滤波
            fast = bandpass_filter(signal, dt, ff - bw_fast / 2, ff + bw_fast / 2)
            amp_fast, _ = instantaneous_amplitude_phase(fast)

            comod[i, j] = modulation_index(amp_fast, phase_slow, num_bins)

    return comod


# -------------------------------------------------------------------
#  带通滤波器 (简化版 FIR)
# -------------------------------------------------------------------
def bandpass_filter(
    signal: np.ndarray,
    dt: float,
    f_low: float,
    f_high: float,
    order: int = 64,
) -> np.ndarray:
    """
    简单 FIR 带通滤波器 (频域实现).
    """
    n = signal.size
    S = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=dt)

    # 带通掩码
    mask = (np.abs(freq) >= f_low) & (np.abs(freq) <= f_high)
    # 平滑过渡
    S_filtered = S * mask
    return np.real(np.fft.ifft(S_filtered))


# -------------------------------------------------------------------
#  振荡事件检测
# -------------------------------------------------------------------
def detect_oscillation_bursts(
    amplitude: np.ndarray,
    dt: float,
    threshold_factor: float = 2.0,
    min_duration: int = 5,
) -> Dict[str, np.ndarray]:
    """
    检测振荡爆发事件 (burst detection).

    当振幅超过 (均值 + threshold * 标准差) 时认为爆发.

    返回:
        dict: "start_idx", "end_idx", "duration", "peak_amp"
    """
    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    threshold = mean_amp + threshold_factor * std_amp

    above = amplitude > threshold
    starts = []
    ends = []
    peaks = []

    in_burst = False
    start_idx = 0
    for i in range(len(above)):
        if above[i] and not in_burst:
            in_burst = True
            start_idx = i
        elif not above[i] and in_burst:
            in_burst = False
            duration = i - start_idx
            if duration >= min_duration:
                starts.append(start_idx)
                ends.append(i)
                peaks.append(np.max(amplitude[start_idx:i]))

    return {
        "start_idx": np.array(starts),
        "end_idx": np.array(ends),
        "duration": np.array(ends) - np.array(starts),
        "peak_amp": np.array(peaks),
    }
