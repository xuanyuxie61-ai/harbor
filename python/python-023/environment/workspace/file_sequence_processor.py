#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
波场时间序列处理器
================================================================================

基于 214_contour_sequence4 的序列数据处理思想，
处理等离子体波场随时间演化的序列数据。

核心功能：
1. 时间序列波场数据的顺序处理
2. 数据网格重排（从向量到二维数组）
3. 统计量计算（极值、均值、涨落水平）
4. 序列文件名的自动递增处理

物理背景：

电磁波包的时间演化可描述为：
    E(x, t) = E_0(x, t) exp[i φ(x, t)]
    
其中包络 E_0 满足非线性Schrödinger方程：
    i (∂E/∂t + v_g ∂E/∂x) + (1/2) v_g' ∂²E/∂x² - ω_{nl} |E|² E = 0

v_g = ∂ω/∂k 为群速度，v_g' = ∂²ω/∂k² 为群速度色散。

时间序列分析：
- 场振幅的均方根涨落：δE_rms = √(⟨|E|²⟩ - |⟨E⟩|²)
- 功率谱密度：S(ω) = |∫ E(t) e^{-iωt} dt|²
- 相干时间：τ_c = ∫ |⟨E(t) E*(t+τ)⟩|² dτ / |⟨|E|²⟩|²
================================================================================
"""

import numpy as np


def generate_filename_sequence(base_name, n_files):
    """
    基于 214_contour_sequence4 的文件名递增思想，
    生成序列文件名，如 'field001.dat', 'field002.dat', ...
    
    参数
    ----
    base_name : str
        基础文件名（不含序号）。
    n_files : int
        文件数量。
        
    返回
    ----
    filenames : list of str
        文件名列表。
    """
    filenames = []
    
    for i in range(1, n_files + 1):
        # 查找文件名中的数字位置并递增
        suffix = f"_{i:03d}"
        
        # 在扩展名前插入序号
        if '.' in base_name:
            parts = base_name.rsplit('.', 1)
            filename = f"{parts[0]}{suffix}.{parts[1]}"
        else:
            filename = f"{base_name}{suffix}"
        
        filenames.append(filename)
    
    return filenames


def process_field_timeseries(omega_solutions, params, n_frames=20):
    """
    处理波场的时间序列数据。
    
    参数
    ----
    omega_solutions : ndarray
        色散解。
    params : dict
        物理参数。
    n_frames : int
        时间步数。
        
    返回
    ----
    stats : dict
        统计信息。
    """
    n_modes = len(omega_solutions)
    if n_modes == 0:
        return {'n_frames': 0, 'min_amp': 0.0, 'max_amp': 0.0}
    
    # 生成模拟的波场时间序列
    Omega_e = params['Omega_e']
    t_values = np.linspace(0, 10.0 / Omega_e, n_frames)
    
    # 每个模式的振幅时间序列
    amplitudes = np.zeros((n_modes, n_frames))
    phases = np.zeros((n_modes, n_frames))
    
    for m in range(n_modes):
        k = omega_solutions[m, 0]
        omega = complex(omega_solutions[m, 1])
        omega_r = omega.real
        gamma = omega.imag
        
        for t_idx, t in enumerate(t_values):
            # 波包振幅：exp(γ t) × 随机相位噪声
            amp = np.exp(gamma * t) * (1.0 + 0.1 * np.sin(omega_r * t))
            
            # 边界处理：避免指数爆炸
            amp = np.clip(amp, 0.0, 10.0)
            
            phase = omega_r * t + 0.05 * np.random.randn()
            
            amplitudes[m, t_idx] = amp
            phases[m, t_idx] = phase
    
    # 计算统计量
    all_amps = amplitudes.flatten()
    min_amp = np.min(all_amps)
    max_amp = np.max(all_amps)
    mean_amp = np.mean(all_amps)
    rms_amp = np.sqrt(np.mean(all_amps**2))
    
    # 计算功率谱密度
    dt = t_values[1] - t_values[0]
    
    # 对总场做FFT
    total_field = np.sum(amplitudes * np.cos(phases), axis=0)
    fft_field = np.fft.rfft(total_field)
    freqs = np.fft.rfftfreq(n_frames, dt)
    psd = np.abs(fft_field)**2 * dt / n_frames
    
    # 主导频率
    if len(freqs) > 1:
        dominant_freq = freqs[np.argmax(psd[1:]) + 1]
    else:
        dominant_freq = 0.0
    
    # 序列文件名生成
    filenames = generate_filename_sequence("wave_field.dat", n_frames)
    
    stats = {
        'n_frames': n_frames,
        'n_modes': n_modes,
        'min_amp': min_amp,
        'max_amp': max_amp,
        'mean_amp': mean_amp,
        'rms_amp': rms_amp,
        'dominant_freq': dominant_freq,
        'filenames': filenames
    }
    
    return stats


def extract_grid_from_sequence(data_sequence, nx, ny, orientation=1):
    """
    从序列数据中提取二维网格数据。
    
    基于 214_contour_sequence4 的网格重排思想。
    
    参数
    ----
    data_sequence : ndarray, shape (N,)
        序列数据向量。
    nx, ny : int
        网格维度。
    orientation : int
        数据排列方向（1=行优先，2=列优先）。
        
    返回
    ----
    grid : ndarray, shape (nx, ny)
        二维网格数据。
    """
    N = len(data_sequence)
    
    if N != nx * ny:
        # 边界处理：填充或截断
        if N < nx * ny:
            padded = np.zeros(nx * ny)
            padded[:N] = data_sequence
            data_sequence = padded
        else:
            data_sequence = data_sequence[:nx * ny]
    
    if orientation == 1:
        grid = data_sequence.reshape((nx, ny))
    else:
        grid = data_sequence.reshape((ny, nx)).T
    
    return grid


def compute_temporal_correlation(field_series, max_lag=None):
    """
    计算波场的时间自相关函数。
    
    C(τ) = ⟨E(t) E*(t+τ)⟩ / ⟨|E(t)|²⟩
    """
    n = len(field_series)
    if max_lag is None:
        max_lag = n // 4
    
    max_lag = min(max_lag, n - 1)
    
    # 去均值
    field_centered = field_series - np.mean(field_series)
    norm = np.sum(field_centered**2)
    
    if norm < 1e-30:
        return np.ones(max_lag + 1)
    
    correlation = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            correlation[lag] = 1.0
        else:
            corr = np.sum(field_centered[:-lag] * field_centered[lag:])
            correlation[lag] = corr / norm
    
    return correlation
