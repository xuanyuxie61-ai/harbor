"""
wavelet_analysis.py
基于Haar小波变换的海洋内波信号时频分析

融合项目:
- 496_haar_transform: Haar小波变换

核心科学:
Haar小波是分析内波信号多尺度特征的理想工具。通过递归的平均-差分分解，
可以识别不同尺度上的内波破碎事件。

数学公式:
对于信号序列 v = [v_1, v_2, ..., v_{2n}]，一级Haar变换:
    近似系数: a_i = (v_{2i-1} + v_{2i}) / √2
    细节系数: d_i = (v_{2i-1} - v_{2i}) / √2

多尺度能量谱:
    E_j = Σ d_{j,k}²  （第 j 尺度的能量）
"""

import numpy as np


def haar_1d_transform(signal):
    """
    一维Haar小波变换
    
    参数:
        signal: 输入信号数组 (长度应为2的幂)
    
    返回:
        coeffs: 变换后系数 [近似, 细节_1, 细节_2, ...]
        energies: 各尺度能量
    """
    v = np.asarray(signal, dtype=float)
    n = len(v)
    
    # 找到不超过n的最大2的幂
    m = 2**int(np.floor(np.log2(n)))
    v = v[:m]
    
    coeffs = []
    energies = []
    current = v.copy()
    
    while len(current) > 1:
        k = len(current) // 2
        
        # 平均系数
        avg = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        # 细节系数
        diff = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        
        energies.append(np.sum(diff**2))
        coeffs.append(diff)
        current = avg
    
    coeffs.append(current)  # 最后一级近似系数
    coeffs.reverse()  # [近似, 细节_最低频, ..., 细节_最高频]
    energies.reverse()
    
    return coeffs, energies


def haar_1d_inverse(coeffs):
    """
    一维Haar小波逆变换
    
    参数:
        coeffs: 变换后系数 [近似, 细节_1, 细节_2, ...]
    
    返回:
        signal: 重建信号
    """
    approx = coeffs[0].copy()
    
    for j in range(1, len(coeffs)):
        detail = coeffs[j]
        n = len(approx)
        
        reconstructed = np.zeros(2 * n)
        reconstructed[0::2] = (approx + detail) / np.sqrt(2.0)
        reconstructed[1::2] = (approx - detail) / np.sqrt(2.0)
        approx = reconstructed
    
    return approx


def haar_2d_transform(field):
    """
    二维Haar小波变换 (可分离变换)
    
    先对列做1D变换，再对行做1D变换。
    
    参数:
        field: 2D数组
    
    返回:
        LL, LH, HL, HH: 近似、水平细节、垂直细节、对角细节
    """
    u = np.asarray(field, dtype=float)
    rows, cols = u.shape
    
    # 确保2的幂
    m_r = 2**int(np.floor(np.log2(rows)))
    m_c = 2**int(np.floor(np.log2(cols)))
    u = u[:m_r, :m_c]
    
    # 对每列做Haar变换
    col_approx = np.zeros((m_r // 2, m_c))
    col_detail = np.zeros((m_r // 2, m_c))
    
    for j in range(m_c):
        c, _ = haar_1d_transform(u[:, j])
        if len(c) >= 2:
            col_approx[:, j] = c[0][:m_r//2]
            # 最高频细节
            col_detail[:, j] = c[-1][:m_r//2]
    
    # 对每行做Haar变换
    LL = np.zeros((m_r // 2, m_c // 2))
    LH = np.zeros((m_r // 2, m_c // 2))
    HL = np.zeros((m_r // 2, m_c // 2))
    HH = np.zeros((m_r // 2, m_c // 2))
    
    for i in range(m_r // 2):
        c_a, _ = haar_1d_transform(col_approx[i, :])
        c_d, _ = haar_1d_transform(col_detail[i, :])
        
        if len(c_a) >= 2 and len(c_d) >= 2:
            LL[i, :] = c_a[0][:m_c//2]
            LH[i, :] = c_a[-1][:m_c//2] if len(c_a) > 1 else 0.0
            HL[i, :] = c_d[0][:m_c//2] if len(c_d) > 1 else 0.0
            HH[i, :] = c_d[-1][:m_c//2] if len(c_d) > 1 else 0.0
    
    return LL, LH, HL, HH


def detect_breaking_events(signal, threshold_factor=3.0):
    """
    利用小波细节系数检测内波破碎事件
    
    破碎事件在小波域表现为高频细节系数的突增。
    
    参数:
        signal: 内波速度/位移信号
        threshold_factor: 阈值因子 (标准差的倍数)
    
    返回:
        breaking_indices: 破碎事件索引
        wavelet_energy: 小波能量时间序列
    """
    coeffs, energies = haar_1d_transform(signal)
    
    # 最高频细节系数
    if len(coeffs) >= 2:
        detail = coeffs[-1]
    else:
        detail = np.zeros(1)
    
    # 滑动窗口计算局部能量
    window = max(4, len(detail) // 32)
    wavelet_energy = np.zeros(len(detail))
    
    for i in range(len(detail)):
        start = max(0, i - window // 2)
        end = min(len(detail), i + window // 2 + 1)
        wavelet_energy[i] = np.sum(detail[start:end]**2)
    
    # 阈值检测
    mean_energy = np.mean(wavelet_energy)
    std_energy = np.std(wavelet_energy)
    threshold = mean_energy + threshold_factor * std_energy
    
    breaking_indices = np.where(wavelet_energy > threshold)[0]
    
    return breaking_indices, wavelet_energy


def multi_scale_spectrum(signal):
    """
    计算内波信号的多尺度能量谱
    
    能量谱定义:
        E(j) = Σ_k |d_{j,k}|²
    
    其中 j 为小波尺度，k 为位置索引。
    
    参数:
        signal: 输入信号
    
    返回:
        scales: 尺度数组
        spectrum: 各尺度能量
    """
    coeffs, energies = haar_1d_transform(signal)
    
    # 去掉近似系数
    scales = np.arange(1, len(energies) + 1)
    spectrum = np.array(energies)
    
    # 归一化
    if np.sum(spectrum) > 1.0e-12:
        spectrum = spectrum / np.sum(spectrum)
    
    return scales, spectrum
