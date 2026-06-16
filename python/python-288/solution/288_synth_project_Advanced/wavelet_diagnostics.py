"""
wavelet_diagnostics.py - 小波分析与偏滤器热负荷诊断

本模块融合以下种子项目的核心算法：
  - 1403_wavelet → Daubechies小波变换

功能：
  1. Daubechies小波滤波器系数计算
  2. 离散小波变换(DWT)与逆变换(IDWT)
  3. 多分辨率分析(MRA)
  4. 小波系数阈值去噪
  5. 热负荷信号的特征提取
  6. ELM（边缘局域模）事件检测
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


# =============================================================================
# Daubechies滤波器系数（来自1403_wavelet）
# =============================================================================
def daub_coefficients(p: int) -> np.ndarray:
    """
    计算Daubechies D2p小波的滤波器系数

    滤波器系数满足以下条件：
    1. 正交性: sum(h_k * h_{k-2n}) = delta_{n,0}
    2. vanishing moments: sum((-1)^k * k^m * h_k) = 0, m=0,...,p-1
    3. 归一化: sum(h_k) = sqrt(2)

    通过求解多项式方程组获得

    参数:
        p: 消失矩数（D2p表示有2p个系数）
    返回:
        h: shape (2p,) 低通滤波器系数
    """
    if p == 1:
        # Haar小波
        return np.array([1.0, 1.0]) / np.sqrt(2)

    elif p == 2:
        # D4 (Daubechies-4)
        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        h = np.array([
            (1 + sqrt3) / (4 * sqrt2),
            (3 + sqrt3) / (4 * sqrt2),
            (3 - sqrt3) / (4 * sqrt2),
            (1 - sqrt3) / (4 * sqrt2),
        ])
        return h

    elif p == 3:
        # D6
        h = np.array([
            0.332670552950,
            0.806891509311,
            0.459877502118,
            -0.135011020010,
            -0.085441273882,
            0.035226291882,
        ])
        return h

    elif p == 4:
        # D8
        h = np.array([
            0.230377813309,
            0.714846570553,
            0.630880767930,
            -0.027983769417,
            -0.187034811719,
            0.030841381836,
            0.032883011667,
            -0.010597401785,
        ])
        return h

    elif p == 5:
        # D10
        h = np.array([
            0.160102397974,
            0.603829269797,
            0.724308528438,
            0.138428145901,
            -0.242294887066,
            -0.032244869585,
            0.077571493840,
            -0.006241490213,
            -0.012580751999,
            0.003335725285,
        ])
        return h

    else:
        # 默认使用D4
        return daub_coefficients(2)


def daub_highpass(h: np.ndarray) -> np.ndarray:
    """
    从低通滤波器构造高通滤波器

    g_k = (-1)^k * h_{2p-1-k}

    参数:
        h: 低通滤波器系数
    返回:
        g: 高通滤波器系数
    """
    n = len(h)
    g = np.zeros(n)
    for k in range(n):
        g[k] = (-1)**k * h[n - 1 - k]
    return g


# =============================================================================
# 离散小波变换 (DWT)
# =============================================================================
def dwt_1d(signal: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    一级离散小波变换

    算法:
    1. 低通滤波: a[n] = sum_k h[k-2n] * x[k]  (近似系数)
    2. 高通滤波: d[n] = sum_k g[k-2n] * x[k]  (细节系数)
    3. 下采样2倍

    参数:
        signal: 输入信号
        h: 低通滤波器
    返回:
        approx: 近似系数
        detail: 细节系数
    """
    n = len(signal)
    g = daub_highpass(h)
    filt_len = len(h)

    # 周期延拓处理边界
    signal_ext = np.concatenate([signal[-(filt_len // 2):], signal, signal[:filt_len // 2]])

    # 卷积并下采样
    approx = np.zeros(n // 2)
    detail = np.zeros(n // 2)

    for i in range(n // 2):
        idx = 2 * i
        for k in range(filt_len):
            if idx + k < len(signal_ext):
                approx[i] += h[k] * signal_ext[idx + k]
                detail[i] += g[k] * signal_ext[idx + k]

    return approx, detail


def idwt_1d(approx: np.ndarray, detail: np.ndarray,
              h: np.ndarray, original_length: int) -> np.ndarray:
    """
    一级逆离散小波变换

    算法:
    1. 上采样2倍
    2. 滤波（使用合成滤波器）
    3. 相加

    参数:
        approx: 近似系数
        detail: 细节系数
        h: 低通滤波器
        original_length: 原始信号长度
    返回:
        reconstructed: 重构信号
    """
    n = len(approx)
    g = daub_highpass(h)

    # 上采样
    approx_up = np.zeros(2 * n)
    detail_up = np.zeros(2 * n)
    approx_up[::2] = approx
    detail_up[::2] = detail

    # 合成滤波器（时间反转）
    h_syn = h[::-1]
    g_syn = g[::-1]

    # 滤波
    reconstructed = np.convolve(approx_up, h_syn, mode='full') + \
                    np.convolve(detail_up, g_syn, mode='full')

    # 裁剪到原始长度
    start = len(h) - 1
    reconstructed = reconstructed[start:start + original_length]

    if len(reconstructed) < original_length:
        reconstructed = np.pad(reconstructed, (0, original_length - len(reconstructed)))

    return reconstructed[:original_length]


# =============================================================================
# 多级小波分解
# =============================================================================
def wavedec(signal: np.ndarray, level: int = 3,
              wavelet_order: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    多级小波分解

    参数:
        signal: 输入信号
        level: 分解层数
        wavelet_order: 小波阶数 (p for D2p)
    返回:
        details: 各级细节系数列表 [d1, d2, ..., d_level]
        approx: 最终近似系数
    """
    h = daub_coefficients(wavelet_order // 2)
    details = []
    current = signal.copy()

    for lev in range(level):
        n = len(current)
        if n < 2 * len(h):
            break

        # 确保长度为偶数
        if n % 2 != 0:
            current = np.append(current, current[-1])

        approx, detail = dwt_1d(current, h)
        details.append(detail)
        current = approx

    return details, current


def waverec(details: List[np.ndarray], approx: np.ndarray,
              wavelet_order: int = 4,
              original_length: int = None) -> np.ndarray:
    """
    多级小波重构

    参数:
        details: 各级细节系数
        approx: 最终近似系数
        wavelet_order: 小波阶数
        original_length: 原始信号长度
    返回:
        reconstructed: 重构信号
    """
    h = daub_coefficients(wavelet_order // 2)
    current = approx.copy()

    for lev in range(len(details) - 1, -1, -1):
        detail = details[lev]
        n = 2 * len(detail)
        current = idwt_1d(current, detail, h, n)

    if original_length is not None:
        current = current[:original_length]

    return current


# =============================================================================
# 小波阈值去噪
# =============================================================================
def wavelet_denoise(signal: np.ndarray, level: int = 3,
                      threshold_method: str = 'universal',
                      wavelet_order: int = 4) -> np.ndarray:
    """
    小波阈值去噪

    步骤:
    1. 多级小波分解
    2. 对细节系数施加阈值
    3. 重构信号

    阈值方法:
    - 'universal': lambda = sigma * sqrt(2*log(N))
    - 'minimax': Minimax估计
    - 'sure': SURE阈值

    参数:
        signal: 含噪信号
        level: 分解层数
        threshold_method: 阈值方法
        wavelet_order: 小波阶数
    返回:
        denoised: 去噪后信号
    """
    original_length = len(signal)
    details, approx = wavedec(signal, level, wavelet_order)

    # 估计噪声标准差（从最细尺度系数）
    if len(details) > 0:
        sigma = np.median(np.abs(details[0])) / 0.6745
    else:
        sigma = 0.0

    N = len(signal)

    # 阈值
    if threshold_method == 'universal':
        threshold = sigma * np.sqrt(2.0 * np.log(N + 1))
    elif threshold_method == 'minimax':
        if N > 32:
            threshold = sigma * (0.3936 + 0.1829 * np.log(N / np.log(2) + 1e-30))
        else:
            threshold = 0.0
    else:
        threshold = sigma * np.sqrt(2.0 * np.log(N + 1))

    # 软阈值
    denoised_details = []
    for d in details:
        d_thresh = np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
        denoised_details.append(d_thresh)

    # 重构
    reconstructed = waverec(denoised_details, approx, wavelet_order, original_length)

    return reconstructed


# =============================================================================
# 热负荷信号分析
# =============================================================================
def compute_wavelet_energy(details: List[np.ndarray]) -> np.ndarray:
    """
    计算各级小波系数的能量

    E_j = sum_k |d_{j,k}|^2

    参数:
        details: 各级细节系数
    返回:
        energy: 各级能量
    """
    energy = np.zeros(len(details))
    for j, d in enumerate(details):
        energy[j] = np.sum(d**2)
    return energy


def compute_wavelet_entropy(details: List[np.ndarray]) -> float:
    """
    计算小波包熵

    S = -sum(p_j * log(p_j))
    其中 p_j = E_j / sum(E_j)

    参数:
        details: 各级细节系数
    返回:
        entropy: 小波熵
    """
    energy = compute_wavelet_energy(details)
    total = np.sum(energy)

    if total < 1e-30:
        return 0.0

    p = energy / total
    p = p[p > 0]  # 过滤零值
    entropy = -np.sum(p * np.log(p + 1e-30))

    return float(entropy)


def detect_elm_events(signal: np.ndarray, threshold_factor: float = 3.0,
                        min_distance: int = 10) -> np.ndarray:
    """
    检测ELM（Edge Localized Mode）事件

    ELM表现为热负荷信号中的瞬态尖峰

    算法:
    1. 小波分解获取瞬态特征
    2. 计算包络线
    3. 检测超过阈值的尖峰

    参数:
        signal: 热负荷时间序列
        threshold_factor: 阈值因子（标准差的倍数）
        min_distance: 最小事件间隔
    返回:
        event_indices: 检测到的事件位置
    """
    # 小波分解
    details, _ = wavedec(signal, level=3)

    if len(details) == 0:
        return np.array([], dtype=int)

    # 使用第一级细节系数的幅值作为包络
    envelope = np.abs(details[0])

    # 上采样到原始信号长度
    n_orig = len(signal)
    if len(envelope) < n_orig:
        # 线性插值
        x_old = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, n_orig)
        envelope = np.interp(x_new, x_old, envelope)

    # 阈值
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    threshold = mean_env + threshold_factor * std_env

    # 检测尖峰
    above_threshold = envelope > threshold
    event_indices = []

    i = 0
    while i < n_orig:
        if above_threshold[i]:
            # 找到这个尖峰的峰值位置
            j = i
            while j < n_orig and above_threshold[j]:
                j += 1
            peak_idx = i + np.argmax(envelope[i:j])
            event_indices.append(peak_idx)
            i = j + min_distance
        else:
            i += 1

    return np.array(event_indices, dtype=int)


# =============================================================================
# 热负荷湍流特征分析
# =============================================================================
def heat_flux_intermittency(signal: np.ndarray) -> dict:
    """
    计算热负荷信号的间歇性特征

    参数:
        signal: 热负荷时间序列
    返回:
        dict包含偏度、峰度、Hurst指数等
    """

    # 去均值
    fluct = signal - np.mean(signal)
    sigma = np.std(fluct)

    if sigma < 1e-30:
        return {
            'skewness': 0.0,
            'kurtosis': 3.0,
            'hurst_exponent': 0.5,
            'intermittency_param': 0.0,
        }

    fluct_norm = fluct / sigma
    n = len(fluct_norm)

    # 偏度 (skewness)
    skewness = np.mean(fluct_norm**3)

    # 峰度 (kurtosis) - 高斯分布为3
    kurtosis = np.mean(fluct_norm**4)

    # Hurst指数 (R/S方法)
    max_k = min(n // 2, 64)
    log_ns = []
    log_rs = []

    for k in [4, 8, 16, 32, 64]:
        if k > n // 2:
            break
        n_blocks = n // k
        if n_blocks < 1:
            break

        rs_values = []
        for b in range(n_blocks):
            segment = fluct_norm[b * k:(b + 1) * k]
            mean_seg = np.mean(segment)
            deviation = np.cumsum(segment - mean_seg)
            R = np.max(deviation) - np.min(deviation)
            S = np.std(segment)
            if S > 1e-30:
                rs_values.append(R / S)

        if len(rs_values) > 0:
            log_ns.append(np.log(k))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_ns) >= 2:
        # 线性拟合求Hurst指数
        coeffs = np.polyfit(log_ns, log_rs, 1)
        hurst = coeffs[0]
    else:
        hurst = 0.5

    # 间歇性参数
    intermittency = (kurtosis - 3.0) / (kurtosis + 1e-30)

    return {
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'hurst_exponent': float(hurst),
        'intermittency_param': float(intermittency),
        'sigma': float(sigma),
    }
