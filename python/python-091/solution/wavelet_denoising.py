"""
超声回波信号Haar小波变换去噪与特征提取模块

基于种子项目 496_haar_transform 的核心算法，
为超声A-scan/B-scan信号提供多分辨率分析与自适应阈值去噪。

物理背景:
超声回波信号包含:
- 组织反射信号（有用信号）
- 电子噪声（高频随机成分）
- 混响伪影（低频相干噪声）

小波变换通过多分辨率分解将信号投影到不同尺度空间:
    f(t) = Σ c_{J,k}·φ_{J,k}(t) + Σ Σ d_{j,k}·ψ_{j,k}(t)
其中:
    φ_{j,k}(t) = 2^{-j/2}·φ(2^{-j}t - k)  尺度函数
    ψ_{j,k}(t) = 2^{-j/2}·ψ(2^{-j}t - k)  小波函数

Haar小波是最简单的正交小波:
    ψ(t) = 1  (0 ≤ t < 0.5)
           -1 (0.5 ≤ t < 1)
           0  (其他)

分解步骤（每级）:
    c_j[k] = (c_{j+1}[2k] + c_{j+1}[2k+1]) / √2  (近似系数)
    d_j[k] = (c_{j+1}[2k] - c_{j+1}[2k+1]) / √2  (细节系数)
"""

import numpy as np
from typing import Tuple, List


def haar_step_1d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """执行单步一维Haar小波分解。
    
    参数:
        signal: 输入信号（长度应为偶数）
    
    返回:
        approx: 近似（低频）系数
        detail: 细节（高频）系数
    """
    n = len(signal)
    if n % 2 != 0:
        # 奇数长度时，截断最后一个样本
        signal = signal[:-1]
        n -= 1
    
    if n < 2:
        return signal.copy(), np.zeros_like(signal)
    
    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)
    
    for k in range(half):
        approx[k] = (signal[2*k] + signal[2*k+1]) / np.sqrt(2.0)
        detail[k] = (signal[2*k] - signal[2*k+1]) / np.sqrt(2.0)
    
    return approx, detail


def haar_1d(signal: np.ndarray, n_levels: int = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """执行完整的一维Haar小波变换（多级分解）。
    
    返回的系数数组包含各级近似和细节系数:
        coeffs = [c_J, d_J, d_{J-1}, ..., d_1]
    
    参数:
        signal: 输入信号
        n_levels: 分解级数，None时自动计算最大级数
    
    返回:
        coeffs: 小波系数数组（第0个元素为最低频近似系数）
        details: 各级细节系数列表
    """
    n = len(signal)
    max_levels = int(np.floor(np.log2(n)))
    
    if n_levels is None:
        n_levels = max_levels
    else:
        n_levels = min(n_levels, max_levels)
    
    if n_levels < 1:
        return signal.copy(), []
    
    details = []
    current = signal.astype(float)
    
    for _ in range(n_levels):
        approx, detail = haar_step_1d(current)
        details.append(detail)
        current = approx
    
    # 组装系数: [近似系数, detail_n, detail_{n-1}, ..., detail_1]
    coeffs = [current]
    for detail in reversed(details):
        coeffs.append(detail)
    
    return np.concatenate(coeffs), details


def haar_1d_inverse(coeffs: np.ndarray, n_levels: int) -> np.ndarray:
    """执行一维Haar小波逆变换（信号重构）。
    
    重构公式（单步）:
        c_{j+1}[2k]   = (c_j[k] + d_j[k]) / √2
        c_{j+1}[2k+1] = (c_j[k] - d_j[k]) / √2
    
    参数:
        coeffs: 小波系数数组
        n_levels: 分解级数
    
    返回:
        signal: 重构信号
    """
    if n_levels < 1:
        return coeffs.copy()
    
    # 分离近似系数和细节系数
    # coeffs = [c_J, d_J, d_{J-1}, ..., d_1]
    # 先找到c_J的长度
    n_total = len(coeffs)
    approx_len = n_total // (2**n_levels)
    
    # 更精确地计算各级长度
    current = coeffs[:approx_len].copy()
    idx = approx_len
    
    for level in range(n_levels):
        detail_len = len(current)
        detail = coeffs[idx:idx + detail_len]
        idx += detail_len
        
        # 单步重构
        reconstructed = np.zeros(2 * detail_len)
        for k in range(detail_len):
            reconstructed[2*k] = (current[k] + detail[k]) / np.sqrt(2.0)
            reconstructed[2*k+1] = (current[k] - detail[k]) / np.sqrt(2.0)
        
        current = reconstructed
    
    return current


def universal_threshold(details: List[np.ndarray], sigma: float = None) -> float:
    """计算Donoho-Johnstone通用阈值。
    
    公式: λ = σ·√(2·log(N))
    
    其中:
    - σ 为噪声标准差（可用最高级细节系数的MAD估计）
    - N 为信号长度
    - MAD (Median Absolute Deviation): σ ≈ median(|d|) / 0.6745
    
    参数:
        details: 各级细节系数列表
        sigma: 噪声标准差，None时自动估计
    
    返回:
        threshold: 阈值
    """
    # 使用最高频细节系数估计噪声水平
    finest_detail = details[-1]
    N = sum(len(d) for d in details) + len(details[0]) if details else len(finest_detail)
    
    if sigma is None:
        # MAD估计
        median_abs = np.median(np.abs(finest_detail))
        sigma = median_abs / 0.6745
        if sigma < 1e-14:
            sigma = 1e-14
    
    threshold = sigma * np.sqrt(2.0 * np.log(N))
    return threshold


def soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """软阈值函数（Soft Thresholding）。
    
    公式:
        η_λ(x) = sign(x)·max(|x| - λ, 0)
    
    软阈值是L1正则化（Lasso）的解析解，具有良好的统计性质。
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def hard_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """硬阈值函数（Hard Thresholding）。
    
    公式:
        H_λ(x) = x  if |x| > λ
                 0  otherwise
    """
    result = coeffs.copy()
    result[np.abs(result) <= threshold] = 0.0
    return result


def denoise_ultrasound_signal(signal: np.ndarray, n_levels: int = None,
                              threshold_mode: str = 'soft') -> Tuple[np.ndarray, dict]:
    """对超声回波信号进行小波去噪。
    
    处理流程:
    1. Haar小波多分辨率分解
    2. 估计噪声水平（MAD）
    3. 计算通用阈值
    4. 对细节系数进行阈值处理
    5. 小波逆变换重构信号
    
    参数:
        signal: 输入超声信号（A-scan）
        n_levels: 分解级数
        threshold_mode: 'soft' 或 'hard'
    
    返回:
        denoised: 去噪后的信号
        info: 包含处理参数的字典
    """
    # 小波分解
    coeffs, details = haar_1d(signal, n_levels)
    
    # 估计噪声水平
    threshold = universal_threshold(details)
    
    # 阈值处理（保留近似系数，只处理细节系数）
    # coeffs = [c_J, d_J, d_{J-1}, ..., d_1]
    n_approx = len(details[0]) if details else len(coeffs) // 2
    
    # 更精确地分离
    approx_len = len(coeffs)
    for d in details:
        approx_len -= len(d)
    
    approx_coeffs = coeffs[:approx_len].copy()
    detail_coeffs = coeffs[approx_len:].copy()
    
    # 应用阈值
    if threshold_mode == 'soft':
        detail_coeffs = soft_threshold(detail_coeffs, threshold)
    else:
        detail_coeffs = hard_threshold(detail_coeffs, threshold)
    
    # 重构
    denoised_coeffs = np.concatenate([approx_coeffs, detail_coeffs])
    denoised = haar_1d_inverse(denoised_coeffs, len(details))
    
    # 截取到原始长度
    denoised = denoised[:len(signal)]
    
    info = {
        'n_levels': len(details),
        'threshold': float(threshold),
        'noise_estimate': float(threshold / np.sqrt(2.0 * np.log(len(signal)))),
        'threshold_mode': threshold_mode,
        'original_energy': float(np.sum(signal**2)),
        'denoised_energy': float(np.sum(denoised**2))
    }
    
    return denoised, info


def extract_multiscale_features(signal: np.ndarray, n_levels: int = 4) -> dict:
    """提取超声信号的多尺度特征。
    
    特征包括各级细节系数的能量、熵和峰值位置，
    可用于组织特征分类和病变检测。
    
    返回:
        特征字典
    """
    coeffs, details = haar_1d(signal, n_levels)
    
    features = {}
    total_energy = np.sum(signal**2)
    
    for i, detail in enumerate(details):
        level = len(details) - i  # 从粗到细编号
        energy = np.sum(detail**2)
        
        # Shannon熵（归一化系数）
        abs_coeffs = np.abs(detail)
        sum_abs = np.sum(abs_coeffs)
        if sum_abs > 1e-14:
            p = abs_coeffs / sum_abs
            entropy = -np.sum(p * np.log(p + 1e-14))
        else:
            entropy = 0.0
        
        features[f'level_{level}_energy'] = float(energy)
        features[f'level_{level}_energy_ratio'] = float(energy / (total_energy + 1e-14))
        features[f'level_{level}_entropy'] = float(entropy)
        features[f'level_{level}_max_coeff'] = float(np.max(np.abs(detail)))
        features[f'level_{level}_mean_coeff'] = float(np.mean(np.abs(detail)))
    
    return features
