# -*- coding: utf-8 -*-
"""
feature_enhancer.py
======================================================================
光谱对比度增强 —— 邻域加权锐化

物理背景:
    系外行星光谱中的微弱特征 (如特定分子吸收线) 常常被
    强连续谱背景淹没。本模块采用类似图像对比度增强的方法
    (移植自 574_image_contrast), 对一维光谱进行锐化处理。

    增强公式 (类似 unsharp masking):
        s_enhanced[i] = s_val * s[i] + (1 - s_val) * <s>_neighbor
    其中 <s>_neighbor 为邻域均值, s_val > 1 为锐化参数。

    该方法等效于对光谱施加高通滤波:
        s_enhanced = s + (s_val - 1) * (s - <s>_neighbor)
                  = s + (s_val - 1) * Laplacian(s)

    物理意义: 增强吸收线的对比度, 使微弱分子特征可被
    反演算法更好地识别。

数学公式:
    离散 Laplacian (1D):
        L[i] = (s[i-1] - 2 s[i] + s[i+1]) / dx^2

    增强后:
        s_new[i] = s[i] - gamma * L[i] * dx^2
    gamma = s_val - 1 (锐化强度)

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ============================================================
# 光谱对比度增强 (移植自 574_image_contrast)
# ============================================================
def spectral_contrast_enhancement(
    spectrum: np.ndarray,
    sharpness: float = 1.5,
    neighbor_width: int = 4,
) -> np.ndarray:
    """
    一维光谱对比度增强 (移植自 574_image_contrast_gray)。

    原代码处理 2D 图像, 本模块简化为 1D 版本:
        gray3d[k, i] = gray[i + shift_k]  (8 邻域 -> 4 邻域)
        gray_avg = sum(gray3d) / 8
        gray_contrast = (1-s) * gray_avg + s * gray

    1D 版本:
        neighbor_avg[i] = (1/N) sum_{k in neighbors} s[i+k]
        s_enhanced[i] = s_val * s[i] + (1 - s_val) * neighbor_avg[i]

    边界处理: 与 574_image_contrast 相同, 保持边界值不变
    (避免边界伪影)。

    参数:
        spectrum : 输入光谱数组
        sharpness : 锐化参数 (s_val)
            s > 1 : 增强 (锐化)
            s < 1 : 模糊 (平滑)
            s = 1 : 不变
        neighbor_width : 单侧邻域宽度

    返回: 增强后的光谱
    """
    n = len(spectrum)
    if n < 2 * neighbor_width + 1:
        return spectrum.copy()

    s = float(sharpness)
    w = int(neighbor_width)
    spec = spectrum.astype(np.float64)

    # 计算邻域均值 (类似 574_image_contrast 的 gray_average)
    neighbor_sum = np.zeros(n, dtype=np.float64)
    for k in range(-w, w + 1):
        if k == 0:
            continue
        shifted = np.roll(spec, -k)
        neighbor_sum += shifted
    neighbor_avg = neighbor_sum / (2 * w)

    # 对比度增强: s_new = s * spec + (1 - s) * neighbor_avg
    enhanced = s * spec + (1.0 - s) * neighbor_avg

    # 边界保持 (移植自 574_image_contrast_gray 的边界处理)
    enhanced[:w] = spec[:w]
    enhanced[n - w :] = spec[n - w :]

    return enhanced


def multiscale_contrast_enhancement(
    spectrum: np.ndarray,
    scales: Optional[list] = None,
    weights: Optional[list] = None,
) -> np.ndarray:
    """
    多尺度对比度增强。
    在不同邻域宽度上应用增强, 然后加权叠加。

    物理应用: 同时增强窄线 (分子吸收) 和宽线 (压力展宽特征)。
    """
    if scales is None:
        scales = [2, 4, 8, 16]
    if weights is None:
        weights = [0.4, 0.3, 0.2, 0.1]

    if len(scales) != len(weights):
        raise ValueError("scales 与 weights 长度必须相同")

    n = len(spectrum)
    result = np.zeros(n, dtype=np.float64)
    total_weight = sum(weights)

    for scale, weight in zip(scales, weights):
        enhanced = spectral_contrast_enhancement(spectrum, sharpness=1.3, neighbor_width=scale)
        result += weight * enhanced

    result = result / total_weight
    return result


def detect_spectral_features(
    wavelength_grid: np.ndarray,
    spectrum: np.ndarray,
    original: Optional[np.ndarray] = None,
    threshold_sigma: float = 2.0,
) -> Dict:
    """
    检测光谱中的吸收/发射特征。

    方法: 增强后光谱减去原始光谱, 识别显著偏离的区域。

    返回:
        feature_indices : 特征位置索引
        feature_depths  : 特征深度
        feature_widths  : 特征宽度 (像素)
    """
    if original is None:
        original = spectrum

    enhanced = multiscale_contrast_enhancement(spectrum)
    residual = enhanced - original

    sigma = np.std(residual)
    threshold = threshold_sigma * sigma

    absorption_mask = residual > threshold
    emission_mask = residual < -threshold

    def find_contiguous_regions(mask: np.ndarray):
        regions = []
        in_region = False
        start = 0
        for i in range(len(mask)):
            if mask[i] and not in_region:
                start = i
                in_region = True
            elif not mask[i] and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(mask)))
        return regions

    abs_regions = find_contiguous_regions(absorption_mask)
    em_regions = find_contiguous_regions(emission_mask)

    return {
        "enhanced": enhanced,
        "residual": residual,
        "absorption_regions": abs_regions,
        "emission_regions": em_regions,
        "n_absorption": len(abs_regions),
        "n_emission": len(em_regions),
        "threshold": float(threshold),
        "sigma_residual": float(sigma),
    }
