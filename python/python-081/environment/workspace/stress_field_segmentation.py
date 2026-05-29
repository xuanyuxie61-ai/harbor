"""
应力场分割与损伤区域识别模块
============================
基于种子项目:
  - 586_image_threshold: 灰度图像阈值分割

科学背景:
  在大变形分析中，识别高应力集中区域对预测裂纹萌生至关重要。
  将von Mises应力场视为"灰度图像"(每个单元为一个像素)，
  利用阈值分割技术提取危险区域：
  1. 单阈值分割: 将单元分为安全区(σ_vm ≤ σ_th)和危险区(σ_vm > σ_th)
  2. 双阈值分割: 增加过渡区概念
  3. 基于统计的自适应阈值 (Otsu方法推广)

关键公式:
  - 单阈值分类:
      C(e) = 0 (安全)  if σ_vm(e) ≤ θ
      C(e) = 1 (危险)  if σ_vm(e) > θ
  - 双阈值分类:
      C(e) = 0 (安全)  if σ_vm ≤ θ_low
      C(e) = 1 (过渡)  if θ_low < σ_vm ≤ θ_high
      C(e) = 2 (危险)  if σ_vm > θ_high
  - 类间方差 (Otsu):
      σ_b^2(θ) = ω_0(θ) ω_1(θ) [μ_0(θ) - μ_1(θ)]^2
      最优阈值 θ* = argmax σ_b^2(θ)
"""

import numpy as np
from typing import Tuple, List


def single_threshold_segmentation(sigma_vm: np.ndarray,
                                   threshold: float) -> np.ndarray:
    """
    单阈值分割应力场。

    参数:
        sigma_vm: (E,) 单元von Mises应力数组
        threshold: 阈值

    返回:
        labels: (E,) 分类标签 (0=安全, 1=危险)
    """
    labels = np.where(sigma_vm > threshold, 1, 0)
    return labels


def dual_threshold_segmentation(sigma_vm: np.ndarray,
                                 theta_low: float,
                                 theta_high: float) -> np.ndarray:
    """
    双阈值分割应力场。

    参数:
        sigma_vm: (E,) 单元von Mises应力
        theta_low, theta_high: 低/高阈值

    返回:
        labels: (E,) 分类标签 (0=安全, 1=过渡, 2=危险)
    """
    labels = np.zeros(len(sigma_vm), dtype=np.int32)
    labels[(sigma_vm > theta_low) & (sigma_vm <= theta_high)] = 1
    labels[sigma_vm > theta_high] = 2
    return labels


def otsu_threshold(sigma_vm: np.ndarray, n_bins: int = 256) -> float:
    """
    Otsu自适应阈值分割。
    寻找使类间方差最大的阈值。

    参数:
        sigma_vm: (E,) 应力数组
        n_bins: 直方图分箱数

    返回:
        optimal_threshold: 最优阈值
    """
    if len(sigma_vm) == 0:
        return 0.0
    data = sigma_vm.flatten()
    vmin, vmax = np.min(data), np.max(data)
    if vmax - vmin < 1e-12:
        return vmin

    # 构建直方图
    bins = np.linspace(vmin, vmax, n_bins + 1)
    hist, _ = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    total = len(data)
    total_mean = np.mean(data)
    max_variance = -1.0
    optimal_threshold = vmin

    omega_0 = 0
    sum_0 = 0.0
    for i in range(n_bins):
        omega_0 += hist[i]
        sum_0 += hist[i] * bin_centers[i]
        if omega_0 == 0 or omega_0 == total:
            continue
        omega_1 = total - omega_0
        mu_0 = sum_0 / omega_0
        mu_1 = (total * total_mean - sum_0) / omega_1
        variance = (omega_0 * omega_1 / (total ** 2)) * (mu_0 - mu_1) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bin_centers[i]

    return float(optimal_threshold)


def compute_damage_zone_statistics(sigma_vm: np.ndarray,
                                    labels: np.ndarray) -> dict:
    """
    计算损伤区域的统计信息。

    参数:
        sigma_vm: (E,) 应力数组
        labels: (E,) 区域标签

    返回:
        stats: 统计字典
    """
    unique_labels = np.unique(labels)
    stats = {}
    for lbl in unique_labels:
        mask = labels == lbl
        zone_stresses = sigma_vm[mask]
        stats[f"zone_{lbl}"] = {
            "count": int(np.sum(mask)),
            "mean_stress": float(np.mean(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
            "max_stress": float(np.max(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
            "min_stress": float(np.min(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
        }
    return stats


def identify_critical_elements(sigma_vm: np.ndarray,
                                elements: np.ndarray,
                                top_percentile: float = 5.0) -> np.ndarray:
    """
    识别应力最高的前 top_percentile% 单元作为关键单元。

    参数:
        sigma_vm: (E,) 应力数组
        elements: (E, 4) 单元连接表
        top_percentile: 百分比阈值

    返回:
        critical_indices: 关键单元索引数组
    """
    threshold = np.percentile(sigma_vm, 100.0 - top_percentile)
    critical = np.where(sigma_vm >= threshold)[0]
    return critical
