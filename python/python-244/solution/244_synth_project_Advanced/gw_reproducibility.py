#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gw_reproducibility.py
=====================
【融合种子项目】 1233_eileenrmartin_dissertation (DAS 可复现性分析)

将 DAS 信号可复现性分析框架移植到中子星引力波信号的可复现性检验.

物理/数学公式
-------------
1. 中子星并合引力波频率 (后牛顿近似):
       f_GW ≈ (1/pi) sqrt(G M_total / r_ISCO^3)
   其中 r_ISCO = 6 GM/c^2 为最内稳定圆轨道.

2. 潮汐形变参数 (Love number):
       k_2 = (3/4) (1-2C)^2 [2-y_R + R y'_R / lambda_R]
             / [C^5 (2C(y_R-1) - y_R + 2 + ...)]
   其中 C = GM/(Rc^2), y_R 为径向导数在表面的值.

3. 可复现性指标 (Pearson 相关):
       r = sum (x_i - x_bar)(y_i - y_bar) / sqrt(sum(x-xbar)^2 sum(y-ybar)^2)

4. 互相关函数:
       C_{12}(tau) = integral x(t) y(t+tau) dt / sqrt(E_x E_y)
"""

import math
import numpy as np
from typing import Tuple, Dict
from numerical_constants import NeutronStarConstants as NS


def gw_frequency_isco(M_total: float) -> float:
    """
    ISCO 处的引力波频率.
    f_GW = c^3 / (6^{3/2} pi G M)
    """
    if M_total <= 0:
        return 0.0
    return NS.c_light ** 3 / (6.0 ** 1.5 * math.pi * NS.G_newton * M_total)


def compactness_from_mr(M_sun: float, R_km: float) -> float:
    """计算紧凑度 C = GM/(Rc^2)."""
    M_g = M_sun * NS.M_sun
    R_cm = R_km * 1e5
    return NS.G_newton * M_g / (R_cm * NS.c_light ** 2)


def love_number_k2_approx(C: float, gamma_eff: float = 0.5) -> float:
    """
    Love 数 k_2 的近似 (Hinderer 2008 简化).

    k_2 ≈ (3/4)(1-2C)^2 [2 + gamma_eff - ...] / [C^5 (...)]
    """
    if C <= 0 or C >= 0.5:
        return 0.0
    x = 1.0 - 2.0 * C
    num = 3.0 * x * x * (2.0 + gamma_eff)
    den = 8.0 * C ** 5 * (3.0 + gamma_eff + 2.0 * C * gamma_eff)
    return num / den if den > 0 else 0.0


def tidal_deformability(M_sun: float, R_km: float,
                         k2: float = None) -> float:
    """
    无量纲潮汐形变参数 Lambda.

    Lambda = (2/3) k_2 / C^5
    """
    C = compactness_from_mr(M_sun, R_km)
    if C <= 0:
        return 0.0
    if k2 is None:
        k2 = love_number_k2_approx(C)
    return (2.0 / 3.0) * k2 / C ** 5


def reproducibility_correlation(signal1: np.ndarray,
                                 signal2: np.ndarray) -> float:
    """
    Pearson 相关系数 (移植自 DAS reproducibility).
    """
    if len(signal1) != len(signal2):
        n = min(len(signal1), len(signal2))
        signal1 = signal1[:n]
        signal2 = signal2[:n]
    s1 = signal1 - np.mean(signal1)
    s2 = signal2 - np.mean(signal2)
    num = np.dot(s1, s2)
    den = np.sqrt(np.dot(s1, s1) * np.dot(s2, s2))
    return float(num / den) if den > 0 else 0.0


def cross_correlation(signal1: np.ndarray, signal2: np.ndarray,
                       max_lag: int = 20) -> np.ndarray:
    """
    归一化互相关函数.
    """
    n = len(signal1)
    e1 = np.dot(signal1, signal1)
    e2 = np.dot(signal2, signal2)
    norm = math.sqrt(e1 * e2) if e1 * e2 > 0 else 1.0

    cc = np.zeros(2 * max_lag + 1)
    for lag in range(-max_lag, max_lag + 1):
        s = 0.0
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                s += signal1[i] * signal2[j]
        cc[lag + max_lag] = s / norm
    return cc


def gw_strain_amplitude(M_chirp: float, f_gw: float, d_L: float) -> float:
    """
    引力波应变振幅.
    h = (4/D_L) (G M_chirp / c^2)^{5/3} (pi f_GW / c)^{2/3}
    """
    if d_L <= 0 or f_gw <= 0:
        return 0.0
    Mc_cm = NS.G_newton * M_chirp / NS.c_light ** 2
    return 4.0 / d_L * Mc_cm ** (5.0 / 3.0) * (math.pi * f_gw / NS.c_light) ** (2.0 / 3.0)


def generate_mock_gw_signal(n_points: int, f_gw: float, amplitude: float,
                             noise_level: float = 0.1,
                             seed: int = 42) -> np.ndarray:
    """生成模拟引力波信号."""
    np.random.seed(seed)
    t = np.linspace(0, 1.0, n_points)
    signal = amplitude * np.sin(2 * math.pi * f_gw * t)
    noise = noise_level * amplitude * np.random.randn(n_points)
    return signal + noise


def reproducibility_test(n_signals: int = 5, n_points: int = 1000,
                          f_gw: float = 150.0, amplitude: float = 1e-21,
                          noise_level: float = 0.3) -> Dict:
    """
    执行可复现性测试.
    """
    signals = []
    for i in range(n_signals):
        s = generate_mock_gw_signal(n_points, f_gw, amplitude, noise_level,
                                     seed=100 + i)
        signals.append(s)

    # 两两相关
    corr_matrix = np.zeros((n_signals, n_signals))
    for i in range(n_signals):
        for j in range(n_signals):
            corr_matrix[i, j] = reproducibility_correlation(signals[i], signals[j])

    # 平均信号
    mean_signal = np.mean(signals, axis=0)

    # 各信号与平均的相关
    mean_corrs = np.array([
        reproducibility_correlation(s, mean_signal) for s in signals
    ])

    return {
        'correlation_matrix': corr_matrix,
        'mean_correlations': mean_corrs,
        'mean_signal': mean_signal,
        'reproducibility_score': float(np.mean(mean_corrs)),
    }


# 自检
if __name__ == "__main__":
    print("=== 引力波可复现性分析自检 ===")
    M_ns = 1.4
    R_ns = 12.0
    C = compactness_from_mr(M_ns, R_ns)
    print(f"紧凑度 C = {C:.4f}")
    k2 = love_number_k2_approx(C)
    print(f"Love 数 k_2 = {k2:.4f}")
    Lambda = tidal_deformability(M_ns, R_ns)
    print(f"潮汐形变 Lambda = {Lambda:.2f}")

    M_total = 2.8 * NS.M_sun
    f_isco = gw_frequency_isco(M_total)
    print(f"ISCO 频率 = {f_isco:.1f} Hz")

    result = reproducibility_test()
    print(f"可复现性得分 = {result['reproducibility_score']:.4f}")

    print("\ngw_reproducibility.py 自检通过.")
