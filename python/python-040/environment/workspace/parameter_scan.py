#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameter_scan.py
BSM 参数空间扫描与优化选择模块

融合原项目:
- 628_knapsack_values: 背包问题（分析通道优化选择）

在BSM信号分析中用于:
- 在有限的积分光度下，优化选择对 Z' 信号最灵敏的分析通道组合
- 计算各参数点的预期信号产额与显著性
"""

import numpy as np
from typing import List, Tuple, Optional


def knapsack_channel_selection(
    signal_yields: np.ndarray,
    background_yields: np.ndarray,
    luminosities: np.ndarray,
    max_lumi: float
) -> Tuple[float, np.ndarray]:
    """
    使用背包问题思想优化分析通道选择。

    物理问题: 在总积分光度预算 L_max 下，
    选择哪些衰变道（ee, μμ, ττ, jj, bb, tt）进行分析
    以使总信号显著性最大。

    背包建模:
        - 物品 i = 分析通道 i
        - 重量 w_i = 所需积分光度 L_i [fb^{-1}]
        - 价值 v_i = 该通道的期望显著性 s_i = S_i / sqrt(B_i)
        - 容量 W = L_max

    动态规划求解 0/1 背包问题：
        dp[j] = max(dp[j], dp[j - w_i] + v_i)

    Parameters
    ----------
    signal_yields : np.ndarray
        各通道信号产额预期
    background_yields : np.ndarray
        各通道背景产额预期
    luminosities : np.ndarray
        各通道所需积分光度 [fb^{-1}]
    max_lumi : float
        总积分光度预算

    Returns
    -------
    total_significance : float
        最优组合的总显著性
    selected : np.ndarray
        布尔数组，True 表示选择该通道
    """
    n = signal_yields.size
    if n == 0:
        return 0.0, np.array([], dtype=bool)

    # 计算各通道价值（显著性）
    significances = np.zeros(n)
    for i in range(n):
        b = max(background_yields[i], 1.0)
        s = max(signal_yields[i], 0.0)
        significances[i] = s / np.sqrt(b)

    # 离散化重量（积分光度取整到 1 fb^{-1}）
    weights = np.maximum(np.round(luminosities).astype(int), 1)
    capacity = max(int(np.round(max_lumi)), 1)

    # 动态规划
    dp = np.zeros(capacity + 1)
    choice = np.full((n, capacity + 1), -1, dtype=int)

    for i in range(n):
        w = weights[i]
        v = significances[i]
        for j in range(capacity, w - 1, -1):
            if dp[j - w] + v > dp[j]:
                dp[j] = dp[j - w] + v
                choice[i, j] = j - w

    # 回溯找最优解
    selected = np.zeros(n, dtype=bool)
    j = capacity
    for i in range(n - 1, -1, -1):
        if choice[i, j] >= 0:
            selected[i] = True
            j = choice[i, j]

    total_sig = dp[capacity]
    return total_sig, selected


def smolyak_parameter_scan(
    mass_range: Tuple[float, float],
    coupling_range: Tuple[float, float],
    width_range: Tuple[float, float],
    max_level: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Smolyak 稀疏网格在三维 BSM 参数空间采样。

    参数空间: (M_{Z'}, g_q, Γ_{Z'} / M_{Z'})

    Smolyak 层级 ℓ = (ℓ1, ℓ2, ℓ3)，约束 |ℓ|_1 ≤ q。
    对于 q = max_level + 2，形成稀疏网格。

    相比全张量积网格，Smolyak 网格的节点数从 O(N^d) 降为
    O(N (log N)^{d-1})，同时保持相近的代数精度。

    Parameters
    ----------
    mass_range : Tuple[float, float]
        质量范围 [GeV]
    coupling_range : Tuple[float, float]
        耦合范围
    width_range : Tuple[float, float]
        相对宽度范围
    max_level : int
        最大层级

    Returns
    -------
    mass_points : np.ndarray
        质量采样点
    coupling_points : np.ndarray
        耦合采样点
    width_points : np.ndarray
        宽度采样点
    """
    # 简化的 Smolyak 网格：各维独立采样后取子集
    try:
        from interpolation_utils import order_from_level_135, cc_compute_points
    except ImportError:
        from .interpolation_utils import order_from_level_135, cc_compute_points

    levels = list(range(max_level + 1))
    all_mass = []
    all_coupling = []
    all_width = []

    for lm in levels:
        n_m = order_from_level_135(lm)
        pts_m = cc_compute_points(n_m)
        pts_m = 0.5 * ((1.0 - pts_m) * mass_range[0] + (1.0 + pts_m) * mass_range[1])

        for lc in levels:
            n_c = order_from_level_135(lc)
            pts_c = cc_compute_points(n_c)
            pts_c = 0.5 * ((1.0 - pts_c) * coupling_range[0] + (1.0 + pts_c) * coupling_range[1])

            for lw in levels:
                # Smolyak 层级约束
                if lm + lc + lw > max_level + 2:
                    continue

                n_w = order_from_level_135(lw)
                pts_w = cc_compute_points(n_w)
                pts_w = 0.5 * ((1.0 - pts_w) * width_range[0] + (1.0 + pts_w) * width_range[1])

                for m in pts_m:
                    for c in pts_c:
                        for w in pts_w:
                            all_mass.append(m)
                            all_coupling.append(c)
                            all_width.append(w)

    return np.array(all_mass), np.array(all_coupling), np.array(all_width)


def expected_signal_yield(
    cross_section: float,
    luminosity: float,
    efficiency: float = 0.5,
    branching_ratio: float = 0.1
) -> float:
    """
    计算预期信号产额。

        N_sig = σ × L × ε × BR

    其中:
        σ = 产生截面 [pb]
        L = 积分光度 [fb^{-1}] = 1000 pb^{-1}
        ε = 分析选择效率
        BR = 分支比

    Parameters
    ----------
    cross_section : float
        产生截面 [pb]
    luminosity : float
        积分光度 [fb^{-1}]
    efficiency : float
        选择效率
    branching_ratio : float
        分支比

    Returns
    -------
    float
        预期信号事件数
    """
    if cross_section < 0.0 or luminosity < 0.0:
        return 0.0
    # 单位转换: fb^{-1} → pb^{-1} (1 fb = 1000 pb)
    lumi_pb = luminosity * 1000.0
    eff = np.clip(efficiency, 0.0, 1.0)
    br = np.clip(branching_ratio, 0.0, 1.0)
    return cross_section * lumi_pb * eff * br


def exclusion_contour_2d(
    mass_grid: np.ndarray,
    coupling_grid: np.ndarray,
    significance_grid: np.ndarray,
    cl_threshold: float = 1.96
) -> List[Tuple[float, float]]:
    """
    在 (M_{Z'}, g_q) 平面上计算 95% CL 排除等高线。

    使用 Marching Squares 算法的简化版：
    对每个网格单元，检查四个角点的显著性是否跨越阈值，
    若是，用线性插值估计等高线通过位置。

    Parameters
    ----------
    mass_grid : np.ndarray
        质量轴采样点
    coupling_grid : np.ndarray
        耦合轴采样点
    significance_grid : np.ndarray
        显著性矩阵，形状 (n_mass, n_coupling)
    cl_threshold : float
        1.96 对应 95% CL（单边）

    Returns
    -------
    List[Tuple[float, float]]
        等高线上的点 (mass, coupling)
    """
    contour_points = []
    nm = len(mass_grid)
    nc = len(coupling_grid)

    for i in range(nm - 1):
        for j in range(nc - 1):
            # 四个角点
            vals = [
                significance_grid[i, j],
                significance_grid[i + 1, j],
                significance_grid[i, j + 1],
                significance_grid[i + 1, j + 1]
            ]
            above = [v >= cl_threshold for v in vals]
            n_above = sum(above)

            if n_above > 0 and n_above < 4:
                # 跨越等高线，取中心点作为近似
                m_mid = (mass_grid[i] + mass_grid[i + 1]) / 2.0
                c_mid = (coupling_grid[j] + coupling_grid[j + 1]) / 2.0
                contour_points.append((m_mid, c_mid))

    return contour_points


def discovery_potential(
    signal_cross_sections: np.ndarray,
    background_cross_sections: np.ndarray,
    luminosities: np.ndarray,
    systematic_errors: np.ndarray
) -> np.ndarray:
    """
    计算各参数点的发现潜力（Z 值）。

    简化的 Asimov 显著性:
        Z = sqrt(2 ((s + b) ln(1 + s/b) - s))

    若 s << b，近似为:
        Z ≈ s / sqrt(b + (ε b)^2)

    Parameters
    ----------
    signal_cross_sections : np.ndarray
        信号截面 [pb]
    background_cross_sections : np.ndarray
        背景截面 [pb]
    luminosities : np.ndarray
        积分光度 [fb^{-1}]
    systematic_errors : np.ndarray
        相对系统误差

    Returns
    -------
    np.ndarray
        各点的 Z 值
    """
    s = np.maximum(signal_cross_sections * luminosities * 1000.0, 0.0)
    b = np.maximum(background_cross_sections * luminosities * 1000.0, 1.0)
    epsilon = np.clip(systematic_errors, 0.0, 1.0)

    # 避免数值问题
    ratio = s / b
    ratio = np.clip(ratio, 1e-10, 1e6)

    z = np.sqrt(2.0 * ((s + b) * np.log(1.0 + ratio) - s))

    # 系统误差修正
    sigma_b = epsilon * b
    denom = np.sqrt(b + sigma_b ** 2)
    z_approx = s / denom

    # 取两者较小者作为保守估计
    z = np.minimum(z, z_approx)

    return z
