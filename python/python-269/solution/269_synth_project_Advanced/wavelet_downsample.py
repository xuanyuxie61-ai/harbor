"""
wavelet_downsample.py — 波函数多尺度分析与降采样
============================================================

本模块实现波函数的多尺度分析, 用于:

1. 识别波函数的局域化/扩展特性
    - 参与比 (Participation Ratio): PR = |Σ|ψ|²|² / Σ|ψ|⁴
    - PR ~ N: 扩展态 (遍历全系统)
    - PR ~ 1: 强局域态

2. 多尺度降采样 (源自 image_decimate)
    - 粗粒化波函数: ψ_coarse(x) = Σ_{cell} w_i ψ(x_i)
    - 用于提取大尺度特征, 消除小尺度涨落

3. 逆参与比 (IPR) 分析
    - IPR = Σ|ψ|⁴ / (Σ|ψ|²)²
    - IPR ~ 1/N: 扩展态
    - IPR ~ 1: 局域态

4. 小波分析 (Haar 小波)
    - 多分辨率分解: ψ = Σ c_j φ_j + Σ d_j ψ_j
    - 系数分布揭示波函数的分形特性

在量子霍尔效应中:
    - 朗道能级体态: 扩展 (但有边界局域化)
    - 杂质态: 局域 (Anderson 局域化)
    - 边缘态: 准一维扩展

参考文献:
    [1] Wegman, A. "The Quantum Hall Effect" Ch. 4 (1988)
    [2] Evers, F. & Mirlin, A. D. Rev. Mod. Phys. 80, 1355 (2008)
"""

import numpy as np
from typing import Tuple, Dict, Any, List


def participation_ratio(psi: np.ndarray) -> float:
    """计算参与比 (Participation Ratio)

    PR = (Σ_n |ψ_n|²)² / (Σ_n |ψ_n|⁴)

    物理含义:
        PR 衡量波函数在格点上的有效展布范围.
        - 完全局域在一个格点: PR = 1
        - 均匀分布在 N 个格点: PR = N
        - 分形态: PR ~ N^D₂ (D₂ 是分形维数)

    对于二维量子霍尔系统:
        体扩展态: PR ~ L² (系统面积)
        边缘态: PR ~ L (边界长度)
        局域态: PR ~ ξ² (局域化长度的平方)

    Args:
        psi: 波函数 (一维数组)
    Returns:
        参与比
    """
    prob = np.abs(psi) ** 2
    norm_sq = np.sum(prob)
    if norm_sq < 1e-30:
        return 0.0
    prob_normalized = prob / norm_sq
    ipr = np.sum(prob_normalized ** 2)
    return 1.0 / ipr if ipr > 0 else 0.0


def inverse_participation_ratio(psi: np.ndarray) -> float:
    """计算逆参与比 (IPR)

    IPR = Σ_n |ψ_n|⁴ / (Σ_n |ψ_n|²)²

    IPR 是 PR 的倒数, 在文献中更常用.

    标度行为:
        IPR ~ L^{-D₂} (对于分形维数 D₂ 的态)
        对于 d 维系统的扩展态: D₂ = d, IPR ~ L^{-d}
        对于局域态: D₂ = 0, IPR → const

    在 Anderson 转变点 (量子霍尔平台转变):
        波函数具有多分形特性: D₂ ≈ 1.5 (对于 d=2)

    Args:
        psi: 波函数
    Returns:
        逆参与比
    """
    prob = np.abs(psi) ** 2
    norm_sq = np.sum(prob)
    if norm_sq < 1e-30:
        return 0.0
    prob_normalized = prob / norm_sq
    return np.sum(prob_normalized ** 2)


def downsample_2d(psi_2d: np.ndarray, factor: int = 2) -> np.ndarray:
    """二维波函数降采样 (源自 image_decimate_gray)

    将波函数从 N×N 格点降到 (N/factor)×(N/factor) 格点.
    使用块平均:
        ψ_coarse(i,j) = (1/f²) Σ_{block} ψ(i·f+di, j·f+dj)

    降采样保留了波函数的大尺度包络, 过滤掉小尺度振荡.
    对于朗道能级波函数, 降采样后可以更清楚地看到导向中心的位置.

    Args:
        psi_2d: 二维波函数数组
        factor: 降采样因子 (默认 2)
    Returns:
        降采样后的波函数
    """
    Ny, Nx = psi_2d.shape
    Ny_new = Ny // factor
    Nx_new = Nx // factor

    psi_coarse = np.zeros((Ny_new, Nx_new), dtype=psi_2d.dtype)
    for i in range(Ny_new):
        for j in range(Nx_new):
            block = psi_2d[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            psi_coarse[i, j] = np.mean(block)

    # 重新归一化
    norm = np.sqrt(np.sum(np.abs(psi_coarse)**2))
    if norm > 1e-30:
        psi_coarse /= norm

    return psi_coarse


def haar_wavelet_decompose(signal: np.ndarray,
                            levels: int = 3) -> List[np.ndarray]:
    """Haar 小波多分辨率分解

    Haar 小波是最简单的小波基:
        φ(t) = 1 for 0 ≤ t < 1 (尺度函数)
        ψ(t) = 1 for 0 ≤ t < 1/2, -1 for 1/2 ≤ t < 1 (小波函数)

    分解:
        Level 0: c_0 = 平均 (1个系数)
        Level j: d_j = 细节 (2^j 个系数)

    对于量子霍尔波函数:
        低频分量 → 大尺度包络 (体特性)
        高频分量 → 小尺度振荡 (杂质散射)

    Args:
        signal: 输入信号 (长度必须是 2 的幂)
        levels: 分解层数
    Returns:
        各层系数列表 [c_0, d_1, d_2, ..., d_L]
    """
    n = len(signal)
    # 确保长度是 2 的幂
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    sig = np.zeros(n_padded, dtype=signal.dtype)
    sig[:n] = signal

    coeffs = []
    current = sig.copy()

    for level in range(levels):
        length = len(current) // 2
        if length < 1:
            break
        approx = np.zeros(length, dtype=signal.dtype)
        detail = np.zeros(length, dtype=signal.dtype)
        for i in range(length):
            approx[i] = (current[2*i] + current[2*i+1]) / np.sqrt(2)
            detail[i] = (current[2*i] - current[2*i+1]) / np.sqrt(2)
        coeffs.append(detail)
        current = approx

    coeffs.append(current)  # 最后一级近似
    coeffs.reverse()
    return coeffs


def multifractal_spectrum(psi: np.ndarray, q_values: np.ndarray,
                          box_sizes: np.ndarray) -> Dict[str, np.ndarray]:
    """计算多分形谱 f(α)

    对于波函数测度 μ_i = |ψ_i|²:
        τ(q) = lim_{ε→0} (1/(q-1)) log(Σ_i μ_i^q) / log(ε)
        α = dτ/dq
        f(α) = qα - τ(q)

    在 Anderson 转变点:
        τ(q) 是非线性的 → 多分形
        D_q = τ(q)/(q-1) 随 q 变化

    Args:
        psi: 波函数 (一维)
        q_values: q 值数组
        box_sizes: 盒子尺寸数组
    Returns:
        多分形分析结果
    """
    prob = np.abs(psi) ** 2
    prob = prob / np.sum(prob)  # 归一化
    N = len(prob)

    tau_q = np.zeros(len(q_values))

    for iq, q in enumerate(q_values):
        chi_q = []
        for eps in box_sizes:
            n_boxes = max(1, N // eps)
            chi = 0.0
            for b in range(n_boxes):
                start = b * eps
                end = min(start + eps, N)
                mu_box = np.sum(prob[start:end])
                if mu_box > 0:
                    chi += mu_box ** q
            chi_q.append(chi)

        chi_q = np.array(chi_q)
        valid = chi_q > 0
        if np.sum(valid) >= 2:
            log_chi = np.log(chi_q[valid])
            log_eps = np.log(box_sizes[valid])
            slope, _ = np.polyfit(log_eps, log_chi, 1)
            tau_q[iq] = slope / (q - 1) if abs(q - 1) > 1e-10 else slope
        else:
            tau_q[iq] = 0.0

    return {
        'q': q_values,
        'tau_q': tau_q,
        'D_q': tau_q / (q_values - 1.0 + 1e-10),
    }


def localize_length(psi: np.ndarray, x: np.ndarray) -> float:
    """估计波函数的局域化长度 ξ

    通过拟合指数衰减:
        |ψ(x)|² ~ exp(-2|x-x₀|/ξ)
    → log|ψ|² ~ -2|x-x₀|/ξ
    → ξ = -2 / slope

    对于扩展态: ξ → ∞
    对于局域态: ξ < L (系统尺寸)

    Args:
        psi: 波函数
        x: 坐标数组
    Returns:
        局域化长度估计
    """
    prob = np.abs(psi) ** 2
    prob = prob / np.max(prob)  # 归一化到峰值

    # 找到峰值位置
    peak_idx = np.argmax(prob)

    # 在峰值两侧拟合
    left_mask = (x < x[peak_idx]) & (prob > 1e-10)
    right_mask = (x > x[peak_idx]) & (prob > 1e-10)

    xi_left = float('inf')
    xi_right = float('inf')

    if np.sum(left_mask) > 3:
        log_prob = np.log(prob[left_mask])
        slope, _ = np.polyfit(x[left_mask], log_prob, 1)
        if slope > 0:  # 左侧应该正斜率
            xi_left = 2.0 / slope

    if np.sum(right_mask) > 3:
        log_prob = np.log(prob[right_mask])
        slope, _ = np.polyfit(x[right_mask], log_prob, 1)
        if slope < 0:  # 右侧应该负斜率
            xi_right = -2.0 / slope

    return min(xi_left, xi_right)
