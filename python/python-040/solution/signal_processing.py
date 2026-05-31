#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_processing.py
信号/背景判别与降维分析模块

融合原项目:
- 1187_svd_fingerprint: SVD 降维与低秩近似（指纹特征提取）

在BSM信号分析中用于:
- 对探测器径迹图像进行 SVD 主成分分析
- 提取区分信号（Z' → ℓ⁺ℓ⁻）与背景（Drell-Yan）的低维特征
- 计算奇异值谱的熵以量化事例复杂度
"""

import numpy as np
from typing import Tuple, List


def svd_low_rank_approximation(
    data_matrix: np.ndarray,
    rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对数据矩阵进行 SVD 分解并提取低秩近似。

    SVD 分解:
        A = U Σ V^T

    秩-r 近似:
        A_r = U_r Σ_r V_r^T = Σ_{i=1}^r σ_i u_i v_i^T

    Eckart-Young-Mirsky 定理保证 A_r 是在 Frobenius 范数下
    对 A 的最佳秩-r 近似。

    物理意义: 在 BSM 搜索中，信号事例通常在径迹图像上具有
    特定的低秩结构（两条清晰径迹），而背景（多喷注）具有
    更高的秩（更复杂）。

    Parameters
    ----------
    data_matrix : np.ndarray
        输入矩阵 A(M, N)
    rank : int
        截断秩 r

    Returns
    -------
    u_r : np.ndarray
        左奇异向量 U_r(M, r)
    s_r : np.ndarray
        奇异值 σ_i，长度 r
    vh_r : np.ndarray
        右奇异向量 V_r^T(r, N)
    approx : np.ndarray
        低秩近似矩阵 A_r(M, N)
    """
    m, n = data_matrix.shape
    rank = max(1, min(rank, min(m, n)))

    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    # 重建低秩近似
    approx = u_r @ np.diag(s_r) @ vh_r

    return u_r, s_r, vh_r, approx


def singular_value_entropy(singular_values: np.ndarray) -> float:
    """
    计算奇异值谱的归一化 Shannon 熵。

        S = - Σ p_i ln(p_i) / ln(r)

    其中 p_i = σ_i² / Σ σ_j² 是第 i 个主成分的能量占比。

    物理意义:
        - S ≈ 0: 低熵，信号结构简单（如 Z' → ℓ⁺ℓ⁻ 的两体衰变）
        - S ≈ 1: 高熵，信号结构复杂（如多体衰变或强子背景）

    Parameters
    ----------
    singular_values : np.ndarray
        奇异值数组

    Returns
    -------
    float
        归一化熵 S ∈ [0, 1]
    """
    s = np.asarray(singular_values)
    s = s[s > 1e-15]
    if s.size == 0:
        return 0.0

    p = s ** 2
    p_sum = np.sum(p)
    if p_sum < 1e-15:
        return 0.0

    p = p / p_sum
    # 只保留 p > 0 的项
    p = p[p > 1e-15]

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(s.size)

    if max_entropy < 1e-15:
        return 0.0

    return float(entropy / max_entropy)


def signal_background_discriminator(
    hit_maps: List[np.ndarray],
    labels: np.ndarray,
    n_components: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 SVD 主成分的信号/背景判别器训练。

    算法:
        1. 将所有事例的击中图向量化，组成设计矩阵 X(N_events, N_pixels)
        2. 对 X 进行 SVD: X = U Σ V^T
        3. 取前 n_components 个右奇异向量 V_k 作为基
        4. 将每个事例投影到该基上: c_i = V_k^T · x_i
        5. 使用投影系数的范数作为判别分数

    信号（两体衰变）通常在低维子空间中有较大投影，
    而背景（多喷注）投影分散。

    Parameters
    ----------
    hit_maps : List[np.ndarray]
        事例击中图列表
    labels : np.ndarray
        标签: 1 = 信号, 0 = 背景
    n_components : int
        保留的主成分数

    Returns
    -------
    basis : np.ndarray
        主成分基 V_k^T(n_components, N_pixels)
    scores : np.ndarray
        每个事例的判别分数
    """
    n_events = len(hit_maps)
    if n_events == 0:
        raise ValueError("没有输入事例")

    # 向量化
    pixel_size = hit_maps[0].size
    X = np.zeros((n_events, pixel_size))
    for i, hm in enumerate(hit_maps):
        X[i, :] = hm.ravel()

    # 中心化处理
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # SVD
    _, _, vh = np.linalg.svd(X_centered, full_matrices=False)
    basis = vh[:n_components, :]

    # 投影并计算分数（Frobenius 范数）
    projections = X_centered @ basis.T
    scores = np.linalg.norm(projections, axis=1)

    # 根据标签调整分数方向（信号分数应更高）
    mean_sig = np.mean(scores[labels == 1]) if np.any(labels == 1) else 0.0
    mean_bkg = np.mean(scores[labels == 0]) if np.any(labels == 0) else 1.0
    if mean_sig < mean_bkg:
        scores = -scores

    return basis, scores


def pca_denoise(
    data_matrix: np.ndarray,
    variance_threshold: float = 0.95
) -> np.ndarray:
    """
    基于 PCA 的噪声抑制。

    保留累计解释方差达到阈值的前 k 个主成分：
        Σ_{i=1}^k σ_i² / Σ_{j=1}^r σ_j² ≥ threshold

    Parameters
    ----------
    data_matrix : np.ndarray
        数据矩阵
    variance_threshold : float
        方差保留阈值

    Returns
    -------
    np.ndarray
        去噪后的数据
    """
    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    total_var = np.sum(s ** 2)
    if total_var < 1e-15:
        return data_matrix

    cumvar = np.cumsum(s ** 2) / total_var
    k = np.searchsorted(cumvar, variance_threshold) + 1
    k = min(k, s.size)

    u_k = u[:, :k]
    s_k = s[:k]
    vh_k = vh[:k, :]

    return u_k @ np.diag(s_k) @ vh_k


def resonance_peak_finder(
    invariant_mass: np.ndarray,
    counts: np.ndarray,
    window_width: float = 10.0
) -> Tuple[float, float, float]:
    """
    在一维不变质量谱中寻找共振峰。

    使用滑动窗口计算局部信噪比：
        S/B = (N_window - N_side) / sqrt(N_side)

    Parameters
    ----------
    invariant_mass : np.ndarray
        质量轴 [GeV]
    counts : np.ndarray
        每个质量区间的计数
    window_width : float
        搜索窗口半宽 [GeV]

    Returns
    -------
    peak_mass : float
        峰位质量 [GeV]
    peak_height : float
        峰高度（超出背景的计数）
    significance : float
        局部显著性
    """
    n = invariant_mass.size
    if n < 3:
        return 0.0, 0.0, 0.0

    best_sig = 0.0
    best_mass = invariant_mass[n // 2]
    best_height = 0.0

    for i in range(n):
        m_central = invariant_mass[i]
        # 信号窗口
        in_window = np.abs(invariant_mass - m_central) <= window_width
        # 侧边带（用于背景估计）
        in_sideband = (np.abs(invariant_mass - m_central) > window_width) & \
                      (np.abs(invariant_mass - m_central) <= 2.0 * window_width)

        n_window = np.sum(counts[in_window])
        n_side = np.sum(counts[in_sideband])
        n_side_bins = np.sum(in_sideband)

        if n_side_bins > 0 and n_side > 0:
            # 外推背景到窗口
            bkg_est = n_side * np.sum(in_window) / n_side_bins
            excess = max(n_window - bkg_est, 0.0)
            sig = excess / np.sqrt(max(bkg_est, 1.0))

            if sig > best_sig:
                best_sig = sig
                best_mass = m_central
                best_height = excess

    return best_mass, best_height, best_sig
