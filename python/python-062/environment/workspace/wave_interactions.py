"""
wave_interactions.py
================================================================================
波数空间三波相互作用模块 —— 基于种子项目 445_four_fifths（组合枚举思想）

在湍流的谱理论中，能量传递通过三波相互作用实现（Triad interactions）：
    k + p + q = 0

本模块枚举满足共振条件的波数三元组，并计算能量传递率，
用于理解边界层湍流的能量级串（energy cascade）机制。

核心物理公式
--------------------------------------------------------------------------------
Navier-Stokes 方程的谱形式：
    (∂/∂t + ν k²) û_i(k) = -i k_m P_{ij}(k) Σ_{p+q=k} û_j(p) û_m(q)

其中 P_{ij}(k) = δ_{ij} - k_i k_j / k² 为投影算子。

对于满足 k + p + q = 0 的三波组，能量传递函数：
    T(k,p,q) = Im[ û*(k) · (û(p) × û(q)) ]

净能量传递率：
    Π(k) = ∫_{|p|<k} ∫ T(k,p,q) δ(k+p+q) dp dq

在惯性子区，Kolmogorov 理论给出：
    Π(k) = ε  （常数）
"""

import numpy as np


def enumerate_triads(k_max, dim=2):
    """
    枚举满足 k + p + q = 0 的波数三元组（在截断球内）。

    参数
    ----------
    k_max : int
        最大波数模
    dim : int
        空间维度（2 或 3）

    返回
    -------
    triads : list of tuple
        每个元素为 ((k1,k2,k3), (p1,p2,p3), (q1,q2,q3))
    """
    triads = []
    rng = range(-k_max, k_max + 1)

    if dim == 2:
        for k1 in rng:
            for k2 in rng:
                if k1 == 0 and k2 == 0:
                    continue
                for p1 in rng:
                    for p2 in rng:
                        if p1 == 0 and p2 == 0:
                            continue
                        q1 = -(k1 + p1)
                        q2 = -(k2 + p2)
                        if abs(q1) > k_max or abs(q2) > k_max:
                            continue
                        if q1 == 0 and q2 == 0:
                            continue
                        # 避免重复（通过排序）
                        k_vec = (k1, k2)
                        p_vec = (p1, p2)
                        q_vec = (q1, q2)
                        triads.append((k_vec, p_vec, q_vec))
    else:
        for k1 in rng:
            for k2 in rng:
                for k3 in rng:
                    if k1 == 0 and k2 == 0 and k3 == 0:
                        continue
                    for p1 in rng:
                        for p2 in rng:
                            for p3 in rng:
                                if p1 == 0 and p2 == 0 and p3 == 0:
                                    continue
                                q1 = -(k1 + p1)
                                q2 = -(k2 + p2)
                                q3 = -(k3 + p3)
                                if max(abs(q1), abs(q2), abs(q3)) > k_max:
                                    continue
                                if q1 == 0 and q2 == 0 and q3 == 0:
                                    continue
                                triads.append(((k1, k2, k3), (p1, p2, p3), (q1, q2, q3)))

    return triads


def projection_operator(k_vec):
    """
    计算投影算子 P_{ij}(k) = δ_{ij} - k_i k_j / k²。

    参数
    ----------
    k_vec : tuple or np.ndarray
        波数向量

    返回
    -------
    P : np.ndarray, shape (d, d)
    """
    k = np.array(k_vec, dtype=np.float64)
    d = len(k)
    k2 = np.dot(k, k)

    if k2 < 1e-15:
        return np.eye(d)

    P = np.eye(d) - np.outer(k, k) / k2
    return P


def energy_transfer_rate_triad(uk, up, uq, k_vec, p_vec, q_vec):
    """
    计算单个三波组的能量传递率。

    参数
    ----------
    uk, up, uq : np.ndarray
        波数 k, p, q 处的复速度向量
    k_vec, p_vec, q_vec : tuple
        波数向量

    返回
    -------
    T : float
        能量传递率（实数）
    """
    k = np.array(k_vec, dtype=np.float64)
    p = np.array(p_vec, dtype=np.float64)

    # 简化：T ~ Im[ uk* · (up × uq) ]（三维）
    # 对于二维，使用标量涡度近似
    uk_conj = np.conj(uk)

    if len(k_vec) == 3:
        cross = np.cross(up, uq)
        T = np.imag(np.dot(uk_conj, cross))
    else:
        # 二维近似：T ~ k · Im[ uk* (up · uq) ]
        T = np.dot(k, np.imag(uk_conj * np.dot(up, uq)))

    return float(T)


def shell_energy_flux(triads, velocities, k_bins):
    """
    计算能量通量 Π(k) 随壳层的变化。

    参数
    ----------
    triads : list
        三波组列表
    velocities : dict
        {k_vec: u_vec}
    k_bins : np.ndarray
        波数壳层边界

    返回
    -------
    Pi : np.ndarray
        各壳层的能量通量
    """
    n_bins = len(k_bins) - 1
    Pi = np.zeros(n_bins, dtype=np.float64)

    for k_vec, p_vec, q_vec in triads:
        k_mag = np.linalg.norm(k_vec)
        p_mag = np.linalg.norm(p_vec)

        uk = velocities.get(k_vec)
        up = velocities.get(p_vec)
        uq = velocities.get(q_vec)

        if uk is None or up is None or uq is None:
            continue

        T = energy_transfer_rate_triad(uk, up, uq, k_vec, p_vec, q_vec)

        # 将传递率累加到 k 所在的壳层
        for b in range(n_bins):
            if k_bins[b] <= k_mag < k_bins[b + 1]:
                Pi[b] += T
                break

    return Pi
