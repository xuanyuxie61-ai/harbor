# -*- coding: utf-8 -*-
"""
optimal_sampling.py
-------------------
基于 Centroidal Voronoi Tessellation with Mirror-Periodic (CVTM) 的
最优动量空间采样算法。

对应种子项目：263_cvtm_1d
核心算法：
  - Lloyd 松弛 / MacQueen 方法的采样版本
  - 镜像周期边界条件处理
  - 能量泛函最小化

在超导问题中用于：对一维费米面弧段进行均匀覆盖采样，
保证布里渊区高对称路径上的 k 点分布最优。
"""

import numpy as np


def cvtm_1d(g_num, it_num, s_num, seed=None):
    """
    一维镜像周期 CVT 采样器。

    在区间 [0, 1] 上生成 g_num 个生成元，通过 Lloyd 松弛
    使得 Voronoi 单元质心等于生成元自身。
    周期边界通过镜像 [-1,0] 和 [1,2] 实现。

    能量泛函：
        E = Σ_i Σ_{x∈V_i} |x - g_i|^2
    其中 V_i 为第 i 个 Voronoi 单元。

    Parameters
    ----------
    g_num : int
        生成元数量，必须 >=1。
    it_num : int
        Lloyd 迭代次数。
    s_num : int
        每轮采样点数。
    seed : int, optional
        随机种子。

    Returns
    -------
    generators : ndarray, shape (g_num,)
        最终生成元位置（已排序）。
    energy_history : ndarray, shape (it_num,)
        每轮总能量。
    motion_history : ndarray, shape (it_num,)
        每轮平均移动量。
    """
    if g_num < 1:
        raise ValueError("g_num 必须 >= 1。")
    if it_num < 0:
        it_num = 0
    if s_num < 1:
        s_num = 1
    if seed is not None:
        np.random.seed(seed)

    # 初始化生成元
    g = np.sort(np.random.rand(g_num))
    energy_history = np.zeros(it_num)
    motion_history = np.zeros(it_num)

    for it in range(it_num):
        s = np.random.rand(s_num)
        # 镜像样本
        sa = -s
        sb = 2.0 - s

        # 为每个样本找到最近生成元（含镜像）
        # 计算到真实、左镜像、右镜像的距离
        # 通过广播: s 形状 (s_num,1), g 形状 (g_num,)
        d_real = np.abs(s[:, np.newaxis] - g[np.newaxis, :])
        d_left = np.abs(sa[:, np.newaxis] - g[np.newaxis, :])
        d_right = np.abs(sb[:, np.newaxis] - g[np.newaxis, :])

        d_all = np.stack([d_real, d_left, d_right], axis=2)
        min_idx = np.argmin(d_all, axis=2)
        min_val = np.min(d_all, axis=2)

        # 若镜像最近，则将样本映射回 [0,1]
        s_eff = s.copy()
        mask_left = min_idx[:, 0] == 1
        mask_right = min_idx[:, 0] == 2
        s_eff[mask_left] = 0.0
        s_eff[mask_right] = 1.0

        # 重新确定归属（映射后按真实距离）
        d_eff = np.abs(s_eff[:, np.newaxis] - g[np.newaxis, :])
        nearest = np.argmin(d_eff, axis=1)

        # 累加
        g_new = np.zeros(g_num)
        w_new = np.zeros(g_num)
        e_new = np.zeros(g_num)

        for i in range(g_num):
            mask = nearest == i
            if np.any(mask):
                g_new[i] = np.sum(s_eff[mask])
                w_new[i] = np.count_nonzero(mask)
                e_new[i] = np.sum(min_val[mask] ** 2)
            else:
                g_new[i] = g[i]
                w_new[i] = 0
                e_new[i] = 0.0

        # 更新为质心
        with np.errstate(divide='ignore', invalid='ignore'):
            g_new = np.where(w_new > 0, g_new / w_new, g)
        g_new = np.clip(g_new, 0.0, 1.0)
        g_new = np.sort(g_new)

        motion = np.mean((g_new - g) ** 2)
        energy = np.sum(e_new)
        energy_history[it] = energy
        motion_history[it] = motion
        g = g_new

    return g, energy_history, motion_history


def optimal_k_path_sampling(k_min, k_max, n_points, it_num=50, s_num=5000):
    """
    对一维 k 路径 [k_min, k_max] 进行 CVT 最优采样。

    返回排序后的 k 点数组。
    """
    if k_min >= k_max:
        raise ValueError("k_min 必须 < k_max。")
    if n_points < 1:
        raise ValueError("n_points 必须 >= 1。")
    generators, _, _ = cvtm_1d(n_points, it_num, s_num)
    return k_min + generators * (k_max - k_min)
