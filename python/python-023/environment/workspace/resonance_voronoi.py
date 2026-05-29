#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
共振粒子检测与Voronoi邻域搜索
================================================================================

基于 1398_voronoi_plot 的Voronoi距离搜索思想，检测满足回旋共振条件的
粒子，并在速度空间构建Voronoi邻域以识别共振粒子团簇。

核心物理模型：

回旋共振条件（Doppler-shifted cyclotron resonance）：
    ω_k - k_∥ v_∥ - n Ω_e / γ = 0

对于 whistler 模与电子（n = -1）：
    v_∥,res = (ω_k + Ω_e / γ) / k_∥

Voronoi图定义：
    给定速度空间中的种子点集 {𝐯_i}，每个种子点的Voronoi单元为：
        C_i = { 𝐯 ∈ ℝ² : ||𝐯 - 𝐯_i|| ≤ ||𝐯 - 𝐯_j||, ∀j≠i }

最近邻搜索：
    对于给定的共振速度 𝐯_res，找到最近的粒子 𝐯_i。
    这等价于找到包含 𝐯_res 的Voronoi单元。

数值方法：
- 暴力最近邻搜索（适用于粒子数 < 10^4）
- 距离使用p-范数：||𝐯||_p = (|v_∥|^p + |v_⊥|^p)^{1/p}
  p=2: Euclidean, p=1: Manhattan, p=∞: Chebyshev
================================================================================
"""

import numpy as np


def detect_resonant_particles(v_grid, omega_solutions, params,
                               resonance_width=0.15, p_norm=2):
    """
    检测满足回旋共振条件的粒子。
    
    参数
    ----
    v_grid : ndarray, shape (N, 3)
        粒子速度向量 [m/s]。
    omega_solutions : ndarray, shape (Nk, 2)
        (k, ω) 色散解。
    params : dict
        物理参数。
    resonance_width : float
        共振宽度（以热速度为单位）。
    p_norm : int or float
        距离范数类型。
        
    返回
    ----
    resonant_indices : list of int
        共振粒子的索引列表。
    """
    N = v_grid.shape[0]
    Omega_e = params['Omega_e']
    v_te = params['v_te']
    c = params['c']
    
    resonant_indices = []
    
    for i in range(N):
        v = v_grid[i]
        v_parallel = v[2]  # 假设z方向为平行方向
        v_perp = np.sqrt(v[0]**2 + v[1]**2)
        
        # 洛伦兹因子
        v_sq = np.sum(v**2)
        gamma = 1.0 / np.sqrt(max(1.0 - v_sq / c**2, 1e-10))
        gamma = min(gamma, 100.0)
        
        is_resonant = False
        
        # 检查每支波模式
        for k_omega in omega_solutions:
            k = k_omega[0]
            omega = complex(k_omega[1])
            omega_r = omega.real
            
            if omega_r <= 0 or np.abs(k) < 1e-20:
                continue
            
            # 共振速度
            v_res = (omega_r + Omega_e / gamma) / k
            
            # 检查是否在该波模式的共振宽度内
            delta_v = np.abs(v_parallel - v_res)
            
            # 使用p-范数判断
            if p_norm == 2:
                dist = delta_v
            elif p_norm == 1:
                dist = delta_v + 0.1 * v_perp
            elif p_norm == np.inf:
                dist = max(delta_v, 0.1 * v_perp)
            else:
                dist = (delta_v**p_norm + (0.1 * v_perp)**p_norm)**(1.0 / p_norm)
            
            if dist < resonance_width * v_te:
                is_resonant = True
                break
        
        if is_resonant:
            resonant_indices.append(i)
    
    return resonant_indices


def voronoi_nearest_neighbor(query_points, centers, p_norm=2):
    """
    对查询点进行Voronoi最近邻搜索。
    
    参数
    ----
    query_points : ndarray, shape (Nq, D)
        查询点。
    centers : ndarray, shape (Nc, D)
        Voronoi种子点（粒子位置）。
    p_norm : int or float
        距离范数。
        
    返回
    ----
    nearest_idx : ndarray, shape (Nq,)
        每个查询点对应的最近邻种子索引。
    min_dist : ndarray, shape (Nq,)
        最小距离。
    """
    Nq = query_points.shape[0]
    Nc = centers.shape[0]
    
    nearest_idx = np.zeros(Nq, dtype=int)
    min_dist = np.zeros(Nq)
    
    for i in range(Nq):
        q = query_points[i]
        
        # 计算到所有中心的距离
        diff = centers - q
        
        if p_norm == 2:
            dists = np.sum(diff**2, axis=1)
        elif p_norm == 1:
            dists = np.sum(np.abs(diff), axis=1)
        elif p_norm == np.inf:
            dists = np.max(np.abs(diff), axis=1)
        else:
            dists = np.sum(np.abs(diff)**p_norm, axis=1)**(1.0 / p_norm)
        
        nearest_idx[i] = np.argmin(dists)
        min_dist[i] = dists[nearest_idx[i]]
    
    return nearest_idx, min_dist


def compute_resonance_region_volume(v_parallel, v_perp, omega_solutions, params):
    """
    计算速度空间中共振区域的体积。
    
    使用Monte Carlo积分：
        V_res = ∫∫ I_res(v_∥, v_⊥) dv_∥ dv_⊥
    
    其中 I_res 为共振指示函数。
    """
    n_samples = 2000
    v_max = 3.0 * params['v_te']
    
    samples = np.random.uniform(-v_max, v_max, (n_samples, 3))
    samples[:, 0] = np.abs(samples[:, 0])  # v_perp ≥ 0
    
    resonant = detect_resonant_particles(samples, omega_solutions, params)
    
    fraction = len(resonant) / n_samples
    total_volume = (2 * v_max) * (2 * v_max) * v_max  # v_perp > 0 的半空间
    
    return fraction * total_volume
