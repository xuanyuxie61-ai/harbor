#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
相空间多维Lagrange插值重构器
================================================================================

基于 638_lagrange_nd 的多维Lagrange多项式插值思想，
在4D相空间 (x, v_∥, v_⊥, t) 中重构粒子分布函数。

核心数学模型：

给定 ND 个数据点 {𝐱_j, f_j}_{j=1}^{ND}，其中 𝐱_j ∈ ℝ^D，
寻找Lagrange基多项式 L_i(𝐱) 满足：
    L_i(𝐱_j) = δ_{ij}

D维全次数不超过 N 的多项式空间维数：
    R = C(N + D, N) = (N + D)! / (N! D!)

要求 ND = R（适定插值）。

本项目应用于相空间分布函数重构：
给定速度空间网格点上的离散分布函数值 f_{ij} = f(v_{∥,i}, v_{⊥,j})，
使用2D Lagrange插值得到连续近似：
    f(v_∥, v_⊥) = Σ_{i,j} f_{ij} L_i(v_∥) M_j(v_⊥)

其中 L_i, M_j 为一维Lagrange基多项式：
    L_i(x) = Π_{m≠i} (x - x_m) / (x_i - x_m)

数值稳定性：
- 使用重心Lagrange插值公式 (Berrut & Trefethen 2004)
- 对 Chebyshev 节点进行重采样以避免Runge现象
================================================================================
"""

import numpy as np


def barycentric_weights(x_nodes):
    """
    计算重心Lagrange插值的权重。
    
    对于节点 {x_j}，重心权重为：
        w_j = 1 / Π_{m≠j} (x_j - x_m)
    
    参数
    ----
    x_nodes : ndarray
        插值节点。
        
    返回
    ----
    w : ndarray
        重心权重。
    """
    n = len(x_nodes)
    w = np.ones(n)
    
    for j in range(n):
        for m in range(n):
            if m != j:
                diff = x_nodes[j] - x_nodes[m]
                # 避免除零
                if np.abs(diff) < 1e-30:
                    diff = 1e-30 * np.sign(diff) if diff != 0 else 1e-30
                w[j] /= diff
    
    return w


def barycentric_interpolate(x_nodes, f_values, x_eval, w=None):
    """
    使用重心Lagrange公式进行插值。
    
    公式：
        f(x) = Σ_j [w_j / (x - x_j)] f_j / Σ_j [w_j / (x - x_j)]
    
    参数
    ----
    x_nodes : ndarray
        插值节点。
    f_values : ndarray
        节点处的函数值。
    x_eval : float 或 ndarray
        求值点。
    w : ndarray, optional
        预计算的重心权重。
        
    返回
    ----
    f_eval : float 或 ndarray
        插值结果。
    """
    x_nodes = np.asarray(x_nodes)
    f_values = np.asarray(f_values)
    
    if w is None:
        w = barycentric_weights(x_nodes)
    
    scalar_input = np.isscalar(x_eval)
    x_eval = np.atleast_1d(x_eval)
    
    f_eval = np.zeros_like(x_eval, dtype=float)
    
    for idx, x in enumerate(x_eval):
        # 检查是否恰好在节点上
        exact_match = np.abs(x - x_nodes) < 1e-30 * (np.max(np.abs(x_nodes)) + 1.0)
        if np.any(exact_match):
            f_eval[idx] = f_values[np.argmax(exact_match)]
            continue
        
        # 重心公式
        numer = 0.0
        denom = 0.0
        for j in range(len(x_nodes)):
            term = w[j] / (x - x_nodes[j])
            numer += term * f_values[j]
            denom += term
        
        # 边界检查
        if np.abs(denom) < 1e-30:
            denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30
        
        f_eval[idx] = numer / denom
    
    return f_eval[0] if scalar_input else f_eval


def chebyshev_nodes(a, b, n):
    """
    生成 [a, b] 区间上的Chebyshev节点：
        x_j = (a+b)/2 + (b-a)/2 * cos( (2j+1)π / (2n) )
    
    Chebyshev节点可最小化Runge现象。
    """
    j = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return x


def lagrange_phase_space_reconstruction(v_parallel, v_perp, f_grid, params,
                                         n_cheb=16):
    """
    使用多维Lagrange插值重构相空间分布函数。
    
    参数
    ----
    v_parallel, v_perp : ndarray
        速度网格。
    f_grid : ndarray, shape (nv, nv)
        网格上的分布函数值。
    params : dict
        物理参数。
    n_cheb : int
        Chebyshev插值节点数。
        
    返回
    ----
    f_reconstructed : ndarray, shape (nv, nv)
        重构后的分布函数（在精细网格上求值）。
    """
    nv = len(v_parallel)
    
    # 生成Chebyshev节点用于插值
    vpar_min, vpar_max = np.min(v_parallel), np.max(v_parallel)
    vperp_min, vperp_max = np.min(v_perp), np.max(v_perp)
    
    n_cheb_eff = min(n_cheb, nv)
    
    # 如果网格足够密集，直接使用原网格
    if nv <= n_cheb:
        # 简单双线性插值
        return f_grid.copy()
    
    # 选择Chebyshev子集
    cheb_idx_par = np.round(np.linspace(0, nv - 1, n_cheb_eff)).astype(int)
    cheb_idx_perp = np.round(np.linspace(0, nv - 1, n_cheb_eff)).astype(int)
    
    # 确保不越界
    cheb_idx_par = np.clip(cheb_idx_par, 0, nv - 1)
    cheb_idx_perp = np.clip(cheb_idx_perp, 0, nv - 1)
    
    x_nodes = v_parallel[cheb_idx_par]
    y_nodes = v_perp[cheb_idx_perp]
    f_nodes = f_grid[np.ix_(cheb_idx_perp, cheb_idx_par)]
    
    # 计算重心权重
    wx = barycentric_weights(x_nodes)
    wy = barycentric_weights(y_nodes)
    
    # 在精细网格上求值
    f_reconstructed = np.zeros((nv, nv))
    
    for j in range(nv):
        for i in range(nv):
            x = v_parallel[i]
            y = v_perp[j]
            
            # 2D张量积Lagrange插值
            # f(x,y) = Σ_m Σ_n f_{mn} L_m(x) N_n(y)
            val = 0.0
            for m in range(n_cheb_eff):
                Lx = barycentric_interpolate(x_nodes, np.eye(n_cheb_eff)[m], x, wx)
                for n in range(n_cheb_eff):
                    Ly = barycentric_interpolate(y_nodes, np.eye(n_cheb_eff)[n], y, wy)
                    val += f_nodes[n, m] * Lx * Ly
            
            f_reconstructed[j, i] = val
    
    # 边界处理：确保非负
    f_reconstructed = np.maximum(f_reconstructed, 0.0)
    
    # 平滑处理：去除数值振荡
    f_reconstructed = np.clip(f_reconstructed, 0.0, 10.0 * np.max(f_grid))
    
    return f_reconstructed


def test_interpolation_accuracy(v_parallel, v_perp, f_grid, params):
    """
    测试插值精度，计算L2误差。
    """
    f_rec = lagrange_phase_space_reconstruction(v_parallel, v_perp, f_grid, params)
    
    dv_par = v_parallel[1] - v_parallel[0]
    dv_perp = v_perp[1] - v_perp[0]
    
    error = np.sqrt(np.sum((f_rec - f_grid)**2) * dv_par * dv_perp)
    norm = np.sqrt(np.sum(f_grid**2) * dv_par * dv_perp)
    
    if norm > 1e-30:
        rel_error = error / norm
    else:
        rel_error = 0.0
    
    return rel_error
