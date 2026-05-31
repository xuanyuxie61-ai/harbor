#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
速度空间矩计算器
================================================================================

基于 931_pyramid_felippa_rule 的高维数值积分思想，
在速度空间中计算分布函数的各阶矩。

核心物理模型：

分布函数的各阶速度矩：
    M^{(n)} = ∫ f(𝐯) 𝐯^{⊗n} d³v

重要矩：

0阶矩（密度扰动）：
    δn = n_0 × [∫ f d³v / ∫ f_M d³v - 1]

1阶矩（平行动量流）：
    Γ_∥ = ∫ v_∥ f d³v

2阶矩（压力张量）：
    P_{ij} = m_e ∫ v_i v_j f d³v

平行/垂直温度：
    T_∥ = (m_e / n) ∫ (v_∥ - u_∥)² f d³v / q_e    [eV]
    T_⊥ = (m_e / 2n) ∫ v_⊥² f d³v / q_e            [eV]

温度各向异性：
    A = T_⊥ / T_∥ - 1

热流矢量（3阶矩）：
    Q_∥ = (1/2) m_e ∫ (v_∥ - u_∥)³ f d³v

数值积分方法：
在柱坐标 (v_∥, v_⊥, φ) 下，d³v = v_⊥ dv_∥ dv_⊥ dφ。
对 φ 积分解析完成（轴对称假设），对 (v_∥, v_⊥) 使用数值积分。

Felippa金字塔积分规则（用于3D速度空间积分）：
    ∫_P f(x,y,z) dV ≈ Σ_{i=1}^{N_q} w_i f(x_i, y_i, z_i)
================================================================================
"""

import numpy as np


def integrate_2d_velocity_space(v_parallel, v_perp, integrand):
    """
    在 (v_∥, v_⊥) 网格上执行2D数值积分。
    
    使用复合Simpson规则：
        I ≈ Σ_i Σ_j w_i w_j g(v_∥,i, v_⊥,j)
    
    参数
    ----
    v_parallel, v_perp : ndarray
        速度网格。
    integrand : ndarray, shape (nv_perp, nv_par)
        被积函数值（已包含 v_⊥ Jacobian）。
        
    返回
    ----
    result : float
        积分值。
    """
    nv_par = len(v_parallel)
    nv_perp = len(v_perp)
    
    dv_par = v_parallel[1] - v_parallel[0] if nv_par > 1 else 1.0
    dv_perp = v_perp[1] - v_perp[0] if nv_perp > 1 else 1.0
    
    # Simpson权重
    w_par = np.ones(nv_par)
    w_par[0] = 0.5
    w_par[-1] = 0.5
    if nv_par > 2:
        w_par[1:-1:2] = 2.0
        w_par[2:-1:2] = 2.0
        # 修正为标准Simpson
        w_par = np.ones(nv_par)
        w_par[0] = 1.0 / 3.0
        w_par[-1] = 1.0 / 3.0
        if nv_par % 2 == 0:
            # 偶数点用梯形收尾
            w_par[-2] = 4.0 / 3.0
        else:
            for i in range(1, nv_par - 1):
                if i % 2 == 1:
                    w_par[i] = 4.0 / 3.0
                else:
                    w_par[i] = 2.0 / 3.0
    
    w_perp = np.ones(nv_perp)
    w_perp[0] = 1.0 / 3.0
    w_perp[-1] = 1.0 / 3.0
    if nv_perp > 2:
        for i in range(1, nv_perp - 1):
            if i % 2 == 1:
                w_perp[i] = 4.0 / 3.0
            else:
                w_perp[i] = 2.0 / 3.0
    
    # 处理偶数点情况（简化：使用复合梯形）
    if nv_par % 2 == 0 or nv_par < 3:
        w_par = np.ones(nv_par)
        w_par[0] = 0.5
        w_par[-1] = 0.5
    
    if nv_perp % 2 == 0 or nv_perp < 3:
        w_perp = np.ones(nv_perp)
        w_perp[0] = 0.5
        w_perp[-1] = 0.5
    
    result = 0.0
    for j in range(nv_perp):
        for i in range(nv_par):
            result += w_perp[j] * w_par[i] * integrand[j, i] * dv_perp * dv_par
    
    # 对 φ 的积分贡献：2π（轴对称）
    result *= 2.0 * np.pi
    
    return result


def compute_velocity_space_moments(v_parallel, v_perp, f_grid, params):
    """
    计算分布函数的各阶速度空间矩。
    
    参数
    ----
    v_parallel, v_perp : ndarray
        速度网格 [m/s]。
    f_grid : ndarray, shape (nv_perp, nv_par)
        分布函数 f(v_⊥, v_∥)。
    params : dict
        物理参数。
        
    返回
    ----
    moments : dict
        包含各阶矩的字典。
    """
    m_e = params['m_e']
    q_e = params['q_e']
    n0 = params['n0']
    v_te = params['v_te']
    
    nv_par = len(v_parallel)
    nv_perp = len(v_perp)
    
    VP, VPL = np.meshgrid(v_perp, v_parallel, indexing='ij')
    
    # 边界处理：确保 f ≥ 0
    f_grid = np.maximum(f_grid, 0.0)
    
    # 计算参考 Maxwellian
    v_sq = VPL**2 + VP**2
    f_maxwell = (1.0 / (np.pi * v_te**2))**(1.5) * np.exp(-v_sq / v_te**2)
    n_maxwell = integrate_2d_velocity_space(v_parallel, v_perp, f_maxwell * VP)
    
    # 将 f_grid 按峰值缩放到与 f_maxwell 相同的量级
    f_peak = np.max(f_grid)
    fM_peak = np.max(f_maxwell)
    if f_peak > 1e-30 and fM_peak > 1e-30:
        f_grid = f_grid * (fM_peak / f_peak)
    
    # 0阶矩：密度
    integrand_n = f_grid * VP  # v_⊥ Jacobian
    n_density = integrate_2d_velocity_space(v_parallel, v_perp, integrand_n)
    
    density_perturbation = (n_density - n_maxwell) / (n_maxwell + 1e-30)
    
    # 1阶矩：平行动量流
    integrand_gamma = f_grid * VP * VPL
    gamma_parallel = integrate_2d_velocity_space(v_parallel, v_perp, integrand_gamma)
    u_parallel = gamma_parallel / (n_density + 1e-30)
    
    # 2阶矩：平行压力
    integrand_ppar = f_grid * VP * (VPL - u_parallel)**2
    p_parallel = m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_ppar)
    
    # 2阶矩：垂直压力
    integrand_pperp = f_grid * VP * VP**2 / 2.0
    p_perp = m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_pperp)
    
    # 温度 [eV]
    T_parallel = p_parallel / (n_density * q_e + 1e-30)
    T_perp = p_perp / (n_density * q_e + 1e-30)
    
    # 边界处理
    T_parallel = max(T_parallel, 0.01)
    T_perp = max(T_perp, 0.01)
    
    # 温度各向异性
    anisotropy = T_perp / T_parallel - 1.0
    
    # 3阶矩：平行热流
    integrand_q = f_grid * VP * (VPL - u_parallel)**3
    Q_parallel = 0.5 * m_e * integrate_2d_velocity_space(v_parallel, v_perp, integrand_q)
    
    # 熵（Boltzmann H-定理）
    f_pos = np.maximum(f_grid, 1e-30)
    integrand_s = -f_pos * np.log(f_pos) * VP
    entropy = integrate_2d_velocity_space(v_parallel, v_perp, integrand_s)
    
    moments = {
        'density': n_density,
        'density_perturbation': density_perturbation,
        'parallel_flow': u_parallel,
        'p_parallel': p_parallel,
        'p_perp': p_perp,
        'T_parallel': T_parallel,
        'T_perp': T_perp,
        'anisotropy': anisotropy,
        'Q_parallel': Q_parallel,
        'entropy': entropy
    }
    
    return moments
