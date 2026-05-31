#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
准线性扩散算子组装器
================================================================================

基于 853_pce_legendre 的随机Galerkin有限元矩阵组装思想，
将速度空间的准线性扩散系数组装为稀疏线性算子矩阵。

核心物理模型：

准线性扩散张量（ Kennel & Engelmann 1966 ）：

对于 whistler 模与电子的回旋共振 (n = -1, 电子回旋阻尼)：

    D_{∥∥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² J_1²(k_⊥ v_⊥ / Ω_e)
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ) × (1 - k_∥ v_∥ / ω_k)²

    D_{⊥⊥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² [J_0'(x_e)]²
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ) × (k_∥ v_⊥ / ω_k)²

    D_{∥⊥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² J_1(x_e) J_0'(x_e)
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ)
                  × (1 - k_∥ v_∥ / ω_k) × (k_∥ v_⊥ / ω_k)

其中 x_e = k_⊥ v_⊥ / Ω_e，J_n 为Bessel函数。

δ函数的宽化（准线性近似）：
    δ(x) → 1/(√π Δv) exp(-x²/Δv²)

速度空间扩散方程的离散形式：
    ∂f/∂t = D_{∥∥} ∂²f/∂v_∥² + D_{⊥⊥} (1/v_⊥) ∂/∂v_⊥ (v_⊥ ∂f/∂v_⊥)
          + 2 D_{∥⊥} ∂²f/∂v_∥∂v_⊥

有限差分离散（中心差分）：
    ∂²f/∂v_∥² ≈ (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / Δv_∥²
    (1/v_⊥) ∂/∂v_⊥ (v_⊥ ∂f/∂v_⊥) ≈ 
        [v_{⊥,j+1/2}(f_{i,j+1}-f_{i,j}) - v_{⊥,j-1/2}(f_{i,j}-f_{i,j-1})]
        / (v_{⊥,j} Δv_⊥²)
================================================================================
"""

import numpy as np
from scipy.special import jv


def compute_ql_diffusion_coefficients(v_parallel, v_perp, omega_solutions, params):
    """
    计算速度空间的准线性扩散系数 D_{∥∥}, D_{⊥⊥}, D_{∥⊥}。
    
    参数
    ----
    v_parallel : ndarray, shape (nv,)
        平行速度网格 [m/s]。
    v_perp : ndarray, shape (nv,)
        垂直速度网格 [m/s]。
    omega_solutions : ndarray, shape (Nk, 2)
        (k, ω) 色散解。
    params : dict
        物理参数。
        
    返回
    ----
    D_par : ndarray, shape (nv, nv)
        D_{∥∥} 在 (v_∥, v_⊥) 网格上。
    D_perp : ndarray, shape (nv, nv)
        D_{⊥⊥} 在 (v_∥, v_⊥) 网格上。
    D_cross : ndarray, shape (nv, nv)
        D_{∥⊥} 在 (v_∥, v_⊥) 网格上。
    """
    nv = len(v_parallel)
    q_e = params['q_e']
    m_e = params['m_e']
    B0 = params['B0']
    Omega_e = params['Omega_e']
    c = params['c']
    
    VP, VPL = np.meshgrid(v_perp, v_parallel, indexing='ij')
    
    # 洛伦兹因子
    v_sq = VPL**2 + VP**2
    gamma = 1.0 / np.sqrt(1.0 - v_sq / c**2)
    # 边界：确保 γ 有限
    gamma = np.clip(gamma, 1.0, 100.0)
    
    D_par = np.zeros((nv, nv), dtype=np.float64)
    D_perp = np.zeros((nv, nv), dtype=np.float64)
    D_cross = np.zeros((nv, nv), dtype=np.float64)
    
    # 波振幅（简化模型）
    E_wave_amp = 1e-4  # V/m
    
    # 对每支波模式求和
    for k_omega in omega_solutions:
        k = float(np.real(k_omega[0]))
        omega = complex(k_omega[1])
        omega_r = float(np.real(omega))
        
        if omega_r <= 0:
            continue
        
        # 共振条件宽化宽度
        delta_v = 0.1 * params['v_te']
        
        # 垂直波数（假设 k_⊥ ~ 0.3 k_∥）
        k_perp = 0.3 * np.abs(k)
        
        # Bessel函数参数
        x_e = k_perp * VP / Omega_e
        x_e = np.clip(x_e, -100.0, 100.0)  # 避免Bessel函数溢出
        
        J1 = jv(1, x_e)
        J0p = -jv(1, x_e)  # J_0'(x) = -J_1(x)
        
        # 共振条件：ω - k_∥ v_∥ - Ω_e/γ
        resonance = omega_r - k * VPL - Omega_e / gamma
        
        # 宽化的δ函数
        delta_func = (1.0 / (np.sqrt(np.pi) * delta_v)) * np.exp(-resonance**2 / delta_v**2)
        
        # 系数
        prefactor = float(np.pi * q_e**2 / m_e**2 * E_wave_amp**2)
        
        # 投影因子
        P_par = (1.0 - k * VPL / omega_r).astype(np.float64)
        P_perp = (k * VP / omega_r).astype(np.float64)
        
        # 边界处理
        P_par = np.clip(P_par, -10.0, 10.0)
        P_perp = np.clip(P_perp, -10.0, 10.0)
        
        D_par += (prefactor * (J1**2).astype(np.float64) * delta_func.astype(np.float64) * (P_par**2).astype(np.float64))
        D_perp += (prefactor * (J0p**2).astype(np.float64) * delta_func.astype(np.float64) * (P_perp**2).astype(np.float64))
        D_cross += (prefactor * J1.astype(np.float64) * J0p.astype(np.float64) * delta_func.astype(np.float64) * P_par.astype(np.float64) * P_perp.astype(np.float64))
    
    # 确保非负
    D_par = np.abs(D_par)
    D_perp = np.abs(D_perp)
    D_cross = np.clip(D_cross, -np.sqrt(D_par * D_perp), np.sqrt(D_par * D_perp))
    
    return D_par, D_perp, D_cross


def assemble_ql_diffusion_matrix(v_parallel, v_perp, omega_solutions, params,
                                  n_stochastic=2, p_degree=2):
    """
    组装准线性扩散算子的稀疏矩阵。
    
    基于 853_pce_legendre 的随机Galerkin矩阵组装：
    将 (v_∥, v_⊥) 二维离散与多项式混沌展开结合，
    得到大稀疏线性系统 A f = rhs。
    
    参数
    ----
    v_parallel, v_perp : ndarray
        速度网格。
    omega_solutions : ndarray
        色散解。
    params : dict
        物理参数。
    n_stochastic : int
        随机维度。
    p_degree : int
        PCE阶数。
        
    返回
    ----
    A : ndarray, shape (N, N)
        扩散算子矩阵（稠密，用于小系统）。
    rhs : ndarray, shape (N,)
        初始条件向量。
    """
    nv = len(v_parallel)
    
    # PCE维度
    from pce_expansion import enumerate_multi_indices
    indices = enumerate_multi_indices(n_stochastic, p_degree)
    M_pce = len(indices)
    
    # 总维度
    N_total = nv * nv * M_pce
    
    # 为可行性，使用简化版本：仅空间离散，PCE退化为标量乘性因子
    # 计算扩散系数
    D_par, D_perp, D_cross = compute_ql_diffusion_coefficients(
        v_parallel, v_perp, omega_solutions, params
    )
    
    dv_par = v_parallel[1] - v_parallel[0]
    dv_perp = v_perp[1] - v_perp[0]
    
    # 组装扩散矩阵
    N = nv * nv
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    
    for j in range(nv):      # v_⊥ 索引
        for i in range(nv):  # v_∥ 索引
            idx = j * nv + i
            
            # 中心点系数
            coeff = 0.0
            
            # D_∥∥ 项: ∂²/∂v_∥²
            if i > 0 and i < nv - 1:
                Dpp = D_par[j, i]
                A[idx, idx - 1] += Dpp / dv_par**2
                A[idx, idx] += -2.0 * Dpp / dv_par**2
                A[idx, idx + 1] += Dpp / dv_par**2
            
            # D_⊥⊥ 项: (1/v_⊥) ∂/∂v_⊥ (v_⊥ ∂/∂v_⊥)
            if j > 0 and j < nv - 1:
                v_half_p = 0.5 * (v_perp[j] + v_perp[j+1])
                v_half_m = 0.5 * (v_perp[j-1] + v_perp[j])
                
                Dpp_val = D_perp[j, i]
                
                # 边界检查：避免 v_⊥ = 0 处的奇异性
                v_center = max(v_perp[j], 1e-10)
                
                A[idx, idx - nv] += Dpp_val * v_half_m / (v_center * dv_perp**2)
                A[idx, idx] += -Dpp_val * (v_half_p + v_half_m) / (v_center * dv_perp**2)
                A[idx, idx + nv] += Dpp_val * v_half_p / (v_center * dv_perp**2)
            
            # D_∥⊥ 交叉项（简化：仅保留对称部分）
            if i > 0 and i < nv - 1 and j > 0 and j < nv - 1:
                Dcr = D_cross[j, i]
                # 混合差分（中心差分）
                A[idx, idx + nv + 1] += Dcr / (4.0 * dv_par * dv_perp)
                A[idx, idx + nv - 1] += -Dcr / (4.0 * dv_par * dv_perp)
                A[idx, idx - nv + 1] += -Dcr / (4.0 * dv_par * dv_perp)
                A[idx, idx - nv - 1] += Dcr / (4.0 * dv_par * dv_perp)
            
            # 初始条件：Maxwellian
            v_sq = v_parallel[i]**2 + v_perp[j]**2
            v_te = params['v_te']
            rhs[idx] = (1.0 / (np.pi * v_te**2))**(1.5) * np.exp(-v_sq / v_te**2)
    
    # 归一化rhs
    if np.linalg.norm(rhs) > 1e-30:
        rhs = rhs / np.linalg.norm(rhs)
    
    return A, rhs
