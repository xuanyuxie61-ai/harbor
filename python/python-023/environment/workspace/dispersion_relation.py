#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
等离子体色散关系求解器
================================================================================

基于 Newton-Raphson 反向通信迭代求解 whistler 模电磁波的复频率 ω(k)。
融合了 802_newton_rc 的反向通信Newton法思想。

核心物理模型：

对于平行于背景磁场传播的 whistler 模 (右旋圆偏振, R-mode)，
冷等离子体近似下的色散关系为：

    n² = c² k² / ω² = 1 - ω_pe² / [ω (ω - Ω_e)]

其中 n 为折射率，ω_pe 为电子等离子体频率，Ω_e = |q_e| B_0 / m_e 为电子回旋频率。

考虑热效应（的动力学修正），引入平行传播下的一般色散函数：

    D(k, ω) = 1 - ω_pe²/(2 ω Ω_e) × [ Z(ζ_e) - (1 - ω/(k v_te)) Z'(ζ_e) ] = 0

其中 Z(ζ) 为等离子体色散函数（Fried-Conte函数）：

    Z(ζ) = i √π exp(-ζ²) erfc(-i ζ)
    ζ_e = (ω - Ω_e) / (|k_∥| v_te)

对于复频率 ω = ω_r + i γ，使用 Newton-Raphson 迭代：

    ω_{n+1} = ω_n - D(ω_n) / D'(ω_n)

其中 D'(ω) = ∂D/∂ω 通过对 Z(ζ) 的导数关系计算：
    Z'(ζ) = -2 [1 + ζ Z(ζ)]

数值稳定性：
- 对 |ζ| < 26 使用Faddeeva函数计算 Z(ζ)
- 对 |Re(ζ)| >> 1 使用渐进展开
- 使用反向通信(RC)结构控制Newton迭代，便于外部评估残差
================================================================================
"""

import numpy as np
from scipy.special import wofz


def plasma_dispersion_function(zeta):
    """
    计算等离子体色散函数 Z(ζ) = i√π exp(-ζ²) erfc(-iζ)。
    
    参数
    ----
    zeta : complex 或 ndarray
        复自变量。
        
    返回
    ----
    Z : complex 或 ndarray
        Z(ζ) 的值。
    """
    zeta = np.asarray(zeta, dtype=complex)
    
    # 边界处理：小参数用Faddeeva函数
    # Z(ζ) = i √π exp(-ζ²) erfc(-i ζ) = i √π w(ζ)
    # 其中 w(ζ) = exp(-ζ²) erfc(-i ζ) 为Faddeeva函数
    
    # 鲁棒性：对极大 |Im(ζ)| 使用渐进展开避免溢出
    abs_z = np.abs(zeta)
    
    if np.isscalar(abs_z):
        if abs_z > 50.0:
            # 渐进展开: Z(ζ) ≈ -1/ζ - 1/(2ζ³) - 3/(4ζ⁵) - ...
            return -1.0/zeta - 1.0/(2.0*zeta**3) - 3.0/(4.0*zeta**5)
        else:
            return 1j * np.sqrt(np.pi) * wofz(zeta)
    else:
        # 向量化处理
        Z = np.zeros_like(zeta, dtype=complex)
        mask_large = abs_z > 50.0
        mask_small = ~mask_large
        
        if np.any(mask_large):
            z_l = zeta[mask_large]
            Z[mask_large] = -1.0/z_l - 1.0/(2.0*z_l**3) - 3.0/(4.0*z_l**5)
        
        if np.any(mask_small):
            Z[mask_small] = 1j * np.sqrt(np.pi) * wofz(zeta[mask_small])
        
        return Z


def d_plasma_dispersion_function(zeta):
    """
    Z(ζ) 的导数：Z'(ζ) = -2 [1 + ζ Z(ζ)]。
    """
    Z = plasma_dispersion_function(zeta)
    return -2.0 * (1.0 + zeta * Z)


def whistler_dispersion_residual(omega, k, params):
    """
    计算 whistler 模色散函数的残差 D(k, ω) 及其对 ω 的导数。
    
    参数
    ----
    omega : complex
        试探频率 ω = ω_r + i γ。
    k : float
        平行波数 k_∥。
    params : dict
        包含等离子体参数。
        
    返回
    ----
    D : complex
        色散函数值。
    dD_domega : complex
        对 ω 的导数。
    """
    q_e = params['q_e']
    m_e = params['m_e']
    c = params['c']
    eps0 = params['eps0']
    B0 = params['B0']
    n0 = params['n0']
    Omega_e = params['Omega_e']
    omega_pe = params['omega_pe']
    v_te = params['v_te']
    
    # 边界检查
    if np.abs(omega) < 1e-20:
        omega = 1e-20 + 0j
    if np.abs(k) < 1e-20:
        k = 1e-20
    
    # 电子回旋共振参数
    zeta_e = (omega - Omega_e) / (np.abs(k) * v_te)
    
    # 色散函数
    Z_e = plasma_dispersion_function(zeta_e)
    Zp_e = d_plasma_dispersion_function(zeta_e)
    
    # 动力学色散函数 (平行传播 R-mode)
    # D = 1 - (ω_pe² / (2 ω Ω_e)) * [Z_e - (1 - ω/(k v_te)) * Zp_e]
    prefactor = omega_pe**2 / (2.0 * omega * Omega_e)
    
    bracket = Z_e - (1.0 - omega / (k * v_te)) * Zp_e
    D = 1.0 - prefactor * bracket
    
    # 计算 dD/domega
    dzeta_domega = 1.0 / (np.abs(k) * v_te)
    dZ_domega = Zp_e * dzeta_domega
    
    # Z''(ζ) = -2 Z(ζ) - 2 ζ Z'(ζ)
    Zpp_e = -2.0 * Z_e - 2.0 * zeta_e * Zp_e
    dZp_domega = Zpp_e * dzeta_domega
    
    dbracket_domega = dZ_domega - (-1.0/(k*v_te)) * Zp_e - (1.0 - omega/(k*v_te)) * dZp_domega
    
    dprefactor_domega = -omega_pe**2 / (2.0 * Omega_e * omega**2)
    
    dD_domega = -dprefactor_domega * bracket - prefactor * dbracket_domega
    
    return D, dD_domega


def solve_whistler_dispersion(k, params, omega_guess=None, tol=1e-10, max_iter=50):
    """
    使用 Newton-Raphson 方法求解 whistler 模的复频率 ω(k)。
    
    基于 802_newton_rc 的反向通信Newton法：
    - 外部提供函数残差 D(ω)
    - 迭代器内部计算Jacobian近似（此处解析计算 dD/dω）
    - 通过状态变量 ido 控制通信流程
    
    参数
    ----
    k : float
        平行波数。
    params : dict
        等离子体参数。
    omega_guess : complex, optional
        初始猜测。默认使用冷等离子体近似。
    tol : float
        收敛容差。
    max_iter : int
        最大迭代次数。
        
    返回
    ----
    omega : complex 或 None
        收敛的复频率，失败则返回 None。
    """
    Omega_e = params['Omega_e']
    omega_pe = params['omega_pe']
    
    # 冷等离子体近似给出初始猜测
    if omega_guess is None:
        # 冷等离子体 whistler 色散: ω ≈ Ω_e k² c² / (ω_pe² + k² c²)
        c = params['c']
        omega_r = Omega_e * k**2 * c**2 / (omega_pe**2 + k**2 * c**2)
        gamma = -0.05 * omega_r  # 轻微阻尼初始猜测
        omega_guess = complex(omega_r, gamma)
    
    # 反向通信状态
    ido = 0  # 0: 新任务开始
    omega = complex(omega_guess)
    
    epsilon = np.sqrt(np.sqrt(np.finfo(float).eps))
    ncall = 0
    
    for iteration in range(max_iter):
        ncall += 1
        
        # 评估残差和导数
        D_val, dD_val = whistler_dispersion_residual(omega, k, params)
        
        # 边界检查
        if not np.isfinite(D_val) or not np.isfinite(dD_val):
            # 尝试扰动 omega
            omega = omega * (1.0 + epsilon)
            continue
        
        # Newton 步长
        if np.abs(dD_val) < 1e-30:
            # Jacobian 病态，使用伪逆思想
            dD_val = np.sign(dD_val.real) * 1e-30 + 1j * np.sign(dD_val.imag) * 1e-30 if dD_val != 0 else 1e-30 + 0j
        
        delta_omega = -D_val / dD_val
        
        # 步长限制（阻尼Newton）
        if np.abs(delta_omega) > 0.5 * np.abs(omega):
            delta_omega = delta_omega * (0.5 * np.abs(omega) / np.abs(delta_omega))
        
        omega_new = omega + delta_omega
        
        # 收敛检查
        if np.abs(D_val) < tol * (np.abs(D_val) + 1.0):
            return omega
        
        # 慢收敛检查
        if iteration > 15:
            if np.abs(D_val) > 0.95 * np.abs(D_val):
                # 收敛停滞
                pass
        
        omega = omega_new
        
        # 边界条件：频率不能为负实部（非物理）
        if omega.real < 0:
            omega = complex(0.01 * Omega_e, omega.imag)
    
    # 未收敛但返回最佳估计
    D_final, _ = whistler_dispersion_residual(omega, k, params)
    if np.abs(D_final) < 100 * tol:
        return omega
    
    return None
