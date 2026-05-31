# -*- coding: utf-8 -*-
"""
nonlinear_resonator.py
基于 rubber_band_ode（分段非线性弹簧）与 pendulum_ode_period（非线性摆），
构建超表面单元（meta-atom）的非线性电磁共振模型。

核心科学问题：
  在高入射光强下，介电超表面单元的有效极化率呈现 Kerr 非线性：
      p = ε0 α E_inc + ε0 χ^(3) |E_inc|² E_inc
  共振相位随光强发生非线性偏移（光学双稳态）。
  将 meta-atom 等效为受驱非线性振子：
      d²u/dt² + γ du/dt + ω0² u + α u³ = F cos(ωt)
  其中 u 为等效偶极矩，F ∝ E_inc。

关键公式：
  1. 橡皮筋振子（rubber_band_ode）:
       u'' + 0.01 u' + a·u⁺ - b·u⁻ = 10 + λ sin(μt)
     其中 u⁺ = max(u,0), u⁻ = max(-u,0)。
     对应双稳态 meta-atom：正/负偏置下具有不同等效刚度。
  2. 非线性摆（pendulum_ode）:
       u' = v
       v' = -(g/l) sin(u)
     对应大角度失谐时的相位饱和效应。
  3. Duffing 振子（驱动非线性）:
       u'' + γ u' + ω0² u + β u³ = F0 cos(ωt)
  4. 有效相位偏移:
       Δφ(I) = Δφ0 · (1 + κ I / I_sat)^{-1}
     其中 I_sat 为饱和光强。
  5. 周期估算（参考 pendulum_period）:
       对于小角度 T ≈ 2π√(l/g)
       对于大角度使用椭圆积分
"""

import numpy as np
from math import pi, sqrt, sin, cos


def rubber_band_resonator(t, y, a=1.0, b=2.0, lam=5.0, mu=1.0):
    """
    橡皮筋分段非线性振子（参考 rubber_band_deriv）。
    对应 rubber_band_ode: y'' + 0.01 y' + a·y⁺ - b·y⁻ = 10 + λ sin(μt)
    这里将驱动力项改为与 t 相关的显式参数。
    """
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [u, v]")
    u = float(np.real(y[0]))
    v = float(np.real(y[1]))
    up = v
    vp = (10.0 + lam * sin(mu * t) - 0.01 * v -
          a * max(u, 0.0) + b * max(-u, 0.0))
    return np.array([up, vp], dtype=complex)


def pendulum_meta_atom(t, y, g=9.81, l=1.0):
    """
    非线性摆模型（参考 pendulum_deriv）。
    对应大角度 meta-atom 失谐：
        u' = v
        v' = -(g/l) sin(u)
    小角度近似时退化为简谐振子，v' ≈ -(g/l) u。
    """
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [θ, ω]")
    theta = float(np.real(y[0]))
    omega = float(np.real(y[1]))
    # 边界保护：角度归一化到 [-π, π] 以避免数值发散
    theta = ((theta + pi) % (2.0 * pi)) - pi
    dtheta = omega
    domega = -(g / l) * sin(theta)
    return np.array([dtheta, domega], dtype=complex)


def duffing_resonator(t, y, alpha=1.0, beta=0.1, gamma=0.05, omega=1.0, F=1.0):
    """
    受驱 Duffing 振子，描述 Kerr 非线性 meta-atom：
        u'' + γ u' + α u + β u³ = F cos(ω t)
    参数:
        alpha: 线性刚度 (ω0²)
        beta:  非线性刚度 (χ^(3) 等效)
        gamma: 阻尼系数
        omega: 驱动频率
        F:     驱动幅度 ∝ E_inc
    """
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [u, v]")
    u = float(np.real(y[0]))
    v = float(np.real(y[1]))
    up = v
    vp = F * cos(omega * t) - gamma * v - alpha * u - beta * (u ** 3)
    # 边界保护：限制加速度避免数值爆炸
    if abs(vp) > 1e6:
        vp = np.sign(vp) * 1e6
    return np.array([up, vp], dtype=complex)


def effective_phase_shift_nonlinear(incident_intensity, params):
    """
    基于非线性振子稳态响应计算 meta-atom 的有效相位偏移。

    模型: Δφ(I) = arctan( (ω0² - ω²) / (γ ω) ) ·
                   (1 + κ I / I_sat)^{-1/2}
    其中 κ 为非线性耦合系数，I_sat 为饱和光强。
    """
    I = float(incident_intensity)
    if I < 0:
        I = 0.0
    omega0 = params.get('omega0', 1.0)
    omega = params.get('omega', 1.0)
    gamma = params.get('gamma', 0.05)
    kappa = params.get('kappa', 0.01)
    I_sat = params.get('I_sat', 1.0)

    if I_sat < 1e-15:
        I_sat = 1e-15

    # 线性响应相位
    detuning = omega0 ** 2 - omega ** 2
    denom = max(gamma * omega, 1e-15)
    phi_linear = np.arctan2(detuning, denom)

    # 非线性饱和修正
    saturation_factor = 1.0 / sqrt(1.0 + kappa * I / I_sat)
    phi_eff = phi_linear * saturation_factor
    return phi_eff


def pendulum_period_small_angle(g, l):
    """
    小角度非线性摆周期（参考 pendulum_period）：
        T = 2π √(l/g)
    """
    if g <= 0 or l <= 0:
        raise ValueError("g 和 l 必须为正")
    return 2.0 * pi * sqrt(l / g)


def pendulum_period_elliptic(theta0, g, l, n_terms=10):
    """
    大角度非线性摆周期（椭圆积分展开）：
        T = 4 √(l/g) K(sin²(θ0/2))
        K(k) = (π/2) Σ_{n=0}^∞ [ (2n)! / (2^{2n} (n!)²) ]² k^n
    """
    if g <= 0 or l <= 0:
        raise ValueError("g 和 l 必须为正")
    if abs(theta0) >= pi:
        theta0 = np.sign(theta0) * (pi - 1e-6)
    k = sin(theta0 / 2.0) ** 2
    K = pi / 2.0
    term = 1.0
    for n in range(1, n_terms):
        # 递推计算级数项
        term *= ((2.0 * n - 1.0) / (2.0 * n)) ** 2 * k
        K += term
        if abs(term) < 1e-15:
            break
    T = 4.0 * sqrt(l / g) * K
    return T


def nonlinear_transmission_coefficient(intensity, params):
    """
    计算非线性超表面单元的复透射系数：
        t(I) = A(I) exp(i φ(I))
    其中幅度 A(I) 也受非线性影响：
        A(I) = A0 / √(1 + I / I_sat)
    """
    I = float(intensity)
    if I < 0:
        I = 0.0
    A0 = params.get('A0', 1.0)
    I_sat = params.get('I_sat', 1.0)
    if I_sat < 1e-15:
        I_sat = 1e-15
    phi = effective_phase_shift_nonlinear(I, params)
    A = A0 / sqrt(1.0 + I / I_sat)
    A = np.clip(A, 0.0, 1.0)
    return A * np.exp(1j * phi)
