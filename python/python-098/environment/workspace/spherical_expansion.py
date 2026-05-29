# -*- coding: utf-8 -*-
"""
spherical_expansion.py
基于 polpak 中的 spherical_harmonic、legendre_associated、legendre_poly，
实现远场电磁全息图案的球谐函数展开与重构。

核心科学问题：
  超表面在远场产生的散射场 E(θ,φ) 可在球面上展开为矢量球谐函数（VSWF）
  或标量球谐函数的叠加：
      E(θ, φ) = Σ_{l=0}^{L} Σ_{m=-l}^{l} a_{lm} Y_l^m(θ, φ)
  其中 Y_l^m 为归一化球谐函数：
      Y_l^m(θ, φ) = √((2l+1)/(4π) · (l-m)!/(l+m)!) · P_l^m(cosθ) · e^{imφ}

关键公式：
  1. 连带 Legendre 函数（Ferrers 函数）:
       P_l^m(x) = (-1)^m (1-x^2)^{m/2} d^m/dx^m P_l(x)
  2. 球谐函数归一化:
       N_l^m = √((2l+1)/(4π) · (l-m)!/(l+m)!)
  3. 正交性:
       ∫_0^{2π} ∫_0^π Y_l^m Y_{l'}^{m'*} sinθ dθ dφ = δ_{ll'} δ_{mm'}
  4. 展开系数:
       a_{lm} = ∫_0^{2π} ∫_0^π E(θ,φ) Y_l^{m*}(θ,φ) sinθ dθ dφ
"""

import numpy as np
from math import factorial


def associated_legendre(l_max, m, x):
    """
    计算归一化连带 Legendre 函数 P_l^m(x)（l = m,...,l_max）。
    采用递推法以保证数值稳定性。
    参考 polpak 中的 legendre_associated_normalized。

    返回数组 plm，长度 l_max+1，其中 plm[l] = P_l^m(x)（l < m 时为 0）。
    """
    x = float(x)
    m_abs = abs(m)
    if l_max < m_abs:
        return np.zeros(l_max + 1)

    plm = np.zeros(l_max + 1)
    # 初始条件
    p_mm = 1.0
    if m_abs > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m_abs + 1):
            p_mm *= -fact * somx2
            fact += 2.0
    plm[m_abs] = p_mm

    if l_max > m_abs:
        p_mp1m = x * (2.0 * m_abs + 1.0) * p_mm
        plm[m_abs + 1] = p_mp1m

    for l in range(m_abs + 2, l_max + 1):
        plm[l] = ((2.0 * l - 1.0) * x * plm[l - 1] -
                  (l + m_abs - 1.0) * plm[l - 2]) / (l - m_abs)
    return plm


def spherical_harmonic_y(l, m, theta, phi):
    """
    计算归一化球谐函数 Y_l^m(θ, φ)（参考 spherical_harmonic）。
    返回复数值。

    参数:
        l: 阶数 (int, >= 0)
        m: 次数 (int, -l <= m <= l)
        theta: 极角 [0, π]
        phi:   方位角 [0, 2π)
    """
    if not (-l <= m <= l):
        raise ValueError(f"m={m} 不在 [-l, l] = [{-l}, {l}] 范围内")
    theta = float(theta)
    phi = float(phi)
    x = np.cos(theta)
    plm = associated_legendre(l, abs(m), x)
    # 归一化系数
    m_abs = abs(m)
    norm_coeff = np.sqrt(
        (2.0 * l + 1.0) / (4.0 * np.pi) *
        factorial(l - m_abs) / factorial(l + m_abs)
    )
    # Condon-Shortley 相位已包含在递推中（(-1)^m）
    y_val = norm_coeff * plm[l] * np.exp(1j * m * phi)
    return y_val


def expand_far_field_spherical(field_samples, theta_grid, phi_grid, l_max):
    """
    将远场采样数据投影到球谐函数基上，返回展开系数 a_{lm}。

    参数:
        field_samples: 2-D array shape (ntheta, nphi)
        theta_grid: 1-D array, 极角采样点
        phi_grid:   1-D array, 方位角采样点
        l_max:      最大阶数
    返回:
        coeffs: dict，键为 (l,m)，值为复数系数
    """
    field_samples = np.asarray(field_samples, dtype=complex)
    theta_grid = np.asarray(theta_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    ntheta = len(theta_grid)
    nphi = len(phi_grid)
    if field_samples.shape != (ntheta, nphi):
        raise ValueError("field_samples shape 与 theta_grid/phi_grid 不匹配")

    coeffs = {}
    dtheta = np.gradient(theta_grid)
    dphi = np.gradient(phi_grid)

    # 构建积分权重矩阵 w[i,j] = sin(θ_i) dθ_i dφ_j
    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    dTH, dPH = np.meshgrid(dtheta, dphi, indexing='ij')
    weights = np.sin(TH) * dTH * dPH

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ylm_grid = np.zeros((ntheta, nphi), dtype=complex)
            for i in range(ntheta):
                for j in range(nphi):
                    ylm_grid[i, j] = spherical_harmonic_y(l, m, theta_grid[i], phi_grid[j])
            # 数值积分: a_lm = ∫ E Y_l^{m*} sinθ dθ dφ
            integrand = field_samples * np.conj(ylm_grid) * weights
            a_lm = np.sum(integrand)
            coeffs[(l, m)] = a_lm
    return coeffs


def reconstruct_far_field(coeffs, theta_grid, phi_grid):
    """
    由球谐展开系数重构远场分布。

    返回:
        field: 2-D array shape (len(theta_grid), len(phi_grid))
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    ntheta = len(theta_grid)
    nphi = len(phi_grid)
    field = np.zeros((ntheta, nphi), dtype=complex)
    for (l, m), a_lm in coeffs.items():
        for i in range(ntheta):
            for j in range(nphi):
                field[i, j] += a_lm * spherical_harmonic_y(l, m, theta_grid[i], phi_grid[j])
    return field


def vector_spherical_harmonic_m(l, m, theta, phi):
    """
    矢量球谐函数 M_{lm}（TE 模），在辐射问题中用于描述横电波。
    M_{lm} = ∇ × (r ψ_{lm}) = r̂ × ∇ψ_{lm} · r
    这里给出远场近似下的角向分量：
        M_{lm} ~ i^{l+1} · ∂Y_l^m/∂θ · θ̂ + i^{l+1} · (im/sinθ) Y_l^m · φ̂
    为简化起见，本实现返回标量幅度（对应 θ 分量）。
    """
    ylm = spherical_harmonic_y(l, m, theta, phi)
    # 数值微分近似 dY/dθ
    delta = 1e-6
    ylm_p = spherical_harmonic_y(l, m, min(theta + delta, np.pi - 1e-8), phi)
    ylm_m = spherical_harmonic_y(l, m, max(theta - delta, 1e-8), phi)
    dydtheta = (ylm_p - ylm_m) / (2.0 * delta)
    return dydtheta


def vector_spherical_harmonic_n(l, m, theta, phi):
    """
    矢量球谐函数 N_{lm}（TM 模）的远场 θ 分量近似：
        N_{lm} ~ i^l · l(l+1) Y_l^m
    """
    ylm = spherical_harmonic_y(l, m, theta, phi)
    return l * (l + 1.0) * ylm


def scattering_coefficients_mie(l_max, k, a, eps_r, mu_r):
    """
    介电纳米柱（圆柱近似）的 Mie 散射系数 a_l、b_l。
    对于半径为 a 的均匀介质球，电偶极子项（l=1）主导：
        a_l = ...
    这里给出小参数 k·a << 1 下的近似：
        a_1 ≈ i (2/3) (ka)^3 (ε_r - 1)/(ε_r + 2)
    返回电多极子系数 a_l（复数数组，长度 l_max+1）。
    """
    ka = k * a
    a_coeffs = np.zeros(l_max + 1, dtype=complex)
    if ka <= 0.0:
        return a_coeffs
    # 小粒子近似（Rayleigh 散射）
    a_coeffs[1] = 1j * (2.0 / 3.0) * (ka ** 3) * (eps_r - 1.0) / (eps_r + 2.0)
    # 高阶项按 (ka)^{2l+1} 衰减，数值保护
    for l in range(2, l_max + 1):
        a_coeffs[l] = a_coeffs[1] * (ka ** (2 * l - 2)) / (2.0 ** l)
        # 防止溢出
        if abs(a_coeffs[l]) > 1e6:
            a_coeffs[l] = 0.0
    return a_coeffs
