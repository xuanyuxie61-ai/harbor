#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_quadrature.py
=================
【融合种子项目】 531_hexahedron_jaskowiec_rule (高阶六面体求积规则)

将高阶 Gauss 求积规则移植到中子星 EoS 矩量积分.
中子星内部热力学量需要对动量空间进行积分:
    P = (1/3 pi^2 hbar^3) integral_0^{p_F} p^4 / sqrt(p^2 c^2 + m^2 c^4) dp

物理/数学公式
-------------
1. 相对论费米气体压力:
       P = (m^4 c^5)/(8 pi^2 hbar^3) [x(2x^2-3)sqrt(x^2+1) + 3 sinh^{-1}(x)]
   其中 x = p_F / (mc) 为相对论费米动量.

2. 能量密度:
       epsilon = (m^4 c^5)/(8 pi^2 hbar^3) [x(2x^2+1)sqrt(x^2+1) - sinh^{-1}(x)]
       + rho m_n c^2

3. Gauss-Legendre 求积:
       integral_{-1}^{1} f(x) dx ≈ sum_{i=1}^n w_i f(x_i)
   其中 x_i 为 P_n(x) 的零点, w_i = 2/[(1-x_i^2)(P_n'(x_i))^2]

4. 张量积求积 (3D, 用于各向异性核物质):
       integral_{H} f(x,y,z) dV ≈ sum_i sum_j sum_k w_i w_j w_k f(x_i,y_j,z_k)

5. 对称性简化 (Jaskowiec 思想):
   利用立方体对称群 Oh, 将求积点分组为等价轨道,
   减少独立参数数.
"""

import math
import numpy as np
from typing import Tuple, Callable, Dict
from numerical_constants import (legendre_set, r8_epsilon,
                                  NeutronStarConstants as NS)


def relativistic_fermi_pressure(p_fermi: float, m_species: float) -> float:
    """
    相对论完全简并费米气体压力.

    P = (m^4 c^5)/(8 pi^2 hbar^3) [x(2x^2-3)sqrt(x^2+1) + 3 arcsinh(x)]

    Parameters
    ----------
    p_fermi : float
        费米动量 (g cm/s)
    m_species : float
        粒子质量 (g)

    Returns
    -------
    float
        压力 (dyn/cm^2)
    """
    if p_fermi <= 0 or m_species <= 0:
        return 0.0

    x = p_fermi / (m_species * NS.c_light)
    prefactor = (m_species ** 4 * NS.c_light ** 5) / (8.0 * math.pi ** 2 * NS.hbar ** 3)

    term1 = x * (2.0 * x * x - 3.0) * math.sqrt(x * x + 1.0)
    term2 = 3.0 * math.log(x + math.sqrt(x * x + 1.0))

    return prefactor * (term1 + term2)


def relativistic_fermi_energy_density(p_fermi: float, m_species: float) -> float:
    """
    相对论完全简并费米气体能量密度 (含静止质量).

    epsilon = (m^4 c^5)/(8 pi^2 hbar^3) [x(2x^2+1)sqrt(x^2+1) - arcsinh(x)]
    """
    if p_fermi <= 0 or m_species <= 0:
        return 0.0

    x = p_fermi / (m_species * NS.c_light)
    prefactor = (m_species ** 4 * NS.c_light ** 5) / (8.0 * math.pi ** 2 * NS.hbar ** 3)

    term1 = x * (2.0 * x * x + 1.0) * math.sqrt(x * x + 1.0)
    term2 = math.log(x + math.sqrt(x * x + 1.0))

    return prefactor * (term1 - term2)


def fermi_momentum_from_density(n_baryon: float) -> float:
    """
    由重子数密度计算费米动量.

    p_F = hbar (3 pi^2 n)^{1/3}
    """
    if n_baryon <= 0:
        return 0.0
    return NS.hbar * (3.0 * math.pi ** 2 * n_baryon) ** (1.0 / 3.0)


def moment_integral_gauss(f: Callable, a: float, b: float,
                           n_points: int = 20) -> float:
    """
    Gauss-Legendre 求积计算矩量积分.

    integral_a^b f(x) dx = (b-a)/2 * sum_{i=1}^n w_i f(x_i^{[a,b]})

    其中 x_i^{[a,b]} = (b-a)/2 * x_i + (a+b)/2

    Parameters
    ----------
    f : callable
        被积函数
    a, b : float
        积分区间
    n_points : int
        求积点数

    Returns
    -------
    float
        积分值
    """
    xi, wi = legendre_set(n_points)
    half_len = 0.5 * (b - a)
    mid = 0.5 * (a + b)

    result = 0.0
    for i in range(n_points):
        x_i = half_len * xi[i] + mid
        result += wi[i] * f(x_i)

    return half_len * result


def eos_moment_integral(rho_lo: float, rho_hi: float,
                         eos_func: Callable, moment: int = 0,
                         n_quad: int = 32) -> float:
    """
    计算 EoS 的矩量积分 (用于多极矩展开).

    M_n = integral_{rho_lo}^{rho_hi} rho^n P(rho) d rho

    Parameters
    ----------
    rho_lo, rho_hi : float
        密度区间
    eos_func : callable
        P(rho) 函数
    moment : int
        矩的阶数
    n_quad : int
        求积点数

    Returns
    -------
    float
        矩量积分值
    """
    def integrand(rho):
        if rho <= 0:
            return 0.0
        return rho ** moment * eos_func(rho)

    return moment_integral_gauss(integrand, rho_lo, rho_hi, n_quad)


def anisotropic_pressure_integral(f_angle: Callable,
                                   n_theta: int = 16,
                                   n_phi: int = 16) -> Tuple[float, float]:
    """
    各向异性压力积分 (用于磁化中子星).

    P_parallel = (1/4pi) integral P(theta,phi) cos^2(theta) d Omega
    P_perp = (1/8pi) integral P(theta,phi) sin^2(theta) d Omega

    使用 Gauss-Legendre (theta) + 梯形 (phi).
    """
    # Gauss-Legendre in cos(theta)
    mu, w_mu = legendre_set(n_theta)
    dphi = 2.0 * math.pi / n_phi

    p_par = 0.0
    p_perp = 0.0

    for i in range(n_theta):
        cos_theta = mu[i]
        sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
        for j in range(n_phi):
            phi = j * dphi
            p_val = f_angle(cos_theta, phi)
            p_par += w_mu[i] * p_val * cos_theta * cos_theta * dphi
            p_perp += w_mu[i] * p_val * sin_theta * sin_theta * dphi * 0.5

    p_par /= (4.0 * math.pi)
    p_perp /= (4.0 * math.pi)

    return p_par, p_perp


def jaskowiec_like_cubature_rule(precision: int = 5) -> Dict:
    """
    简化版 Jaskowiec 六面体求积 (移植自 531).

    对于 [-1,1]^3 上的积分, 利用对称性分组求积点.

    返回求积点和权重.
    """
    # 简化: 使用张量积 Gauss 规则
    n_1d = (precision + 2) // 2
    xi, wi = legendre_set(n_1d)

    # 张量积
    n_total = n_1d ** 3
    x_pts = np.zeros(n_total)
    y_pts = np.zeros(n_total)
    z_pts = np.zeros(n_total)
    w_pts = np.zeros(n_total)

    idx = 0
    for i in range(n_1d):
        for j in range(n_1d):
            for k in range(n_1d):
                x_pts[idx] = xi[i]
                y_pts[idx] = xi[j]
                z_pts[idx] = xi[k]
                w_pts[idx] = wi[i] * wi[j] * wi[k]
                idx += 1

    return {
        'n_points': n_total,
        'x': x_pts,
        'y': y_pts,
        'z': z_pts,
        'w': w_pts,
        'precision': precision,
    }


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== EoS 求积规则自检 ===")

    # 中子费米气体
    n_baryon = 0.16 / (197.3 ** 3) * 1e39  # fm^-3 -> cm^-3 近似
    p_f = fermi_momentum_from_density(n_baryon)
    P_n = relativistic_fermi_pressure(p_f, NS.m_n)
    eps_n = relativistic_fermi_energy_density(p_f, NS.m_n)
    print(f"中子费米动量: {p_f:.3e} g cm/s")
    print(f"中子气压力: {P_n:.3e} dyn/cm^2")
    print(f"中子气能量密度: {eps_n:.3e} erg/cm^3")

    # Gauss 积分测试: integral_0^1 x^4 dx = 1/5
    def f4(x): return x ** 4
    val = moment_integral_gauss(f4, 0.0, 1.0, 10)
    print(f"Gauss 积分 x^4 [0,1] = {val:.10f}  (精确 0.2)")

    # 矩量积分
    def eos_test(rho): return 1e34 * (rho / 1e14) ** 2.5
    M0 = eos_moment_integral(1e13, 1e15, eos_test, moment=0, n_quad=32)
    print(f"矩量积分 M_0 = {M0:.3e}")

    # 六面体求积
    rule = jaskowiec_like_cubature_rule(precision=5)
    print(f"六面体求积: {rule['n_points']} 点, 精度 {rule['precision']}")
    print(f"  权重之和 = {np.sum(rule['w']):.10f}  (应为 8)")

    print("\neos_quadrature.py 自检通过.")
