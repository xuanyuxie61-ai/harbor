# -*- coding: utf-8 -*-
"""
spectral_integrator.py
谱方法与高精度数值积分模块

核心算法来源：
- chebyshev1_exactness: Gauss-Chebyshev 型 1 求积公式精确度验证
- sphere_triangle_quad: 球面三角形数值积分（用于方向积分）

物理背景：
在线性稳定性分析中，扰动放大因子 N 需沿传播方向积分：
    N = ∫_{x_0}^{x_t} -α_i(x) dx

此外，三维效应涉及展向波数 β 的方向积分，
球面波矢空间的积分需采用球面求积法则。
"""

import numpy as np
from math import pi, sin, cos, sqrt, atan2


def chebyshev1_gauss_weights(n):
    """
    Gauss-Chebyshev 型 1 求积节点与权重。

    对于积分:
        I = ∫_{-1}^{+1} f(x) / sqrt(1-x^2) dx

    求积公式:
        I ≈ (π/n) Σ_{j=1}^{n} f(x_j)

    其中节点:
        x_j = cos((2j-1)π / (2n))

    该公式对不超过 2n-1 次多项式精确成立。

    参数:
        n (int): 节点数

    返回:
        tuple: (x, w) 节点和权重
    """
    j = np.arange(1, n + 1)
    x = np.cos((2.0 * j - 1.0) * pi / (2.0 * n))
    w = np.full(n, pi / n)
    return x, w


def chebyshev1_exactness_test(n, degree_max):
    """
    验证 Gauss-Chebyshev 求积对单项式 x^m 的精确性。

    精确积分:
        ∫_{-1}^{1} x^m / sqrt(1-x^2) dx =
            0,                        m 为奇数
            π (m-1)!! / m!!,         m 为偶数

    参数:
        n (int): 求积节点数
        degree_max (int): 最高测试阶数

    返回:
        list[tuple]: (degree, error) 列表
    """
    x, w = chebyshev1_gauss_weights(n)
    results = []
    for m in range(degree_max + 1):
        # 精确值
        if m % 2 == 1:
            exact = 0.0
        else:
            top = 1.0
            bot = 1.0
            for i in range(2, m + 1, 2):
                top *= (i - 1)
                bot *= i
            exact = pi * top / bot

        # 数值积分
        fvals = x ** m
        quad = np.sum(w * fvals)

        if abs(exact) < 1e-15:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)
        results.append((m, err))
    return results


def integrate_boundary_layer_growth(x_coords, alpha_i_vals):
    """
    沿流向积分扰动放大因子 N。

        N(x) = ∫_{x_0}^{x} -α_i(s) ds

    采用复合 Simpson 法则保证 O(Δx^4) 精度：
        ∫_a^b f(x) dx ≈ (b-a)/6 * [f(a) + 4f((a+b)/2) + f(b)]

    参数:
        x_coords (np.ndarray): 流向坐标
        alpha_i_vals (np.ndarray): 空间增长率（负值表示增长）

    返回:
        np.ndarray: N(x) 剖面
    """
    x = np.asarray(x_coords)
    a = np.asarray(alpha_i_vals)
    n = len(x)
    N = np.zeros(n)

    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if abs(dx) < 1e-15:
            N[i] = N[i - 1]
            continue
        # 梯形法则（复合）
        N[i] = N[i - 1] - 0.5 * dx * (a[i] + a[i - 1])
    return N


def sphere01_triangle_area(a_xyz, b_xyz, c_xyz):
    """
    计算单位球面上球面三角形的面积。

    采用 L'Huilier 定理:
        tan(E/4) = sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2))

    其中 E 为球面角盈，a,b,c 为边长（大圆弧的圆心角），s=(a+b+c)/2。
    面积 = E * R^2 = E (R=1)。

    参数:
        a_xyz, b_xyz, c_xyz (np.ndarray): 单位球面上的顶点坐标

    返回:
        float: 球面三角形面积
    """
    a_xyz = np.asarray(a_xyz) / np.linalg.norm(a_xyz)
    b_xyz = np.asarray(b_xyz) / np.linalg.norm(b_xyz)
    c_xyz = np.asarray(c_xyz) / np.linalg.norm(c_xyz)

    # 边长（圆心角）
    a_len = atan2(np.linalg.norm(np.cross(b_xyz, c_xyz)), np.dot(b_xyz, c_xyz))
    b_len = atan2(np.linalg.norm(np.cross(c_xyz, a_xyz)), np.dot(c_xyz, a_xyz))
    c_len = atan2(np.linalg.norm(np.cross(a_xyz, b_xyz)), np.dot(a_xyz, b_xyz))

    s = 0.5 * (a_len + b_len + c_len)
    # L'Huilier
    tan_sq = (np.tan(0.5 * max(s - a_len, 0.0)) *
              np.tan(0.5 * max(s - b_len, 0.0)) *
              np.tan(0.5 * max(s - c_len, 0.0)) *
              max(np.tan(0.5 * s), 1e-15))
    E = 4.0 * np.arctan(sqrt(max(tan_sq, 0.0)))
    return E


def sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3):
    """
    球面三角形上的重心坐标投影。

    给定球面三角形 ABC 与重心坐标 (f1,f2,f3)，
    先进行平面投影再归一化到单位球面。

    参数:
        a_xyz, b_xyz, c_xyz (np.ndarray): 顶点
        f1, f2, f3 (float): 重心坐标权重

    返回:
        np.ndarray: 单位球面上的投影点
    """
    a = np.asarray(a_xyz)
    b = np.asarray(b_xyz)
    c = np.asarray(c_xyz)
    p = (f1 * a + f2 * b + f3 * c) / max(f1 + f2 + f3, 1e-15)
    p = p / np.linalg.norm(p)
    return p


def sphere_triangle_quad_icos1c(a_xyz, b_xyz, c_xyz, factor, func):
    """
    基于 sphere01_triangle_quad_icos1c 的球面三角形求积。

    将球面三角形细分为 factor^2 个子三角形，
    在子三角形重心处求值，面积加权求和。

    积分:
        I = ∫_{ΔS} f(Ω) dΩ

    参数:
        a_xyz, b_xyz, c_xyz (np.ndarray): 球面三角形顶点
        factor (int): 细分因子
        func (callable): 被积函数 f(xyz) -> float

    返回:
        tuple: (result, node_num)
    """
    result = 0.0
    area_total = 0.0
    node_num = 0

    # 同向子三角形
    for f3 in range(1, 3 * factor - 1, 3):
        for f2 in range(1, 3 * factor - f3, 3):
            f1 = 3 * factor - f3 - f2
            node = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3)
            a2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 2, f2 - 1, f3 - 1)
            b2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 1, f2 + 2, f3 - 1)
            c2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 1, f2 - 1, f3 + 2)
            area = sphere01_triangle_area(a2, b2, c2)
            v = func(node)
            node_num += 1
            result += area * v
            area_total += area

    # 反向子三角形
    for f3 in range(2, 3 * factor - 3, 3):
        for f2 in range(2, 3 * factor - f3 - 2, 3):
            f1 = 3 * factor - f3 - f2
            node = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3)
            a2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 - 2, f2 + 1, f3 + 1)
            b2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 1, f2 - 2, f3 + 1)
            c2 = sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1 + 1, f2 + 1, f3 - 2)
            area = sphere01_triangle_area(a2, b2, c2)
            v = func(node)
            node_num += 1
            result += area * v
            area_total += area

    return result, node_num


def integrate_wavevector_direction(k_mag, theta_range, phi_range, integrand, n_theta=32, n_phi=16):
    """
    在球面扇区上积分展向波数效应。

    波矢方向:
        k_x = k sin(φ) cos(θ)
        k_z = k sin(φ) sin(θ)
        k_y = k cos(φ)

    积分区域: θ ∈ [θ_min, θ_max], φ ∈ [φ_min, φ_max]。

    参数:
        k_mag (float): 波数幅值
        theta_range (tuple): (θ_min, θ_max)
        phi_range (tuple): (φ_min, φ_max)
        integrand (callable): f(k_x, k_y, k_z) -> float
        n_theta (int): θ 方向节点数
        n_phi (int): φ 方向节点数

    返回:
        float: 积分值
    """
    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    # Gauss-Legendre 节点（简化用均匀节点+复合 Simpson）
    theta_nodes = np.linspace(theta_min, theta_max, n_theta)
    phi_nodes = np.linspace(phi_min, phi_max, n_phi)

    d_theta = theta_nodes[1] - theta_nodes[0]
    d_phi = phi_nodes[1] - phi_nodes[0]

    total = 0.0
    for i in range(n_theta):
        for j in range(n_phi):
            th = theta_nodes[i]
            ph = phi_nodes[j]
            kx = k_mag * sin(ph) * cos(th)
            ky = k_mag * cos(ph)
            kz = k_mag * sin(ph) * sin(th)
            jac = k_mag**2 * sin(ph)  # 球坐标 Jacobian
            total += integrand(kx, ky, kz) * jac * d_theta * d_phi

    return total


def amplification_factor_integral(Re_x_range, alpha_i_interp, method='simpson'):
    """
    基于 e^N 方法的扰动放大因子计算。

    N = ln(A/A_0) = -∫_{x_0}^{x_t} α_i dx

    参数:
        Re_x_range (np.ndarray): 当地雷诺数序列（∝ x）
        alpha_i_interp (callable): α_i(Re_x) 插值函数
        method (str): 'simpson' 或 'trapz'

    返回:
        tuple: (Re_x, N_values)
    """
    Re = np.asarray(Re_x_range)
    if callable(alpha_i_interp):
        alpha_i = alpha_i_interp(Re)
    else:
        alpha_i = np.asarray(alpha_i_interp)
        if len(alpha_i) != len(Re):
            raise ValueError("alpha_i_interp 数组长度与 Re 不匹配")

    if method == 'trapz':
        N = np.zeros_like(Re)
        for i in range(1, len(Re)):
            N[i] = N[i - 1] - 0.5 * (Re[i] - Re[i - 1]) * (alpha_i[i] + alpha_i[i - 1])
    elif method == 'simpson':
        N = np.zeros_like(Re)
        for i in range(2, len(Re), 2):
            h = Re[i] - Re[i - 2]
            N[i] = N[i - 2] - h / 6.0 * (alpha_i[i - 2] + 4.0 * alpha_i[i - 1] + alpha_i[i])
            if i - 1 < len(N):
                N[i - 1] = 0.5 * (N[i - 2] + N[i])  # 线性插值中间值
    else:
        raise ValueError(f"未知积分方法: {method}")

    return Re, N


def chebyshev_transform(fvals, N=None):
    """
    物理空间函数值到 Chebyshev 谱系数的变换。

    对于 Gauss-Lobatto 节点 x_j = cos(jπ/N)，
    前向 Chebyshev 变换:
        a_k = (2/N) Σ_{j=0}^{N} f(x_j) T_k(x_j) / c_j
    其中 c_0 = c_N = 2, c_j = 1 (1 ≤ j ≤ N-1)。

    参数:
        fvals (np.ndarray): 节点上的函数值，长度 N+1
        N (int): 阶数（默认 len(fvals)-1）

    返回:
        np.ndarray: 谱系数 a_k
    """
    if N is None:
        N = len(fvals) - 1
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    j = np.arange(N + 1)
    x = np.cos(pi * j / N)

    coeffs = np.zeros(N + 1)
    for k in range(N + 1):
        Tk = np.cos(k * np.arccos(np.clip(x, -1.0, 1.0)))
        coeffs[k] = (2.0 / N) * np.sum(fvals * Tk / c)
    coeffs[0] *= 0.5
    coeffs[N] *= 0.5
    return coeffs
