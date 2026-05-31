# -*- coding: utf-8 -*-
"""
moment_integrals.py
基于 square_integrals（正方形区域单重积分），
计算超表面方形贴片上的电磁场矩与数值积分验证。

核心科学问题：
  方形金属/介电贴片上的电流分布可展开为矩量法（MoM）基函数。
  贴片上的单重积分矩用于阻抗矩阵元素的解析计算：
      Z_{mn} = ∫∫_{S_m} ∫∫_{S_n} G(r,r') f_m(r) f_n(r') dS dS'
  其中 G 为格林函数，f_m 为基函数（如屋顶函数、多项式基）。

关键公式：
  1. 单位正方形上的单重积分（参考 square01_monomial_integral）:
       I = ∫_0^1 ∫_0^1 x^{e1} y^{e2} dx dy = 1 / ((e1+1)(e2+1))
  2. 对称正方形 [-1,1]^2 上的积分（参考 squaresym_monomial_integral）:
       I = ∫_{-1}^1 ∫_{-1}^1 x^{e1} y^{e2} dx dy
       若 e1 或 e2 为奇数，则 I = 0
       否则 I = 4 / ((e1+1)(e2+1))
  3. 数值矩（离散采样）:
       M_{pq} = Σ_i Σ_j w_i w_j x_i^p y_j^q f(x_i, y_j)
  4. Gauss-Legendre 求积节点与权重:
       在 [-1,1] 上，∫ f(x) dx ≈ Σ_i w_i f(x_i)
"""

import numpy as np


def square01_monomial_integral(exponents):
    """
    计算单位正方形 [0,1]×[0,1] 上的单重积分（参考 square01_monomial_integral）。

    参数:
        exponents: [e1, e2]，非负整数
    返回:
        integral: float
    """
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != 2:
        raise ValueError("exponents 必须为 [e1, e2]")
    if np.any(exponents < 0):
        raise ValueError("指数必须非负")
    integral = 1.0
    for e in exponents:
        integral /= float(e + 1)
    return integral


def squaresym_monomial_integral(exponents):
    """
    计算对称正方形 [-1,1]×[-1,1] 上的单重积分（参考 squaresym_monomial_integral）。

    若任一指数为奇数，则积分为 0（奇函数在对称区间）。
    否则：
        I = 4 / ((e1+1)(e2+1))
    """
    exponents = np.asarray(exponents, dtype=int)
    if exponents.shape[0] != 2:
        raise ValueError("exponents 必须为 [e1, e2]")
    if np.any(exponents < 0):
        raise ValueError("指数必须非负")
    if np.any(exponents % 2 == 1):
        return 0.0
    integral = 4.0
    for e in exponents:
        integral /= float(e + 1)
    return integral


def gauss_legendre_nodes_weights(n):
    """
    计算 [-1,1] 上的 Gauss-Legendre 求积节点与权重。
    采用 numpy.polynomial.legendre.leggauss。
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_2d_gauss_legendre(func, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), n=8):
    """
    使用二维 Gauss-Legendre 求积计算函数在给定矩形域上的积分。

    公式:
        ∫_{y0}^{y1} ∫_{x0}^{x1} f(x,y) dx dy
        ≈ ((x1-x0)/2) ((y1-y0)/2) Σ_i Σ_j w_i w_j f(ξ_i, η_j)
    其中 ξ_i, η_j 为映射后的节点。
    """
    x_nodes, x_weights = gauss_legendre_nodes_weights(n)
    y_nodes, y_weights = gauss_legendre_nodes_weights(n)

    # 仿射映射到 [xlim] 和 [ylim]
    x_mapped = 0.5 * (xlim[1] - xlim[0]) * x_nodes + 0.5 * (xlim[0] + xlim[1])
    y_mapped = 0.5 * (ylim[1] - ylim[0]) * y_nodes + 0.5 * (ylim[0] + ylim[1])
    wx = 0.5 * (xlim[1] - xlim[0]) * x_weights
    wy = 0.5 * (ylim[1] - ylim[0]) * y_weights

    integral = 0.0
    for i in range(n):
        for j in range(n):
            integral += wx[i] * wy[j] * func(x_mapped[i], y_mapped[j])
    return integral


def compute_field_moments(field_func, max_order, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5)):
    """
    计算方形贴片上的场矩：M_{pq} = ∫∫ x^p y^q f(x,y) dx dy。
    返回二维数组 moments[p, q]。
    """
    moments = np.zeros((max_order + 1, max_order + 1), dtype=float)
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            def integrand(x, y):
                val = field_func(x, y)
                if not np.isfinite(val):
                    return 0.0
                return (x ** p) * (y ** q) * val
            moments[p, q] = integrate_2d_gauss_legendre(
                integrand, xlim=xlim, ylim=ylim, n=max(4, max_order + 2)
            )
    return moments


def verify_monomial_integrals(max_order):
    """
    验证解析单重积分与数值积分的一致性（参考 square_integrals 的测试逻辑）。
    返回最大相对误差。
    """
    max_rel_error = 0.0
    for e1 in range(max_order + 1):
        for e2 in range(max_order + 1):
            exponents = [e1, e2]
            analytic = square01_monomial_integral(exponents)

            def f(x, y):
                return (x ** e1) * (y ** e2)

            numerical = integrate_2d_gauss_legendre(f, xlim=(0.0, 1.0), ylim=(0.0, 1.0), n=8)
            if abs(analytic) > 1e-12:
                rel_error = abs(analytic - numerical) / abs(analytic)
            else:
                rel_error = abs(numerical)
            if rel_error > max_rel_error:
                max_rel_error = rel_error
    return max_rel_error


def patch_impedance_moment(patch_size, wavelength, order=2):
    """
    计算方形贴片在自由空间中的自阻抗矩（简化模型）。
    基于矩量法中的 Galerkin 测试：
        Z_self ≈ (k0 η0 / (4π)) ∬∬ G(r,r') f(r) f(r') dS dS'
    这里用多项式矩近似：
        M = ∫∫ f(x,y) dx dy
    返回归一化矩值。
    """
    k0 = 2.0 * np.pi / wavelength
    # 简化的屋顶函数基
    def roof_func(x, y):
        # 在 [-a/2, a/2] 上为三角形分布
        a = patch_size
        fx = max(0.0, 1.0 - 2.0 * abs(x) / a)
        fy = max(0.0, 1.0 - 2.0 * abs(y) / a)
        return fx * fy

    moments = compute_field_moments(roof_func, order,
                                    xlim=(-patch_size / 2.0, patch_size / 2.0),
                                    ylim=(-patch_size / 2.0, patch_size / 2.0))
    return moments, k0
