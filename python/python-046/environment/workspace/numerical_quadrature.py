"""
numerical_quadrature.py
数值积分模块。
融合种子项目 942_quad_parfor（复合梯形积分）的核心思想，
并扩展至高斯-勒让德（Gauss-Legendre）和三角形高斯求积规则。

在 InSAR 形变反演中，数值积分用于：
1. Okada 位错格林函数的面积分（断层面上的积分）
2. 有限元刚度矩阵和质量矩阵的单元积分
3. 目标泛函中观测残差的 L2 范数计算
"""

import numpy as np
from utils import check_finite


def composite_trapezoidal(f, a, b, n):
    """
    复合梯形数值积分公式。

    积分近似：
        I ≈ h * [ 0.5*f(x_0) + Σ_{i=1}^{n-2} f(x_i) + 0.5*f(x_{n-1}) ]
    其中 h = (b - a) / (n - 1), x_i = a + i * h。

    参数:
        f: 被积函数，接受 ndarray 返回 ndarray
        a, b: 积分区间
        n:  采样点数 (n >= 2)

    返回:
        积分近似值
    """
    if n < 2:
        raise ValueError("composite_trapezoidal: n must be >= 2")
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    fx = f(x)
    check_finite(fx, "composite_trapezoidal fx")
    val = 0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1]
    return val * h


def gauss_legendre_nodes_weights(n):
    """
    计算 Gauss-Legendre 求积节点和权重。
    利用 numpy 的 leggauss，其节点 x_i 和权重 w_i 满足：
        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i * f(x_i)
    精度：2n-1 次多项式精确。
    """
    if n < 1:
        raise ValueError("gauss_legendre_nodes_weights: n must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def gauss_legendre_integral(f, a, b, n):
    """
    在 [a, b] 上使用 n 点 Gauss-Legendre 求积计算积分。
    通过变量替换 x = (b+a)/2 + (b-a)/2 * t，t ∈ [-1, 1]，得：
        ∫_a^b f(x) dx = (b-a)/2 * Σ w_i * f( (b+a)/2 + (b-a)/2 * t_i )
    """
    t, w = gauss_legendre_nodes_weights(n)
    x = 0.5 * (b + a) + 0.5 * (b - a) * t
    fx = f(x)
    check_finite(fx, "gauss_legendre_integral fx")
    return 0.5 * (b - a) * np.sum(w * fx)


def triangle_gauss_rule(order):
    """
    返回参考三角形 {(s,t): s>=0, t>=0, s+t<=1} 上的高斯求积节点和权重。
    融合种子项目 114_box_flow 中 quad_rule 的 7 点规则思想。

    参数:
        order: 当前支持 1, 3, 7 点规则

    返回:
        xy: shape (order, 2), 参考三角形内的节点坐标
        w:  shape (order,), 权重（总和为 1/2，对应参考三角形面积）
    """
    if order == 1:
        xy = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        w = np.array([0.5])
    elif order == 3:
        xy = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]
        ])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 7:
        a = 1.0 / 3.0
        b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        c = (6.0 - np.sqrt(15.0)) / 21.0
        d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        e = (6.0 + np.sqrt(15.0)) / 21.0
        u = 0.225
        v = (155.0 - np.sqrt(15.0)) / 1200.0
        ww = (155.0 + np.sqrt(15.0)) / 1200.0
        xy = np.array([
            [a, a],
            [b, c], [c, b], [c, c],
            [d, e], [e, d], [e, e]
        ])
        w = 0.5 * np.array([u, v, v, v, ww, ww, ww])
    else:
        raise ValueError(f"triangle_gauss_rule: unsupported order {order}")
    return xy, w


def integrate_over_triangle(f, p1, p2, p3, order=7):
    """
    在物理三角形 (p1, p2, p3) 上积分标量函数 f(x, y)。

    映射: (s, t) → x = p1 + (p2-p1)*s + (p3-p1)*t
    Jacobian 行列式 |J| = 2 * Area。
    """
    from utils import compute_triangle_area
    area = compute_triangle_area(p1, p2, p3)
    if area < 1e-14:
        return 0.0
    xy_ref, w = triangle_gauss_rule(order)
    # 参考节点映射到物理节点
    pts = (p1[None, :] +
           (p2 - p1)[None, :] * xy_ref[:, 0:1] +
           (p3 - p1)[None, :] * xy_ref[:, 1:2])
    vals = np.array([f(pt[0], pt[1]) for pt in pts])
    check_finite(vals, "integrate_over_triangle vals")
    return 2.0 * area * np.sum(w * vals)


def integrate_2d_grid(f, xlim, ylim, nx, ny, method='trapezoidal'):
    """
    在二维矩形区域上使用复合梯形或 Simpson 规则积分。
    """
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    dx = (xlim[1] - xlim[0]) / (nx - 1)
    dy = (ylim[1] - ylim[0]) / (ny - 1)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    check_finite(Z, "integrate_2d_grid Z")
    if method == 'trapezoidal':
        # 复合梯形: 对边界点权重为 0.25, 边为 0.5, 内部为 1.0
        W = np.ones((ny, nx))
        W[0, :] = 0.5
        W[-1, :] = 0.5
        W[:, 0] = 0.5
        W[:, -1] = 0.5
        W[0, 0] = 0.25
        W[0, -1] = 0.25
        W[-1, 0] = 0.25
        W[-1, -1] = 0.25
        return dx * dy * np.sum(W * Z)
    else:
        raise ValueError(f"integrate_2d_grid: unknown method {method}")
