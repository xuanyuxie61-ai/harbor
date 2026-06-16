#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quadrature_engine.py — 高阶数值积分引擎

对应种子项目:
  - 931_pyramid_felippa_rule: 金字塔域 Felippa 高阶求积公式
  - 946_quad2d: 二维区域高斯求积
  - 294_disk_integrals: 圆盘域积分与单项式积分

核心数学公式:
  高斯求积:
      ∫_Ω f(x) dx ≈ Σ_{i=1}^{N_q} w_i f(x_i)

  金字塔域积分 (Felippa 2004):
      Ω = {(x,y,z): -1+z ≤ x,y ≤ 1-z, 0 ≤ z ≤ 1}
      ∫_Ω x^a y^b z^c dV = 封闭公式

  圆盘域积分:
      ∫_{x²+y²≤R²} x^m y^n dA
      = (2/(m+n+2)) R^{m+n+2} · B((m+1)/2, (n+1)/2)
        × cos 因子 (m,n 奇偶性)

  2D Gauss-Legendre:
      ∫_{-1}^{1} ∫_{-1}^{1} f(x,y) dx dy ≈ Σ_i Σ_j w_i w_j f(x_i, y_j)
"""

import numpy as np
from typing import Tuple, Callable


# ---------------------------------------------------------------------------
# Gauss-Legendre 节点和权重
# ---------------------------------------------------------------------------
def gauss_legendre_1d(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    一维 Gauss-Legendre 求积节点与权重.

    基于 numpy.polynomial.legendre.leggauss.
    节点 x_i ∈ (-1,1), 权重 w_i > 0.

    代数精度: 2·order - 1

    Parameters
    ----------
    order : int — 求积阶数

    Returns
    -------
    (nodes, weights) — 各 shape (order,)
    """
    if order < 1:
        raise ValueError("求积阶数必须 >= 1")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def gauss_legendre_2d(order_x: int,
                      order_y: int) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
    """
    矩形域 [-1,1]² 上的 2D 张量积 Gauss 求积.

    节点: (x_i, y_j),  权重: w_i · w_j

    Parameters
    ----------
    order_x, order_y : int

    Returns
    -------
    (xy_nodes, w_combined, n_total)
        xy_nodes : shape (N, 2)
        w_combined : shape (N,)
        n_total : int = order_x * order_y
    """
    nx, wx = gauss_legendre_1d(order_x)
    ny, wy = gauss_legendre_1d(order_y)
    xx, yy = np.meshgrid(nx, ny, indexing='ij')
    ww_x, ww_y = np.meshgrid(wx, wy, indexing='ij')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    w = (ww_x * ww_y).ravel()
    return xy, w, len(w)


# ---------------------------------------------------------------------------
# 金字塔域 Felippa 求积 (来自 931)
# ---------------------------------------------------------------------------
def pyramid_unit_nodes_weights(rule_order: int
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    单位金字塔域上的求积规则.

    金字塔域: Ω = {(x,y,z): |x|,|y| ≤ 1-z, 0 ≤ z ≤ 1}
    体积: V = 4/3

    通过分层: 在 z_k 高度, 截面为边长 2(1-z_k) 的正方形.
    对每个 z 层, 使用 2D Gauss 规则在截面上积分.

    基于 Felippa (2004) "A compendium of FEM integration formulas" 的思想.

    Parameters
    ----------
    rule_order : int — z 方向的层数 (也是每层的 1D Gauss 阶数)

    Returns
    -------
    (points, weights):
        points  : shape (N, 3)
        weights : shape (N,)
    """
    if rule_order < 1:
        rule_order = 1

    # z 方向: 映射 [-1,1] → [0,1]
    nz, wz = gauss_legendre_1d(rule_order)
    z_pts = 0.5 * (nz + 1.0)
    z_wts = 0.5 * wz

    # 横向 Gauss 阶数随 z 层变化 — 至少 2 阶
    xy_order = max(2, rule_order)
    nxy, wxy = gauss_legendre_1d(xy_order)

    all_pts = []
    all_wts = []
    for k in range(len(z_pts)):
        zk = z_pts[k]
        wz_k = z_wts[k]
        # 截面: [-1+zk, 1-zk]²  边长 L = 2(1-zk)
        L = 2.0 * (1.0 - zk)
        if L < 1e-15:
            # z = 1, 截面积为零, 跳过
            continue
        # 映射 nxy ∈ [-1,1] → [-L/2, L/2]
        scale = L / 2.0
        xc = nxy * scale
        wc = wxy * scale
        # 2D 张量积
        xx, yy = np.meshgrid(xc, xc, indexing='ij')
        wx2, wy2 = np.meshgrid(wc, wc, indexing='ij')
        w2d = (wx2 * wy2).ravel()
        pts2d = np.column_stack([xx.ravel(), yy.ravel()])
        nz_layer = pts2d.shape[0]
        zk_arr = np.full(nz_layer, zk)
        pts3d = np.column_stack([pts2d, zk_arr])
        wts3d = w2d * wz_k
        all_pts.append(pts3d)
        all_wts.append(wts3d)

    points = np.vstack(all_pts)
    weights = np.concatenate(all_wts)
    return points, weights


def pyramid_monomial_integral(a: int, b: int, c: int) -> float:
    """
    单位金字塔上单项式 x^a · y^b · z^c 的精确积分.

    解析公式:
        ∫_Ω x^a y^b z^c dV
        = I_x(a) · I_y(b) · I_z(c, a, b)

    其中:
        I_x(a) = ∫_{-1}^{1} x^a dx = (1+(-1)^a)/(a+1)  (a 偶数 ≠ 0, 否则 0)
        I_z(c, a, b) = ∫_0^1 z^c (1-z)^{a+b+2} dz
                      = B(c+1, a+b+3)  (Beta 函数)

    Parameters
    ----------
    a, b, c : int — 幂次

    Returns
    -------
    float
    """
    from math import gamma

    def I_1d(p):
        if p % 2 == 1:
            return 0.0
        return 2.0 / (p + 1)

    # I_x 和 I_y 贡献 (关于 z 的依赖在截面积分中)
    if a % 2 == 1 or b % 2 == 1:
        return 0.0

    # z 积分: ∫_0^1 z^c · (1-z)^{a+b+2} · (1-z)^a · (1-z)^b dz
    # 注意: x 的截面积分给出 (2(1-z))^{a+1}/(a+1) 等因子
    # 完整公式:
    # ∫_Ω x^a y^b z^c dV =
    #   [2/(a+1)] · [2/(b+1)] · ∫_0^1 z^c · (1-z)^{a+b+2} dz
    #             × (对偶数 a,b 的额外因子, 这里 a,b 已为偶数)

    # Beta 函数: B(c+1, a+b+3) = Γ(c+1)Γ(a+b+3)/Γ(a+b+c+4)
    beta_val = (gamma(c + 1) * gamma(a + b + 3)
                / gamma(a + b + c + 4))
    result = (2.0 / (a + 1)) * (2.0 / (b + 1)) * beta_val
    return result


# ---------------------------------------------------------------------------
# 圆盘域积分 (来自 294)
# ---------------------------------------------------------------------------
def disk_monomial_integral(m: int, n: int, R: float = 1.0) -> float:
    """
    圆盘域 x² + y² ≤ R² 上 x^m · y^n 的积分.

    解析公式 (极坐标):
        ∫_{D_R} x^m y^n dA
        = ∫_0^R ∫_0^{2π} (r cosθ)^m (r sinθ)^n r dθ dr
        = ∫_0^R r^{m+n+1} dr · ∫_0^{2π} cos^m θ · sin^n θ dθ

    角度积分:
        ∫_0^{2π} cos^m θ · sin^n θ dθ
        = 4 · B((m+1)/2, (n+1)/2)    (m, n 均为偶数)
        = 0                            (m 或 n 为奇数)

    径向积分:
        ∫_0^R r^{m+n+1} dr = R^{m+n+2} / (m+n+2)

    Parameters
    ----------
    m, n : int — 幂次
    R      : float — 半径

    Returns
    -------
    float
    """
    from math import gamma, pi

    if m % 2 == 1 or n % 2 == 1:
        return 0.0

    radial = R ** (m + n + 2) / (m + n + 2)

    # 角度积分 (全周期, m,n 均为偶数):
    # ∫_0^{2π} cos^m θ sin^n θ dθ
    # = 4 ∫_0^{π/2} cos^m θ sin^n θ dθ
    # = 2 B((m+1)/2, (n+1)/2)
    angular = 2.0 * gamma((m + 1) / 2) * gamma((n + 1) / 2) / gamma((m + n + 2) / 2)

    return radial * angular


def disk_sample_uniform(n_samples: int, R: float = 1.0,
                        seed: int = 0) -> np.ndarray:
    """
    在圆盘域内均匀采样.

    使用拒绝采样或极坐标变换:
        r = R √U,  θ = 2π V,  U,V ~ Uniform(0,1)

    Parameters
    ----------
    n_samples : int
    R         : float
    seed      : int

    Returns
    -------
    ndarray, shape (n_samples, 2)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n_samples)
    v = rng.uniform(0, 1, n_samples)
    r = R * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# 通用积分器
# ---------------------------------------------------------------------------
def integrate_2d(func: Callable,
                 domain: str = 'rect',
                 order: int = 5,
                 R: float = 1.0,
                 **kwargs) -> float:
    """
    2D 通用数值积分.

    Parameters
    ----------
    func   : callable (x, y) → float
    domain : 'rect' | 'disk' | 'pyramid_slice'
    order  : 求积阶数
    R      : 圆盘半径 / 矩形半宽

    Returns
    -------
    float — 积分近似值
    """
    if domain == 'rect':
        xy, w, _ = gauss_legendre_2d(order, order)
        # 映射到 [-R, R]²
        xy = xy * R
        w = w * R * R
        vals = np.array([func(p[0], p[1]) for p in xy])
        return float(np.dot(w, vals))
    elif domain == 'disk':
        # 极坐标 Gauss 求积
        nr, wr = gauss_legendre_1d(order)
        nt, wt = gauss_legendre_1d(order)
        # 映射 r ∈ [-1,1] → [0, R]
        r_pts = R * 0.5 * (nr + 1)
        r_wts = R * 0.5 * wr
        # θ ∈ [-1,1] → [0, 2π]
        th_pts = np.pi * (nt + 1)
        th_wts = np.pi * wt

        total = 0.0
        for i in range(len(r_pts)):
            for j in range(len(th_pts)):
                ri = r_pts[i]
                thj = th_pts[j]
                x = ri * np.cos(thj)
                y = ri * np.sin(thj)
                w_ij = r_wts[i] * th_wts[j] * ri  # Jacobian: r
                total += w_ij * func(x, y)
        return float(total)
    elif domain == 'pyramid_slice':
        pts, wts = pyramid_unit_nodes_weights(order)
        vals = np.array([func(p[0], p[1]) for p in pts])
        return float(np.dot(wts, vals))
    else:
        raise ValueError(f"未知域类型: {domain}")
