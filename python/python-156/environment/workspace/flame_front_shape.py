"""
flame_front_shape.py
====================
基于通用蛋形曲线的火焰前锋几何参数化模块。

核心算法源自 universal_egg / pyriform_egg / chicken_egg (Project 093)，
并改造用于描述湍流燃烧中火焰前锋（flame front）的曲面形状。

在湍流燃烧中，火焰前锋的皱褶（wrinkling）对火焰传播速度有重要影响。
Damköhler (1940) 和 Peters (1986) 的火焰面理论指出，湍流火焰速度
S_T 与层流火焰速度 S_L 的关系为：

    S_T / S_L = A_T / A_L

其中 A_T 为皱褶火焰前锋的总面积，A_L 为投影面积。

火焰前锋的局部形状可用类蛋形曲线参数化（Narushin et al., 2021）：

    r(x) = (B/2) * sqrt( (L² - 4x²) / (L² + 8wx + 4w²) )

其中：
    B 为最大宽度（火焰前锋横向尺度），
    L 为长度（火焰前锋纵向尺度），
    w 为最大宽度位置偏移量（描述火焰前锋不对称性）。

通用蛋形修正（考虑液滴/气泡变形）：
    r_universal(x) = r_chicken(x) * [1 - t2 * t3(x)]

其中 t2 和 t3 为与直径 D（在 x = L/4 处）相关的修正系数。

在燃烧应用中，我们引入 Karlovitz 数 Ka 来修正形状参数：
    Ka = (u' / S_L)² * (δ_L / l_t)

其中 u' 为湍流脉动速度，δ_L 为层流火焰厚度，l_t 为湍流积分尺度。
修正后的参数：
    B_eff = B * (1 + 0.1 * Ka)
    w_eff = w * exp(-Ka / 10)
"""

import numpy as np


def chicken_egg_shape(B, L, w, x):
    """
    基本蛋形曲线（对称/不对称椭圆修正）。

    公式：
        r(x) = (B/2) * sqrt( (L² - 4x²) / (L² + 8wx + 4w²) )

    Parameters
    ----------
    B : float
        最大宽度。
    L : float
        总长度。
    w : float
        最大宽度位置偏移。
    x : ndarray
        轴向坐标，范围 [-L/2, L/2]。

    Returns
    -------
    r : ndarray
        径向坐标。
    """
    x = np.clip(x, -L / 2.0 + 1.0e-9, L / 2.0 - 1.0e-9)
    numerator = L ** 2 - 4.0 * x ** 2
    denominator = L ** 2 + 8.0 * w * x + 4.0 * w ** 2

    # 鲁棒性处理
    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(denominator, 1.0e-12)

    r = 0.5 * B * np.sqrt(numerator / denominator)
    return r


def pyriform_egg_shape(B, L, w, x):
    """
    梨形蛋曲线（更尖锐的一端）。

    公式：
        r(x) = (B/2) * sqrt( L(L² - 4x²) / [2(L-2w)x² + (L²+8Lw-4w²)x + 2Lw²+L²w+L³] )

    Parameters
    ----------
    B : float
        最大宽度。
    L : float
        总长度。
    w : float
        最大宽度位置偏移。
    x : ndarray
        轴向坐标。

    Returns
    -------
    r : ndarray
        径向坐标。
    """
    x = np.clip(x, -L / 2.0 + 1.0e-9, L / 2.0 - 1.0e-9)
    t1 = (L ** 2 - 4.0 * x ** 2) * L
    t2 = (2.0 * (L - 2.0 * w) * x ** 2 +
          (L ** 2 + 8.0 * L * w - 4.0 * w ** 2) * x +
          2.0 * L * w ** 2 + L ** 2 * w + L ** 3)

    t1 = np.maximum(t1, 0.0)
    t2 = np.maximum(t2, 1.0e-12)

    r = 0.5 * B * np.sqrt(t1 / t2)
    return r


def universal_egg_shape(B, L, w, D, x):
    """
    通用蛋形曲线（含四分之一长度处直径 D 的修正）。

    公式：
        s1 = sqrt(5.5 L² + 11Lw + 4w²)
        s2 = sqrt(3) B L - 2D sqrt(L² + 2wL + 4w²)
        t2 = (s1 * s2) / (sqrt(3) * (s3 - s4))
        s5 = L(L² + 8wx + 4w²)
        s6 = 2(L-2w)x² + (L²+8Lw-4w²)x + 2Lw²+L²w+L³
        t3 = 1 - sqrt(s5 / s6)
        r = r_chicken * (1 - t2 * t3)

    Parameters
    ----------
    B, L, w : float
        基本蛋形参数。
    D : float
        x = L/4 处的直径。
    x : ndarray
        轴向坐标。

    Returns
    -------
    r : ndarray
        径向坐标。
    """
    r_chicken = chicken_egg_shape(B, L, w, x)

    s1 = np.sqrt(5.5 * L ** 2 + 11.0 * L * w + 4.0 * w ** 2)
    s2 = (np.sqrt(3.0) * B * L -
          2.0 * D * np.sqrt(L ** 2 + 2.0 * w * L + 4.0 * w ** 2))
    s3 = s1
    s4 = 2.0 * np.sqrt(L ** 2 + 2.0 * w * L + 4.0 * w ** 2)

    denom_t2 = np.sqrt(3.0) * (s3 - s4)
    denom_t2 = np.where(np.abs(denom_t2) < 1.0e-12, 1.0e-12, denom_t2)
    t2 = (s1 * s2) / denom_t2

    s5 = L * (L ** 2 + 8.0 * w * x + 4.0 * w ** 2)
    s6 = (2.0 * (L - 2.0 * w) * x ** 2 +
          (L ** 2 + 8.0 * L * w - 4.0 * w ** 2) * x +
          2.0 * L * w ** 2 + L ** 2 * w + L ** 3)

    s5 = np.maximum(s5, 0.0)
    s6 = np.maximum(s6, 1.0e-12)
    t3 = 1.0 - np.sqrt(s5 / s6)

    r = r_chicken * (1.0 - t2 * t3)
    r = np.maximum(r, 0.0)
    return r


def flame_front_surface_area(B, L, w, Ka=0.0, num_points=200):
    """
    计算火焰前锋的皱褶表面积。

    旋转体表面积公式：
        A = 2π ∫_{-L/2}^{L/2} r(x) sqrt(1 + (dr/dx)²) dx

    Parameters
    ----------
    B, L, w : float
        蛋形参数。
    Ka : float
        Karlovitz 数，用于修正形状。
    num_points : int
        积分离散点数。

    Returns
    -------
    area : float
        火焰前锋表面积。
    area_ratio : float
        表面积与投影面积之比 S_T/S_L。
    """
    # Karlovitz 修正
    B_eff = B * (1.0 + 0.1 * Ka)
    w_eff = w * np.exp(-Ka / 10.0)
    L_eff = L * (1.0 + 0.05 * Ka)

    x = np.linspace(-L_eff / 2.0, L_eff / 2.0, num_points)
    dx = x[1] - x[0]

    r = chicken_egg_shape(B_eff, L_eff, w_eff, x)

    # 数值微分（中心差分）
    dr_dx = np.zeros_like(r)
    dr_dx[1:-1] = (r[2:] - r[:-2]) / (2.0 * dx)
    dr_dx[0] = (r[1] - r[0]) / dx
    dr_dx[-1] = (r[-1] - r[-2]) / dx

    integrand = 2.0 * np.pi * r * np.sqrt(1.0 + dr_dx ** 2)
    area = np.trapezoid(integrand, x)

    # 投影面积
    area_projected = np.pi * (B_eff / 2.0) ** 2
    area_ratio = area / area_projected if area_projected > 0 else 1.0

    return area, area_ratio
