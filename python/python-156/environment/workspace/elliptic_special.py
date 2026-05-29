"""
elliptic_special.py
===================
基于不完全椭圆积分的火焰曲率与几何效应计算模块。

核心算法源自 toms577 (rf function, Project 1274)，并改造用于
计算火焰前锋曲率相关的特殊函数。

不完全椭圆积分第一类 RF(X,Y,Z)：

    RF(X,Y,Z) = (1/2) ∫_0^∞ dt / [ sqrt(t+X) sqrt(t+Y) sqrt(t+Z) ]

其中 X, Y, Z ≥ 0 且最多一个为零。

在火焰面理论中，火焰前锋的曲率 κ 影响局部火焰速度（Markstein 效应）：

    S_n = S_L (1 - L_M κ)

其中 L_M 为 Markstein 长度。对于轴对称火焰前锋，曲率可表示为：

    κ = 1/r * dr/ds + d²r/ds²

这可以转化为包含椭圆积分的形式。此外，火焰前锋包围的体积计算
也涉及椭圆积分。

Carlson 对称形式 RF(X,Y,Z) 的计算采用倍增定理（duplication theorem）：

    迭代直至变量几乎相等：
        λ_n = sqrt(X_n Y_n) + sqrt(Y_n Z_n) + sqrt(Z_n X_n)
        X_{n+1} = (X_n + λ_n) / 4
        Y_{n+1} = (Y_n + λ_n) / 4
        Z_{n+1} = (Z_n + λ_n) / 4

    然后五阶 Taylor 展开：
        μ = (X+Y+Z) / 3
        ε = max |2 - (μ+X)/μ|
        s = 1 + (e2/24 - 0.1 - 3e3/44) * e2 + e3/14
        RF = s / sqrt(μ)

Markstein 长度与曲率的关系（对于薄火焰）：
    L_M = δ_L * [ Le ln(1/Le) / (Le - 1) ]

其中 Le 为 Lewis 数，δ_L = α / S_L 为层流火焰厚度。
"""

import numpy as np


def rf_elliptic_integral(x, y, z, errtol=1.0e-3):
    """
    计算 Carlson 对称形式的不完全椭圆积分 RF(X,Y,Z)。

    算法源自 toms577 (Project 1274)，采用倍增定理和五阶 Taylor 展开。

    Parameters
    ----------
    x, y, z : float
        积分参数，必须非负且至多一个为零。
    errtol : float
        误差容差。

    Returns
    -------
    value : float
        RF(X,Y,Z) 的值。
    ierr : int
        错误标志，0 表示成功，1 表示输入错误。
    """
    lolim = 3.0e-78
    uplim = 1.0e+75

    if (x < 0.0 or y < 0.0 or z < 0.0 or
        x + y < lolim or x + z < lolim or y + z < lolim or
        uplim <= x or uplim <= y or uplim <= z):
        return 0.0, 1

    xn = float(x)
    yn = float(y)
    zn = float(z)

    while True:
        mu = (xn + yn + zn) / 3.0
        xndev = 2.0 - (mu + xn) / mu
        yndev = 2.0 - (mu + yn) / mu
        zndev = 2.0 - (mu + zn) / mu
        epslon = max(abs(xndev), max(abs(yndev), abs(zndev)))

        if epslon < errtol:
            c1 = 1.0 / 24.0
            c2 = 3.0 / 44.0
            c3 = 1.0 / 14.0
            e2 = xndev * yndev - zndev * zndev
            e3 = xndev * yndev * zndev
            s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3
            value = s / np.sqrt(mu)
            return value, 0

        xnroot = np.sqrt(xn)
        ynroot = np.sqrt(yn)
        znroot = np.sqrt(zn)
        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        xn = (xn + lamda) * 0.25
        yn = (yn + lamda) * 0.25
        zn = (zn + lamda) * 0.25


def flame_curvature_elliptic(a, b, c_axis, s_param):
    """
    使用椭圆积分计算椭球火焰前锋的曲率。

    对于椭球面 x²/a² + y²/b² + z²/c² = 1，其表面积元素涉及椭圆积分。

    Parameters
    ----------
    a, b, c_axis : float
        椭球半轴长。
    s_param : float
        弧长参数。

    Returns
    -------
    curvature : float
        局部曲率。
    area_element : float
        面积元。
    """
    # 归一化参数
    a = max(a, 1.0e-12)
    b = max(b, 1.0e-12)
    c_axis = max(c_axis, 1.0e-12)

    # 使用 Carlson RF 计算近似曲率
    x_rf = (b / a) ** 2
    y_rf = (c_axis / a) ** 2
    z_rf = 1.0

    rf_val, ierr = rf_elliptic_integral(x_rf, y_rf, z_rf, errtol=1.0e-3)
    if ierr != 0:
        rf_val = 1.0

    # 近似曲率
    curvature = 1.0 / rf_val * (1.0 / a + 1.0 / b) / 2.0

    # 面积元（简化）
    area_element = 4.0 * np.pi * a * b * rf_val

    return curvature, area_element


def markstein_length(Le, alpha_diff=2.0e-5, S_L=0.4):
    """
    计算 Markstein 长度。

    公式（Clavin & Williams, 1982）：
        L_M = δ_L * [ Le * ln(1/Le) / (Le - 1) ]  (Le ≠ 1)
        L_M = δ_L                               (Le = 1)

    其中 δ_L = α / S_L 为层流火焰厚度。

    Parameters
    ----------
    Le : float
        Lewis 数。
    alpha_diff : float
        热扩散系数，m²/s。
    S_L : float
        层流火焰速度，m/s。

    Returns
    -------
    L_M : float
        Markstein 长度，m。
    """
    delta_L = alpha_diff / S_L if S_L > 0 else 1.0e-3

    if abs(Le - 1.0) < 1.0e-6:
        L_M = delta_L
    else:
        # 使用泰勒展开避免 log(1/Le) 的奇异性
        if Le <= 0.0:
            Le = 1.0e-6
        L_M = delta_L * (Le * np.log(1.0 / Le) / (Le - 1.0))

    return L_M


def curved_flame_speed(S_L, curvature, Le, alpha_diff=2.0e-5):
    """
    计算考虑曲率效应的局部火焰速度（Markstein 修正）。

    公式：
        S_n = S_L * (1 - L_M * κ)

    Parameters
    ----------
    S_L : float
        层流火焰速度。
    curvature : float
        局部曲率，m⁻¹。
    Le : float
        Lewis 数。
    alpha_diff : float
        热扩散系数。

    Returns
    -------
    S_n : float
        局部法向火焰速度。
    """
    L_M = markstein_length(Le, alpha_diff, S_L)

    # 数值鲁棒性：限制曲率修正幅度
    correction = 1.0 - L_M * curvature
    correction = np.clip(correction, 0.3, 3.0)

    S_n = S_L * correction
    return S_n
