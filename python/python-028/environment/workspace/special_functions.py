"""
special_functions.py
===================
核物理特殊函数计算模块

本模块实现了原子核壳模型计算中所需的核心特殊函数，包括：
1. 球贝塞尔函数 j_l(x) 与球诺伊曼函数 n_l(x) —— 用于径向薛定谔方程的解析解
2. 正弦积分 Si(x) —— 用于核格林函数与自能计算中的振荡积分正则化
3. 连带勒让德多项式 P_l^m(x) —— 用于球谐函数与形变势展开

数学基础：
- 球贝塞尔函数满足方程：x² y'' + 2x y' + [x² - l(l+1)] y = 0
- 正弦积分定义：Si(x) = ∫₀^x sin(t)/t dt
- 递推关系：j_{l+1}(x) = (2l+1)/x · j_l(x) - j_{l-1}(x)
"""

import numpy as np
from math import factorial, sqrt, pi, sin, cos, exp, log


def spherical_bessel_j(l, x):
    """
    计算球贝塞尔函数 j_l(x)。

    对于小参数 x → 0：j_l(x) ~ x^l / (2l+1)!!
    对于大参数 x → ∞：j_l(x) ~ sin(x - lπ/2) / x

    参数
    ----
    l : int
        角动量量子数，l >= 0
    x : float 或 np.ndarray
        自变量

    返回
    ----
    float 或 np.ndarray
        j_l(x) 的值
    """
    if l < 0:
        raise ValueError("角动量量子数 l 必须非负")
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    result = np.zeros_like(x)

    # 小参数级数展开：j_l(x) = x^l / (2l+1)!! · [1 - x²/(2(2l+3)) + ...]
    tiny = 1e-8
    small_mask = np.abs(x) < tiny
    if np.any(small_mask):
        xs = x[small_mask]
        double_fact = 1.0
        for k in range(1, 2 * l + 2, 2):
            double_fact *= k
        result[small_mask] = (xs ** l) / double_fact

    # 中等参数：向前递推
    med_mask = (~small_mask) & (np.abs(x) <= 30.0)
    if np.any(med_mask):
        xm = x[med_mask]
        j0 = np.sin(xm) / xm
        if l == 0:
            result[med_mask] = j0
        else:
            j1 = (np.sin(xm) / (xm ** 2)) - np.cos(xm) / xm
            if l == 1:
                result[med_mask] = j1
            else:
                j_prev2 = j0
                j_prev1 = j1
                for ll in range(1, l):
                    j_curr = (2 * ll + 1) / xm * j_prev1 - j_prev2
                    j_prev2 = j_prev1
                    j_prev1 = j_curr
                result[med_mask] = j_prev1

    # 大参数：渐近展开
    large_mask = np.abs(x) > 30.0
    if np.any(large_mask):
        xl = x[large_mask]
        result[large_mask] = np.sin(xl - l * pi / 2.0) / xl

    return result.item() if scalar_input else result.reshape(np.asarray(x).shape)


def spherical_neumann_n(l, x):
    """
    计算球诺伊曼函数 n_l(x)（第二类球贝塞尔函数）。

    小参数行为：n_l(x) ~ -(2l-1)!! / x^{l+1}
    大参数行为：n_l(x) ~ -cos(x - lπ/2) / x
    """
    if l < 0:
        raise ValueError("角动量量子数 l 必须非负")
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    result = np.zeros_like(x)

    tiny = 1e-8
    small_mask = np.abs(x) < tiny
    if np.any(small_mask):
        xs = x[small_mask]
        if l == 0:
            result[small_mask] = -1.0 / xs
        else:
            double_fact = 1.0
            for k in range(1, 2 * l, 2):
                double_fact *= k
            result[small_mask] = -double_fact / (xs ** (l + 1))

    med_mask = (~small_mask) & (np.abs(x) <= 30.0)
    if np.any(med_mask):
        xm = x[med_mask]
        n0 = -np.cos(xm) / xm
        if l == 0:
            result[med_mask] = n0
        else:
            n1 = -np.cos(xm) / (xm ** 2) - np.sin(xm) / xm
            if l == 1:
                result[med_mask] = n1
            else:
                n_prev2 = n0
                n_prev1 = n1
                for ll in range(1, l):
                    n_curr = (2 * ll + 1) / xm * n_prev1 - n_prev2
                    n_prev2 = n_prev1
                    n_prev1 = n_curr
                result[med_mask] = n_prev1

    large_mask = np.abs(x) > 30.0
    if np.any(large_mask):
        xl = x[large_mask]
        result[large_mask] = -np.cos(xl - l * pi / 2.0) / xl

    return result.item() if scalar_input else result.reshape(np.asarray(x).shape)


def sine_integral_si(x):
    """
    计算正弦积分 Si(x) = ∫₀^x sin(t)/t dt。

    该函数在核物理中用于处理格林函数中的对数发散与振荡积分。
    当 x → 0 时，Si(x) ~ x；当 x → ∞ 时，Si(x) → π/2。

    实现采用分段策略：
    - |x| ≤ 16：Chebyshev-like 级数展开
    - 16 < |x| ≤ 32：Bessel 函数展开
    - |x| > 32：渐近展开
    """
    x = float(x)
    p2 = pi / 2.0
    el = 0.5772156649015329
    epsilon = 1.0e-15
    x2 = x * x
    xabs = abs(x)
    xsign = -1.0 if x < 0.0 else 1.0

    if xabs == 0.0:
        return 0.0

    elif xabs <= 16.0:
        # 级数展开：Si(x) = Σ (-1)^k x^{2k+1} / [(2k+1)(2k+1)!]
        xr = xabs
        value = xabs
        for k in range(1, 40):
            xr = -0.5 * xr * (2 * k - 1) / k / (4 * k * k + 4 * k + 1) * x2
            value = value + xr
            if abs(xr) < abs(value) * epsilon:
                return xsign * value
        return xsign * value

    elif xabs <= 32.0:
        # 利用 Bessel 函数展开
        m = int(47.2 + 0.82 * xabs)
        bj = np.zeros(m)
        xa1 = 0.0
        xa0 = 1.0e-100
        for k in range(m - 1, -1, -1):
            xa = 4.0 * (k + 1) * xa0 / xabs - xa1
            bj[k] = xa
            xa1 = xa0
            xa0 = xa
        xs = bj[0]
        for k in range(2, m, 2):
            xs = xs + 2.0 * bj[k]
        bj[0] = bj[0] / xs
        for k in range(1, m):
            bj[k] = bj[k] / xs

        xr = 1.0
        xg1 = bj[0]
        for k in range(2, m):
            xr = 0.25 * xr * (2.0 * k - 3.0) ** 2 / ((k - 1.0) * (2.0 * k - 1.0) ** 2) * xabs
            xg1 = xg1 + bj[k] * xr

        xr = 1.0
        xg2 = bj[0]
        for k in range(2, m):
            xr = 0.25 * xr * (2.0 * k - 5.0) ** 2 / ((k - 1.0) * (2.0 * k - 3.0) ** 2) * xabs
            xg2 = xg2 + bj[k] * xr

        xcs = cos(xabs / 2.0)
        xss = sin(xabs / 2.0)
        value = xsign * (xabs * xcs * xg1 + 2.0 * xss * xg2 - sin(xabs))
        return value

    else:
        # 渐近展开
        xr = 1.0
        xf = 1.0
        for k in range(1, 10):
            xr = -2.0 * xr * k * (2 * k - 1) / x2
            xf = xf + xr
        xr = 1.0 / xabs
        xg = xr
        for k in range(1, 9):
            xr = -2.0 * xr * (2 * k + 1) * k / x2
            xg = xg + xr
        value = xsign * (p2 - xf * cos(xabs) / xabs - xg * sin(xabs) / xabs)
        return value


def associated_legendre(l, m, x):
    """
    计算连带勒让德多项式 P_l^m(x)。

    采用递推方法，利用 Bonnet 递推公式与连带递推关系：
    (l - m) P_l^m(x) = x(2l - 1) P_{l-1}^m(x) - (l + m - 1) P_{l-2}^m(x)

    在核壳模型中用于形变势的球谐展开：
    V(r, θ, φ) = Σ_{λμ} V_{λμ}(r) Y_{λμ}(θ, φ)
    其中 Y_{λμ} 与 P_λ^μ 直接相关。
    """
    if abs(m) > l:
        return 0.0
    if abs(x) > 1.0:
        raise ValueError("|x| 必须 ≤ 1 以保证勒让德多项式定义")

    m_abs = abs(m)
    # 使用 Ferrers 函数定义，x ∈ [-1, 1]
    pmm = 1.0
    if m_abs > 0:
        somx2 = sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m_abs + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m_abs:
        return pmm

    pmmp1 = x * (2 * m_abs + 1) * pmm
    if l == m_abs + 1:
        return pmmp1

    pll = 0.0
    for ll in range(m_abs + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m_abs - 1) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll

    return pmmp1 if l == m_abs + 1 else pll


def spherical_harmonic_Y(l, m, theta, phi):
    """
    计算球谐函数 Y_{lm}(θ, φ)。

    Y_{lm}(θ, φ) = (-1)^m √[(2l+1)(l-m)! / (4π(l+m)!)] P_l^m(cos θ) e^{imφ}

    该函数在核壳模型中用于：
    1. 单粒子波函数角向部分
    2. 形变势的多极展开
    3. 电磁跃迁矩阵元的计算
    """
    x = cos(theta)
    plm = associated_legendre(l, m, x)
    norm = sqrt((2 * l + 1) * factorial(l - abs(m)) / (4 * pi * factorial(l + abs(m))))
    phase = (-1) ** ((m + abs(m)) // 2)
    return phase * norm * plm * complex(cos(m * phi), sin(m * phi))


def nuclear_form_factor(q, A, Z, R0=1.2):
    """
    计算原子核形状因子 F(q)。

    基于球贝塞尔函数的解析表达式：
    F(q) = 4π ∫₀^∞ ρ(r) j₀(qr) r² dr

    对于均匀带电球模型：
    F(q) = 3 [sin(qR) - qR cos(qR)] / (qR)³

    参数
    ----
    q : float
        动量转移 (fm⁻¹)
    A : int
        质量数
    Z : int
        电荷数
    R0 : float
        核半径参数 (fm)

    返回
    ----
    float
        形状因子的模平方 |F(q)|²
    """
    R = R0 * (A ** (1.0 / 3.0))
    if abs(q) < 1e-10:
        return 1.0
    qr = q * R
    j0 = spherical_bessel_j(0, qr)
    # 更精确的 Fermi 分布形状因子使用数值积分，此处用简化解析式
    F = 3.0 * j0 / qr ** 2  # 近似
    return abs(F) ** 2
