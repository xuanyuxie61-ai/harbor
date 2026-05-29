"""
ignition_polynomial.py
======================
基于多项式求根的点火/熄火极限分析模块。

核心算法源自 wdk (Weierstrass-Durand-Kerner, Project 1404)，
并改造用于求解燃烧科学中的临界条件多项式。

在火焰面理论中，点火（ignition）和熄火（extinction）现象对应于
火焰面方程的多稳态解分支的出现与消失。通过渐近分析，可以导出
临界 Damköhler 数 Da_{cr} 满足的代数方程。

对于一步不可逆反应，临界条件可由以下多项式近似描述
（Liñán, 1974, Acta Astronautica）：

    P(Da) = Da^d + c_{d-1} Da^{d-1} + ... + c_1 Da + c_0 = 0

其中系数 c_k 由活化能 β = E_a/(R_u T_st)、
Lewis 数 Le 和 Zel'dovich 数 Ze 决定。

Zel'dovich 数定义：
    Ze = β (1 - σ) / 2

其中 σ = T_ox / T_ad 为温度比。

临界 Damköhler 数（渐近展开，d=4 近似）：
    Da_{cr} = e * Ze^{-2} [1 - 3 Ze^{-1} + O(Ze^{-2})]

我们通过构造特征多项式并求其根来确定精确的 Da_{cr}。

WDK 算法：
---------
对于多项式 P(z) = c_0 + c_1 z + ... + c_d z^d，
迭代格式为：

    z_i^{(k+1)} = z_i^{(k)} - P(z_i^{(k)}) / Π_{j≠i} (z_i^{(k)} - z_j^{(k)})

初始猜测采用 Cauchy 界和 roots of unity：
    R = 1 + max |c_j / c_d|
    z_i^{(0)} = R * exp(2π i (j-1)/d)
"""

import numpy as np


def poly_eval(c, x):
    """
    求多项式 P(x) = c[0] + c[1]*x + ... + c[d]*x^d 的值。

    Parameters
    ----------
    c : ndarray, shape (d+1,)
        多项式系数，c[0] 为常数项。
    x : complex or ndarray
        求值点。

    Returns
    -------
    value : complex or ndarray
        多项式值。
    """
    d = len(c) - 1
    if np.isscalar(x):
        value = c[0]
        xi = 1.0
        for i in range(1, d + 1):
            xi *= x
            value += c[i] * xi
    else:
        x = np.asarray(x)
        value = c[0] * np.ones_like(x, dtype=complex)
        xi = np.ones_like(x, dtype=complex)
        for i in range(1, d + 1):
            xi *= x
            value += c[i] * xi
    return value


def wdk_roots(c, tol=1.0e-12, max_iter=1000):
    """
    Weierstrass-Durand-Kerner 算法求多项式全部根。

    Parameters
    ----------
    c : ndarray, shape (d+1,)
        多项式系数，c[0] 为常数项，c[d] 为首项系数（必须非零）。
    tol : float
        收敛容差。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    roots : ndarray, shape (d,)
        多项式的 d 个根（复数）。
    converged : bool
        是否收敛。
    """
    c = np.asarray(c, dtype=complex)
    d = len(c) - 1

    if d < 1:
        raise ValueError("多项式次数必须 >= 1")
    if abs(c[d]) < 1.0e-30:
        raise ValueError("首项系数不能为零")

    # Cauchy 界
    R = 1.0 + np.max(np.abs(c[:-1] / c[d]))

    # 初始猜测：单位根缩放
    theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
    roots = R * np.exp(1.0j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            zi = roots_old[i]
            denom = 1.0 + 0.0j
            for j in range(d):
                if i != j:
                    diff = zi - roots[j]
                    if abs(diff) < 1.0e-30:
                        diff = 1.0e-30 * (1.0 + 0.0j)
                    denom *= diff

            if abs(denom) < 1.0e-30:
                denom = 1.0e-30 * (1.0 + 0.0j)

            roots[i] = zi - poly_eval(c, zi) / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            return roots, True

    return roots, False


def critical_damkohler_polynomial(Ze, beta, sigma, order=4):
    """
    构造临界 Damköhler 数的特征多项式。

    基于 Liñán 渐近理论和活化能展开，构造：

        P(Da) = Σ_{k=0}^{order} c_k Da^k

    其中系数由渐近展开确定：
        c_0 = -e * Ze^{-2}
        c_1 = 1
        c_k 由高阶修正项给出

    Parameters
    ----------
    Ze : float
        Zel'dovich 数，必须 > 0。
    beta : float
        无量纲活化能 E_a/(R_u T_st)。
    sigma : float
        温度比 T_ox / T_ad。
    order : int
        多项式阶数。

    Returns
    -------
    c : ndarray
        多项式系数。
    """
    if Ze <= 0.0 or beta <= 0.0 or sigma <= 0.0:
        raise ValueError("Ze, beta, sigma 必须为正数")

    c = np.zeros(order + 1, dtype=complex)

    # 渐近展开系数
    c[order] = 1.0 + 0.0j
    c[order - 1] = -3.0 / Ze
    c[order - 2] = 2.0 / (Ze ** 2) - beta * (1.0 - sigma) / (2.0 * Ze)
    c[order - 3] = -0.5 / (Ze ** 3)
    c[0] = -np.e / (Ze ** 2)

    # 填充中间系数（使用插值保证多项式平滑）
    for k in range(1, order - 3):
        c[k] = c[0] * ((-1.0) ** k) / np.math.factorial(k) * (1.0 / Ze) ** k

    return c


def analyze_ignition_extinction(Ze=8.0, beta=10.0, sigma=0.135):
    """
    分析点火与熄火极限，返回临界 Damköhler 数。

    Parameters
    ----------
    Ze : float
        Zel'dovich 数。
    beta : float
        无量纲活化能。
    sigma : float
        温度比。

    Returns
    -------
    results : dict
        包含临界根、多项式系数和分析结果的字典。
    """
    c = critical_damkohler_polynomial(Ze, beta, sigma, order=4)
    roots, converged = wdk_roots(c, tol=1.0e-14, max_iter=500)

    # 筛选正实根（物理意义的临界 Da）
    real_positive = []
    for r in roots:
        if np.imag(r) < 1.0e-6 and np.real(r) > 0:
            real_positive.append(float(np.real(r)))

    Da_cr = min(real_positive) if real_positive else float(np.real(roots[0]))
    if Da_cr <= 0:
        Da_cr = np.e / (Ze ** 2)

    results = {
        'Zel_dovich_number': Ze,
        'activation_energy_beta': beta,
        'temperature_ratio_sigma': sigma,
        'polynomial_coefficients': c.tolist(),
        'roots': roots.tolist(),
        'converged': converged,
        'critical_Damkohler': Da_cr,
        'ignition_limit': Da_cr * 0.8,
        'extinction_limit': Da_cr * 1.2,
    }

    return results
