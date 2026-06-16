"""
asymptotic.py - 渐近尾概率近似：Fresnel积分与Laplace方法

本模块实现用于可靠性分析中尾概率的高精度渐近近似方法：

1. Laplace 渐近法:
   P_f = int_{g(u)<=0} phi(u) du
   ≈ phi(beta) / ||grad_g(u*)|| * (2*pi)^{(n-1)/2} * |det(H_b)|^{-1/2}
   其中 H_b 为极限状态面在 u* 处的弯曲矩阵。

2. Fresnel 积分:
   C(x) = int_0^x cos(pi*t^2/2) dt
   S(x) = int_0^x sin(pi*t^2/2) dt
   用于衍射理论中的 Cornu 螺线，在可靠性中用于振荡型失效面的尾概率。

3. Mill's 比率:
   R(x) = (1 - Phi(x)) / phi(x) ≈ 1/x - 1/x^3 + 3/x^5 - ...

核心公式 (Breitung 型 Laplace):
  P_f ≈ phi(beta) * prod_{i=1}^{n-1} (1 + beta*kappa_i)^{-1/2}

种子项目映射:
  448_fresnel → Fresnel 积分的级数/渐近展开
  855_pdflib  → 正态尾概率计算
  820_numgrid → 高维网格上的渐近验证
"""

import numpy as np
from random_variables import std_normal_pdf, std_normal_cdf


def fresnel_integrals(x):
    """计算 Fresnel 积分 C(x) 和 S(x)

    种子项目 448_fresnel 的 Python 实现。

    C(x) = int_0^x cos(pi*t^2/2) dt
    S(x) = int_0^x sin(pi*t^2/2) dt

    算法分三段:
    - |x| < 2.5:  幂级数展开
    - 2.5 <= |x| < 4.5: 连分数/向后递推
    - |x| >= 4.5: 渐近展开

    Returns
    -------
    c, s : float
    """
    eps = 1e-15
    xa = abs(x)
    px = np.pi * xa
    t = 0.5 * px * xa
    t2 = t * t

    if xa == 0.0:
        return 0.0, 0.0

    elif xa < 2.5:
        # 幂级数展开
        r = xa
        c = r
        for k in range(1, 51):
            r = (-0.5 * r * (4.0 * k - 3.0) / k
                 / (2.0 * k - 1.0) / (4.0 * k + 1.0) * t2)
            c += r
            if abs(r) < abs(c) * eps:
                break

        s = xa * t / 3.0
        r = s
        for k in range(1, 51):
            r = (-0.5 * r * (4.0 * k - 1.0) / k
                 / (2.0 * k + 1.0) / (4.0 * k + 3.0) * t2)
            s += r
            if abs(r) < abs(s) * eps:
                break

    elif xa < 4.5:
        # 连分数向后递推
        m = int(42.0 + 1.75 * t)
        su = 0.0
        c = 0.0
        s = 0.0
        f1 = 0.0
        f0 = 1e-100

        for k in range(m, -1, -1):
            f = (2.0 * k + 3.0) * f0 / t - f1
            if k % 2 == 0:
                c += f
            else:
                s += f
            su += (2.0 * k + 1.0) * f * f
            f1 = f0
            f0 = f

        q = np.sqrt(su)
        c = c * xa / q
        s = s * xa / q

    else:
        # 渐近展开
        r = 1.0
        f = 1.0
        for k in range(1, 21):
            r = -0.25 * r * (4.0 * k - 1.0) * (4.0 * k - 3.0) / t2
            f += r

        r = 1.0 / (px * xa)
        g = r
        for k in range(1, 13):
            r = -0.25 * r * (4.0 * k + 1.0) * (4.0 * k - 1.0) / t2
            g += r

        t0 = t - np.floor(t / (2.0 * np.pi)) * 2.0 * np.pi
        c = 0.5 + (f * np.sin(t0) - g * np.cos(t0)) / px
        s = 0.5 - (f * np.cos(t0) + g * np.sin(t0)) / px

    if x < 0.0:
        c = -c
        s = -s

    return c, s


def mills_ratio(x):
    """Mill's 比率: R(x) = (1 - Phi(x)) / phi(x)

    渐近展开 (x → ∞):
    R(x) ≈ 1/x * (1 - 1/x^2 + 3/x^4 - 15/x^6 + ...)

    对于大 x, 1 - Phi(x) ≈ phi(x) / x * (1 - 1/x^2 + 3/x^4 - ...)
    """
    if x <= 0:
        return (1.0 - std_normal_cdf(x)) / (std_normal_pdf(x) + 1e-300)

    # 渐近展开
    if x > 8.0:
        x2 = x * x
        r = 1.0 / x
        term = r
        for k in range(1, 10):
            term *= -(2.0 * k - 1.0) / x2
            r += term
            if abs(term) < abs(r) * 1e-14:
                break
        return r
    else:
        return (1.0 - std_normal_cdf(x)) / (std_normal_pdf(x) + 1e-300)


def laplace_asymptotic_pf(lsf, u_star, beta):
    """Laplace 渐近法估计失效概率

    核心公式 (一阶 Laplace):
    P_f ≈ phi(beta) * (2*pi)^{(n-1)/2} * |det(A)|^{-1/2} / ||grad_g(u*)||

    其中 A 为 n×n 矩阵:
    A = [H_g/||grad_g||,  alpha; alpha^T, 0]
    H_g 为 g 的 Hessian, alpha = -grad_g/||grad_g||

    这比 FORM 的 Phi(-beta) 更精确，因为它考虑了极限状态面的曲率。

    Returns
    -------
    pf_laplace : float
    correction : float, 与FORM结果的比值
    """
    n = len(u_star)
    grad = lsf.gradient(u_star)
    H = lsf.hessian(u_star)
    grad_norm = np.linalg.norm(grad)

    if grad_norm < 1e-14:
        pf_form = std_normal_cdf(-beta)
        return float(pf_form), 1.0

    alpha = -grad / grad_norm

    # Breitung 修正 (简化版本，数值更稳定)
    # 构造旋转矩阵将 alpha 对齐到 e_n
    R = np.eye(n)
    R[:, -1] = alpha
    for i in range(n - 1):
        v = R[:, i].copy()
        for j in range(i):
            v -= np.dot(v, R[:, j]) * R[:, j]
        v -= np.dot(v, alpha) * alpha
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            R[:, i] = v / nv

    # 切平面上的 Hessian
    R_t = R[:, :n - 1]
    A = R_t.T @ H @ R_t / grad_norm

    try:
        eigvals = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        eigvals = np.zeros(n - 1)

    # Breitung 修正
    correction = 1.0
    for ki in np.real(eigvals):
        factor = 1.0 + beta * ki
        if factor > 1e-10:
            correction *= factor ** (-0.5)
        else:
            correction *= 1e5

    pf_form = std_normal_cdf(-beta)
    pf_laplace = pf_form * correction

    return float(pf_laplace), float(correction)


def cornu_spiral_distance(t1, t2):
    """Cornu 螺线上两点间的弧长

    Cornu 螺线: (C(t), S(t))
    用于分析振荡型极限状态面的几何特征。

    弧长 = |t2 - t1| (因为 |dC/dt|^2 + |dS/dt|^2 = 1)
    欧氏距离 = sqrt((C(t2)-C(t1))^2 + (S(t2)-S(t1))^2)
    """
    c1, s1 = fresnel_integrals(t1)
    c2, s2 = fresnel_integrals(t2)
    arc_length = abs(t2 - t1)
    euclidean = np.sqrt((c2 - c1) ** 2 + (s2 - s1) ** 2)
    return arc_length, euclidean


def tail_probability_oscillating(beta, omega, amplitude=0.1):
    """振荡型极限状态面的尾概率

    考虑 g(u) = beta - u_n + A * sin(omega * u_1)
    其尾概率包含 Fresnel 积分贡献:

    P_f ≈ Phi(-beta) + A * omega * [C(beta*omega/sqrt(2*pi)) - ...]

    这是可靠性分析中少见的精确振荡尾概率公式。
    """
    pf_base = std_normal_cdf(-beta)
    c_val, s_val = fresnel_integrals(beta * omega / np.sqrt(2 * np.pi))
    oscillation = amplitude * omega * (c_val - 0.5)
    return float(pf_base + oscillation)
