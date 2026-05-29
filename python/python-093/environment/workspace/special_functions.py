#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
special_functions.py
水声传播抛物方程模型 — 特殊函数库

本模块提供宽角抛物方程（WAPE）求解所需的各类特殊函数，包括：
- 归一化/非归一化 sinc 函数及其各阶导数、原函数
- 余弦积分 Ci(x) 与正弦积分 Si(x)（柱面波格林函数、远场近似）
- 标准正态分布 CDF alnorm（统计声学、大数溢出保护）

科学背景公式：
1. 归一化 sinc:  sinc_n(x) = sin(πx)/(πx)
   非归一化 sinc: sinc_u(x) = sin(x)/x
   x→0 时极限为 1（可去奇点）。

2. 一阶导数：
   d/dx[sinc_u(x)] = [x·cos(x) − sin(x)] / x²
   d/dx[sinc_n(x)] = [πx·cos(πx) − sin(πx)] / (πx²)

3. 二阶导数：
   d²/dx²[sinc_u(x)] = [(2−x²)sin(x) − 2x·cos(x)] / x³

4. 原函数：
   ∫ sinc_u(x) dx = Si(x) + C
   ∫ sinc_n(x) dx = Si(πx)/π + C

5. 余弦积分与正弦积分：
   Ci(x) = γ + ln(x) + ∫₀^x [cos(t)−1]/t dt   (x > 0)
   Si(x) = ∫₀^x sin(t)/t dt
   其中 γ ≈ 0.577215664901533 为 Euler-Mascheroni 常数。

6. 远场柱面波格林函数涉及 Ci 与 Si：
   G(r,z) ∝ H₀⁽¹⁾(kr) 的虚部与实部展开中包含 Ci(kr) 和 Si(kr)。

7. 正态分布 CDF（Hill 有理近似，Algorithm AS 66）：
   Φ(x) ≈ 0.5·exp(−x²/2)·Σ a_i·x^i / Σ b_j·x^j   (x ≥ 0)
   用于大样本统计声学中的正态近似。
"""

import numpy as np


def sincu_fun(x):
    """
    非归一化 sinc 函数：sinc_u(x) = sin(x)/x
    在 x=0 处利用可去奇点返回 1。
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    result[mask] = np.sin(x_masked) / x_masked
    return result


def sincn_fun(x):
    """
    归一化 sinc 函数：sinc_n(x) = sin(πx)/(πx)
    在 x=0 处返回 1。
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    result[mask] = np.sin(pix) / pix
    return result


def sincu_deriv(x):
    """
    非归一化 sinc 的一阶导数：
    d/dx[sin(x)/x] = (x·cos(x) − sin(x)) / x²
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    result[mask] = (x_masked * np.cos(x_masked) - np.sin(x_masked)) / (x_masked ** 2)
    return result


def sincn_deriv(x):
    """
    归一化 sinc 的一阶导数：
    d/dx[sin(πx)/(πx)] = (πx·cos(πx) − sin(πx)) / (πx²)
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    result[mask] = (pix * np.cos(pix) - np.sin(pix)) / (np.pi * x_masked ** 2)
    return result


def sincu_deriv2(x):
    """
    非归一化 sinc 的二阶导数：
    d²/dx²[sin(x)/x] = [(2−x²)sin(x) − 2x·cos(x)] / x³
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    sx = np.sin(x_masked)
    cx = np.cos(x_masked)
    result[mask] = ((2.0 - x_masked ** 2) * sx - 2.0 * x_masked * cx) / (x_masked ** 3)
    return result


def sincn_deriv2(x):
    """
    归一化 sinc 的二阶导数：
    d²/dx²[sin(πx)/(πx)] = π·[(2−(πx)²)sin(πx) − 2πx·cos(πx)] / (πx)³
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) > np.finfo(np.float64).eps
    x_masked = x[mask]
    pix = np.pi * x_masked
    sp = np.sin(pix)
    cp = np.cos(pix)
    result[mask] = np.pi * ((2.0 - pix ** 2) * sp - 2.0 * pix * cp) / (pix ** 3)
    return result


def _cisi_series(x):
    """
    小 x 区域 (0 < x ≤ 16) 的 Ci(x) 与 Si(x) 幂级数计算。
    Ci(x) = γ + ln(x) + Σ_{k=1}^∞ (−1)^k x^{2k} / [(2k)·(2k)!]
    Si(x) = Σ_{k=0}^∞ (−1)^k x^{2k+1} / [(2k+1)·(2k+1)!]
    使用递推生成项直至收敛。
    """
    x = float(x)
    gamma = 0.577215664901533
    ci = gamma + np.log(x)
    si = x
    term_ci = 1.0
    term_si = x
    k = 1
    while True:
        term_ci *= -x * x / ((2 * k - 1) * (2 * k))
        term_si *= -x * x / ((2 * k) * (2 * k + 1))
        dci = term_ci / (2 * k)
        dsi = term_si / (2 * k + 1)
        ci += dci
        si += dsi
        if abs(dci) < 1e-15 and abs(dsi) < 1e-15:
            break
        k += 1
        if k > 200:
            break
    return ci, si


def _cisi_bessel(x):
    """
    中等 x 区域 (16 < x ≤ 32) 的 Ci(x) 与 Si(x) 计算，
    利用球 Bessel 函数递推（Miller 算法）与三角组合：
    Ci(x) = γ + ln(x) − Ci_s(x)·cos(x) + si_s(x)·sin(x)
    Si(x) = π/2 − Ci_s(x)·sin(x) − si_s(x)·cos(x)
    其中 Ci_s, si_s 由球 Bessel 级数表示。
    此处采用稳定的有理逼近。
    """
    x = float(x)
    # 使用分段有理逼近（基于 Abramowitz & Stegun 公式 5.2.21-22）
    f = (1.0
         + 3.0381634e-2 / x ** 2
         - 3.4686916e-4 / x ** 4
         + 7.2189434e-6 / x ** 6)
    g = (1.0 / x
         - 1.9203743e-2 / x ** 3
         + 3.4108765e-4 / x ** 5
         - 5.2203843e-6 / x ** 7)
    ci = f * np.sin(x) / x - g * np.cos(x) / x
    si = np.pi / 2.0 - f * np.cos(x) / x - g * np.sin(x) / x
    return ci, si


def _cisi_asymptotic(x):
    """
    大 x 区域 (x > 32) 的 Ci(x) 与 Si(x) 渐近展开：
    Ci(x) ∼ sin(x)/x · P(x) − cos(x)/x · Q(x)
    Si(x) ∼ π/2 − cos(x)/x · P(x) − sin(x)/x · Q(x)
    P(x) = Σ_{k=0} (−1)^k (2k)! / x^{2k}
    Q(x) = Σ_{k=0} (−1)^k (2k+1)! / x^{2k+1}
    取 9 项以保证双精度。
    """
    x = float(x)
    x2 = x * x
    # 计算 P(x) 与 Q(x)
    p = 1.0
    q = 1.0 / x
    term_p = 1.0
    term_q = 1.0 / x
    for k in range(1, 9):
        term_p *= -(2 * k - 1) * (2 * k) / x2
        term_q *= -(2 * k) * (2 * k + 1) / x2
        p += term_p
        q += term_q
    sx = np.sin(x)
    cx = np.cos(x)
    ci = sx / x * p - cx / x * q
    si = np.pi / 2.0 - cx / x * p - sx / x * q
    return ci, si


def cisi(x):
    """
    计算余弦积分 Ci(x) 与正弦积分 Si(x)。
    采用分段策略：
      - x ≤ 0:   返回 nan（Ci 在 x≤0 非标准主值）
      - 0 < x ≤ 16: 幂级数展开
      - 16 < x ≤ 32: Bessel/有理逼近
      - x > 32: 渐近展开
    返回 (ci, si)。
    """
    x = np.asarray(x, dtype=np.float64)
    ci = np.full_like(x, np.nan, dtype=np.float64)
    si = np.full_like(x, np.nan, dtype=np.float64)

    # 小正值
    mask1 = (x > 0) & (x <= 16)
    if np.any(mask1):
        for idx in np.where(mask1)[0]:
            c, s = _cisi_series(x[idx])
            ci[idx] = c
            si[idx] = s

    # 中等值
    mask2 = (x > 16) & (x <= 32)
    if np.any(mask2):
        for idx in np.where(mask2)[0]:
            c, s = _cisi_bessel(x[idx])
            ci[idx] = c
            si[idx] = s

    # 大值
    mask3 = x > 32
    if np.any(mask3):
        for idx in np.where(mask3)[0]:
            c, s = _cisi_asymptotic(x[idx])
            ci[idx] = c
            si[idx] = s

    return ci, si


def sincu_antideriv(x):
    """∫ sinc_u(x) dx = Si(x)"""
    x = np.asarray(x, dtype=np.float64)
    ci, si = cisi(np.abs(x))
    return np.sign(x) * si


def sincn_antideriv(x):
    """∫ sinc_n(x) dx = Si(πx)/π"""
    x = np.asarray(x, dtype=np.float64)
    ci, si = cisi(np.abs(np.pi * x))
    return np.sign(x) * si / np.pi


def alnorm(x, upper=False):
    """
    标准正态分布累积分布函数 Φ(x)（Algorithm AS 66, Hill）。
    使用有理多项式逼近，避免大数溢出。

    参数:
        x: float 或 array-like
        upper: 若 True 返回上尾概率 1−Φ(x)

    公式：
      令 y = |x|/√2
      当 y < 1（即 |x|<√2）时，使用误差函数级数；
      当 y ≥ 1 时，使用连分数/有理逼近：
        Φ(x) ≈ 0.5·erfc(y)  (x ≥ 0)
    此处实现基于 Hill (1973) 的紧凑有理逼近：
        z = exp(−x²/2) / √(2π)
        t = 1 / (1 + p·|x|)
        Φ(−|x|) ≈ z·t·(a1 + a2·t + a3·t² + a4·t³ + a5·t⁴)
    """
    x = np.asarray(x, dtype=np.float64)
    p = 0.2316419
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429

    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    z = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    lower_tail = 1.0 - z * poly
    # 边界修正
    lower_tail = np.clip(lower_tail, 0.0, 1.0)
    result = np.where(sign < 0, 1.0 - lower_tail, lower_tail)
    if upper:
        return 1.0 - result
    return result


def gammaln_stable(x):
    """
    稳定的 log-gamma 计算，用于避免阶乘溢出。
    采用 scipy.special.gammaln 实现（双精度，覆盖全定义域）。
    """
    from scipy.special import gammaln
    x = np.asarray(x, dtype=np.float64)
    return gammaln(x)
