# -*- coding: utf-8 -*-
"""
special_functions.py
====================
特殊函数与解析工具。

融合种子项目：
- 1270_toms443 : Lambert W 函数（Halley 迭代）
- 1161_steinerberger : Steinerberger 特殊函数与调和数
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# Lambert W 函数（TOMS 443）
# ---------------------------------------------------------------------------

def lambert_w_high_accuracy(x):
    """
    高精度 Lambert W 函数估计（wew_a 的 Python 实现）。
    求解满足 W * exp(W) = x 的 W(x)。

    算法：
      1) 初始猜测（有理函数近似）
      2) 两次 Halley 型迭代精化

    参数
    ----
    x : float or array_like
        自变量，要求 x >= -1/e。

    返回
    ----
    w : float or ndarray
        W(x) 的估计值。
    en : float or ndarray
        最后一次相对修正量。
    """
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)

    # 边界处理
    em1 = -1.0 / math.e
    w = np.zeros_like(x)
    en = np.zeros_like(x)

    for idx in range(x.size):
        xi = x[idx]
        if xi < em1:
            w[idx] = np.nan
            en[idx] = np.nan
            continue
        if abs(xi) < 1e-30:
            w[idx] = 0.0
            en[idx] = 0.0
            continue

        if xi > 0:
            f = math.log(xi)
        else:
            f = -1e300

        # 初始猜测（有理函数）
        c1 = 4.0 / 3.0
        c2 = 7.0 / 3.0
        c3 = 5.0 / 6.0
        c4 = 2.0 / 3.0

        if xi <= 6.46:
            wn = xi * (1.0 + c1 * xi) / (1.0 + xi * (c2 + c3 * xi))
            zn = f - wn - math.log(wn) if wn > 0 else -1e300
        else:
            wn = f
            zn = -math.log(wn) if wn > 0 else 1e300

        # 迭代 1
        temp = 1.0 + wn
        y = 2.0 * temp * (temp + c4 * zn) - zn
        wn = wn * (1.0 + zn * y / (temp * (y - zn)))

        # 迭代 2
        zn = f - wn - math.log(wn) if wn > 0 else -1e300
        temp = 1.0 + wn
        temp2 = temp + c4 * zn
        eni = zn * temp2 / (temp * temp2 - 0.5 * zn)
        wn = wn * (1.0 + eni)

        w[idx] = wn
        en[idx] = eni

    if scalar_input:
        return float(w[0]), float(en[0])
    return w.reshape(x.shape), en.reshape(x.shape)


def lambert_w_fast(x):
    """
    快速 Lambert W 函数估计（wew_b 的 Python 实现）。
    精度略低于 lambert_w_high_accuracy，但速度更快。
    """
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    em1 = -1.0 / math.e
    w = np.zeros_like(x)
    en = np.zeros_like(x)

    for idx in range(x.size):
        xi = x[idx]
        if xi < em1:
            w[idx] = np.nan
            en[idx] = np.nan
            continue
        if abs(xi) < 1e-30:
            w[idx] = 0.0
            en[idx] = 0.0
            continue

        if xi > 0:
            f = math.log(xi)
            c1 = 4.0 / 3.0
            c2 = 7.0 / 3.0
            c3 = 5.0 / 6.0
            c4 = 2.0 / 3.0

            if xi <= 0.7385:
                wn = xi * (1.0 + c1 * xi) / (1.0 + xi * (c2 + c3 * xi))
            else:
                wn = f - 24.0 * ((f + 2.0) * f - 3.0) / ((0.7 * f + 58.0) * f + 127.0)

            zn = f - wn - math.log(wn) if wn > 0 else -1e300
            temp = 1.0 + wn
            y = 2.0 * temp * (temp + c4 * zn) - zn
            den = temp * (y - zn)
            if abs(den) < 1e-30:
                eni = 0.0
            else:
                eni = zn * y / den
            wn = wn * (1.0 + eni)
        else:
            # 牛顿法处理负分支 [-1/e, 0)
            wn = -1.0
            for _ in range(10):
                ew = math.exp(wn)
                num = wn * ew - xi
                den = (wn + 1.0) * ew
                if abs(den) < 1e-30:
                    break
                dw = num / den
                wn = wn - dw
                if abs(dw) < 1e-14:
                    break
            eni = 0.0

        w[idx] = wn
        en[idx] = eni

    if scalar_input:
        return float(w[0]), float(en[0])
    return w.reshape(x.shape), en.reshape(x.shape)


def lambert_w_convergence_rate(kappa):
    """
    利用 Lambert W 函数估计 CG 方法的渐进收敛速率。
    对于条件数为 κ 的 SPD 矩阵，迭代 k 步后的误差上界：
        ||e_k||_A / ||e_0||_A <= 2 * ( (sqrt(κ)-1)/(sqrt(κ)+1) )^k
    等价地，达到精度 ε 所需迭代次数的近似估计：
        k ≈ 0.5 * sqrt(κ) * log(2/ε)
    本函数利用 Lambert W 给出基于谱间隙的精确下界估计。
    """
    if kappa <= 1.0:
        return 0.0, 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    # 利用 W 函数求 log(rho) 的精细行为：
    # 当 κ 极大时，rho ≈ 1 - 2/sqrt(κ)，则
    # log(rho) ≈ -2/sqrt(κ) - 2/κ ...
    # 通过 Lambert W(-2/sqrt(κ)) 修正渐近项
    arg = -2.0 / math.sqrt(kappa)
    if arg >= -1.0 / math.e:
        w_val, _ = lambert_w_fast(arg)
        refined = -2.0 / math.sqrt(kappa) + w_val / kappa
    else:
        refined = math.log(rho)
    return rho, refined


# ---------------------------------------------------------------------------
# Steinerberger 函数与调和数（1161_steinerberger）
# ---------------------------------------------------------------------------

def steinerberger_function(n, x):
    """
    Steinerberger 特殊函数：
        f(n, x) = sum_{k=1}^{n} |sin(π k x)| / k

    用于构造具有大量局部极值的病态测试问题。
    """
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = x.reshape(-1)
    val = np.zeros_like(x)
    for k in range(1, n + 1):
        val += np.abs(np.sin(math.pi * k * x)) / k
    if scalar_input:
        return float(val[0])
    return val.reshape(x.shape)


def steinerberger_integral_01(n):
    """
    Steinerberger 函数在 [0,1] 上的解析积分：
        I(n) = ∫_0^1 f(n,x) dx = 2 H(n) / π
    其中 H(n) 为第 n 个调和数。
    """
    h = harmonic_number(n)
    return 2.0 * h / math.pi


def harmonic_number(n):
    """
    第 n 个调和数：
        H(n) = sum_{i=1}^{n} 1/i
    对 n <= 0 返回 0.0。
    """
    if n <= 0:
        return 0.0
    # 对小 n 直接求和，对大 n 用 Euler-Mascheroni 常数近似：
    # H(n) ≈ log(n) + γ + 1/(2n) - 1/(12n^2) + ...
    if n <= 10000:
        return float(np.sum(1.0 / np.arange(1, n + 1)))
    gamma = 0.5772156649015328606
    return math.log(n) + gamma + 1.0 / (2.0 * n) - 1.0 / (12.0 * n * n)


def steinerberger_rhs(n, x_grid):
    """
    基于 Steinerberger 函数构造右端项 b，用于离散化后的线性系统。
    b_i = f(n, x_i) * h^2，其中 h 为网格步长。
    """
    x_grid = np.asarray(x_grid, dtype=float)
    f_vals = steinerberger_function(n, x_grid)
    h = 1.0
    if x_grid.size > 1:
        h = float(x_grid[1] - x_grid[0])
    return f_vals * (h ** 2)


# ---------------------------------------------------------------------------
# 其他解析工具
# ---------------------------------------------------------------------------

def signed_power(x, p):
    """安全的有符号幂运算：sign(x) * |x|^p，处理 x=0。"""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sign(x) * np.power(np.abs(x), p)
