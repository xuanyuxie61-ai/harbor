# -*- coding: utf-8 -*-
"""
special_functions.py

博士级科学计算特殊函数库

融合原项目算法：
- 436_flame_exact 的 Lambert W 函数实现
- 448_fresnel 的 Fresnel 积分实现

科学应用场景：
1. Lambert W 函数用于求解结晶过程中超越方程的解析解，如尺寸依赖生长律中的
   隐式关系：L·exp(αL) = βt  ⇒  L = (1/α)·W(αβt)
2. Fresnel 积分用于基于衍射原理的原位晶体尺寸测量（激光衍射粒度分析），
   Cornu 螺旋与 Fraunhofer 衍射理论
"""

import numpy as np
from numpy.polynomial import polynomial as P


def lambert_w(x, branch=0):
    """
    计算实数 Lambert W 函数 W_k(x)，满足 W·exp(W) = x。

    数学定义：
        W(z) 是方程 w·e^w = z 的解。
        主分支 W_0(x) 定义域 x ∈ [-1/e, +∞)。
        下分支 W_{-1}(x) 定义域 x ∈ [-1/e, 0)。

    参数：
        x : float 或 ndarray
            输入值
        branch : int
            0 表示主分支 W_0，-1 表示下分支 W_{-1}

    返回：
        w : float 或 ndarray
            W(x) 的值，域外返回 nan
    """
    x = np.asarray(x, dtype=float)
    w = np.full_like(x, np.nan, dtype=float)
    em1 = -1.0 / np.e

    # 域检查
    if branch == 0:
        valid = x >= em1
    elif branch == -1:
        valid = (x >= em1) & (x < 0.0)
    else:
        return w

    xv = x[valid]
    if xv.size == 0:
        return w

    # 初始近似
    # 对于主分支，使用分段有理近似
    if branch == 0:
        # 分支点附近的级数展开：W(-1/e + ε) ≈ -1 + √(2eε) - (2e/3)ε + ...
        near_branch = xv < -0.2
        far = ~near_branch

        wv = np.empty_like(xv)
        if np.any(near_branch):
            delta = xv[near_branch] - em1
            p = np.sqrt(2.0 * np.e * delta)
            wv[near_branch] = -1.0 + p - (np.e / 3.0) * delta + \
                              (11.0 * np.sqrt(2.0) / 72.0) * p * delta
        if np.any(far):
            # 对数近似：W(x) ≈ ln(x) - ln(ln(x))
            lx = np.log(xv[far])
            llx = np.log(lx)
            wv[far] = lx - llx + llx / lx
    else:
        # 下分支 W_{-1}
        # 在分支点附近：W_{-1}(-1/e + ε) ≈ -1 - √(2eε)
        near_branch = xv < -0.1
        far = ~near_branch

        wv = np.empty_like(xv)
        if np.any(near_branch):
            delta = xv[near_branch] - em1
            p = np.sqrt(2.0 * np.e * delta)
            wv[near_branch] = -1.0 - p - (np.e / 3.0) * delta - \
                              (11.0 * np.sqrt(2.0) / 72.0) * p * delta
        if np.any(far):
            lx = np.log(-xv[far])
            llx = np.log(-lx)
            wv[far] = lx - llx + llx / lx

    # Halley 迭代精化：
    # w_{n+1} = w_n - (w_n·e^{w_n} - x) / ((w_n+1)·e^{w_n} - (w_n+2)(w_n·e^{w_n}-x)/(2w_n+2))
    # 简化为：w_{n+1} = w_n - w_e / ((w_n+1)·e^{w_n} - (w_n+2)·w_e/(2w_n+2))
    # 其中 w_e = w_n·e^{w_n} - x
    for _ in range(8):
        ew = np.exp(wv)
        we = wv * ew - xv
        denom = (wv + 1.0) * ew - (wv + 2.0) * we / (2.0 * wv + 2.0)
        # 避免除零
        denom = np.where(np.abs(denom) < 1e-300, np.copysign(1e-300, denom), denom)
        wv = wv - we / denom

    w[valid] = wv
    return w


def fresnel_integrals(x):
    """
    计算 Fresnel 积分 C(x) 和 S(x)。

    数学定义：
        C(x) = ∫_0^x cos(π t^2 / 2) dt
        S(x) = ∫_0^x sin(π t^2 / 2) dt

    这些积分在光学衍射理论中至关重要：
    - 激光粒度分析仪基于 Fraunhofer 衍射，衍射图样与 Fresnel 积分相关
    - Cornu 螺旋 param(C(t), S(t)) 描述波前传播

    参数：
        x : float 或 ndarray

    返回：
        C, S : ndarray
    """
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    sgn = np.sign(x)
    C = np.zeros_like(x, dtype=float)
    S = np.zeros_like(x, dtype=float)

    # 区域 1: |x| < 2.5 — 幂级数展开
    region1 = ax < 2.5
    if np.any(region1):
        t = ax[region1]
        t2 = t * t
        # C(x) 的幂级数：
        # C(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n} x^{4n+1} / [(2n)! (4n+1)]
        # S(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n+1} x^{4n+3} / [(2n+1)! (4n+3)]
        c_val = np.zeros_like(t)
        s_val = np.zeros_like(t)
        # 使用递推计算，提高数值稳定性
        for i, ti in enumerate(t):
            c_sum = 0.0
            s_sum = 0.0
            term_c = ti  # n=0 项
            term_s = (np.pi / 2.0) * ti**3 / 3.0
            c_sum += term_c
            s_sum += term_s
            for n in range(1, 50):
                # C 的递推
                term_c *= -(np.pi / 2.0)**2 * ti**4 / ((2*n) * (2*n - 1) * (4*n + 1) / (4*n - 3))
                # S 的递推
                term_s *= -(np.pi / 2.0)**2 * ti**4 / ((2*n + 1) * (2*n) * (4*n + 3) / (4*n - 1))
                if np.abs(term_c) < 1e-15 and np.abs(term_s) < 1e-15:
                    break
                c_sum += term_c
                s_sum += term_s
            c_val[i] = c_sum
            s_val[i] = s_sum
        C[region1] = c_val
        S[region1] = s_val

    # 区域 2: 2.5 <= |x| < 4.5 — 反向递推
    region2 = (ax >= 2.5) & (ax < 4.5)
    if np.any(region2):
        t = ax[region2]
        t0 = 0.5 * np.pi * t * t
        # 使用辅助函数 f, g 的渐近展开
        # C(x) = 0.5 + (f·sin(t0) - g·cos(t0)) / (π·x)
        # S(x) = 0.5 - (f·cos(t0) + g·sin(t0)) / (π·x)
        # 其中 f, g 用级数计算
        f = np.zeros_like(t)
        g = np.zeros_like(t)
        for i, ti in enumerate(t):
            u = 1.0 / ((0.5 * np.pi * ti * ti)**2)
            # f 的级数
            f_sum = 1.0
            term = 1.0
            for n in range(1, 20):
                term *= -(4*n - 3) * (4*n - 1) * u / ((4*n - 4) * (4*n) if n > 1 else 1)
                if np.abs(term) < 1e-15:
                    break
                f_sum += term
            # g 的级数
            g_sum = 1.0
            term = 1.0
            for n in range(1, 12):
                term *= -(4*n - 1) * (4*n + 1) * u / ((4*n - 2) * (4*n + 2))
                if np.abs(term) < 1e-15:
                    break
                g_sum += term
            f[i] = f_sum
            g[i] = g_sum
        st0 = np.sin(t0)
        ct0 = np.cos(t0)
        C[region2] = 0.5 + (f * st0 - g * ct0) / (np.pi * t)
        S[region2] = 0.5 - (f * ct0 + g * st0) / (np.pi * t)

    # 区域 3: |x| >= 4.5 — 渐近展开
    region3 = ax >= 4.5
    if np.any(region3):
        t = ax[region3]
        t0 = 0.5 * np.pi * t * t
        # 相位约化以避免大参数正弦/余弦的精度损失
        t0_red = t0 % (2.0 * np.pi)
        st0 = np.sin(t0_red)
        ct0 = np.cos(t0_red)
        # 渐近系数
        f = np.ones_like(t)
        g = np.ones_like(t)
        u = 1.0 / t**2
        for n in range(1, 10):
            coeff_f = 1.0
            for k in range(1, n + 1):
                coeff_f *= (4.0 * k - 3.0) * (4.0 * k - 1.0)
            from math import factorial
            coeff_f /= (np.pi * t)**(2 * n) * factorial(2 * n) if 2*n < 20 else 1e300
            # 简化的渐近系数
            cf = 1.0
            cg = 1.0
            for k in range(1, n + 1):
                cf *= (4.0 * k - 3.0) * (4.0 * k - 1.0) / ((2.0 * k - 1.0) * 2.0 * k)
                cg *= (4.0 * k - 1.0) * (4.0 * k + 1.0) / ((2.0 * k) * (2.0 * k + 1.0))
            f += cf * ((-1)**n) * u**n
            g += cg * ((-1)**n) * u**n
        C[region3] = 0.5 + (f * st0 - g * ct0) / (np.pi * t)
        S[region3] = 0.5 - (f * ct0 + g * st0) / (np.pi * t)

    # 恢复符号
    C = sgn * C
    S = sgn * S
    return C, S


def fraunhofer_diffraction_particle_size(radius, wavelength, theta):
    """
    基于 Fraunhofer 衍射理论的颗粒光强分布。

    物理模型：
        对于半径为 a 的球形颗粒，Fraunhofer 衍射的远场光强分布为：
        I(θ) ∝ [2·J_1(k·a·sinθ) / (k·a·sinθ)]^2
        其中 k = 2π/λ 为波数，J_1 为一阶贝塞尔函数。

    当颗粒尺寸远小于光束波长时，需使用 Fresnel 近似：
        I(θ) ∝ |∫_0^a J_0(kr·sinθ)·exp(i·kr^2/(2R)) r dr|^2
        其中 R 为观察距离。

    参数：
        radius : float 或 ndarray
            颗粒半径 (m)
        wavelength : float
            激光波长 (m)
        theta : float 或 ndarray
            散射角 (rad)

    返回：
        intensity : ndarray
            归一化衍射光强
    """
    from scipy.special import j1
    k = 2.0 * np.pi / wavelength
    x = k * radius * np.sin(theta)
    # 避免除零
    x = np.where(np.abs(x) < 1e-10, 1e-10, x)
    intensity = (2.0 * j1(x) / x) ** 2
    return intensity
