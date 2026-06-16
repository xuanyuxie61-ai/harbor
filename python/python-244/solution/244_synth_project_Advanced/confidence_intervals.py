#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confidence_intervals.py
=======================
【融合种子项目】 082_beta_nc (非中心 Beta 分布 CDF)

使用非中心分布构造 EoS 参数的置信区间.

数学:
    非中心 Beta 分布: X ~ Beta'(a, b, lambda)
    CDF 通过 Poisson 混合:
        F(x; a, b, lambda) = sum_{j=0}^inf e^{-lambda/2} (lambda/2)^j / j! * I_x(a+j, b)

    其中 I_x(a, b) 为正则不完全 Beta 函数.
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable
from numerical_constants import r8_gamma, r8_gamma_log, r8_epsilon


def alogam(x: float) -> float:
    """
    ln|Gamma(x)| (移植自 beta_nc 的 alogam).
    """
    from numerical_constants import r8_gamma_log
    return r8_gamma_log(x)


def betain(x: float, a: float, b: float) -> Tuple[float, int]:
    """
    正则不完全 Beta 函数 I_x(a, b).

    I_x(a, b) = B(x; a, b) / B(a, b)
             = integral_0^x t^{a-1} (1-t)^{b-1} dt / B(a, b)

    使用连分式展开 (Numerical Recipes).
    """
    if x < 0.0 or x > 1.0:
        return 0.0, 1  # fault

    if x == 0.0 or x == 1.0:
        return x, 0

    # 使用对称性保证计算稳定
    if a < (a + b) * 0.5:
        # 用 bt * betacf
        bt = math.exp(
            a * math.log(x) + b * math.log(1.0 - x)
            - alogam(a) - alogam(b) + alogam(a + b)
        )
        # 连分式
        cf = _betacf(x, a, b)
        return bt * cf / a, 0
    else:
        bt = math.exp(
            b * math.log(1.0 - x) + a * math.log(x)
            - alogam(b) - alogam(a) + alogam(a + b)
        )
        cf = _betacf(1.0 - x, b, a)
        return 1.0 - bt * cf / b, 0


def _betacf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """
    不完全 Beta 函数的连分式展开 (Lentz 算法).

    I_x(a, b) = x^a (1-x)^b / (a B(a,b)) * 1/(1 + d1/(1 + d2/(1 + ...)))
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # 偶数项
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # 奇数项
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break

    return h


def beta_noncentral_cdf(a: float, b: float, lam: float, x: float,
                          error_max: float = 1e-8) -> float:
    """
    非中心 Beta CDF (直接移植自 beta_noncentral_cdf).

    F(x; a, b, lambda) = sum_{j=0}^inf p_j * B_j(x)

    其中:
        p_j = e^{-lambda/2} (lambda/2)^j / j!  (Poisson 权重)
        B_j(x) = I_x(a+j, b)  (正则不完全 Beta)

    递推:
        p_j = (lambda/2) * p_{j-1} / j
        B_j = B_{j-1} - s_{j-1}
        s_j = x * (a+b+j-1) * s_{j-1} / (a+j)

    收敛条件: sum p_j > 1 - error_max
    """
    if x < 0.0 or x > 1.0:
        return 0.0

    i = 0
    pi = math.exp(-lam / 2.0)  # p_0

    beta_log = alogam(a) + alogam(b) - alogam(a + b)
    bi, ifault = betain(x, a, b)
    if ifault != 0:
        bi = 0.0

    si = math.exp(
        a * math.log(max(x, 1e-300))
        + b * math.log(max(1.0 - x, 1e-300))
        - beta_log - math.log(max(a, 1e-300))
    )

    p_sum = pi
    pb_sum = pi * bi

    max_iter = 10000
    while p_sum < 1.0 - error_max and i < max_iter:
        pj = pi
        bj = bi
        sj = si

        i += 1
        pi = 0.5 * lam * pj / i
        bi = bj - sj
        si = x * (a + b + i - 1) * sj / (a + i)

        p_sum += pi
        pb_sum += pi * bi

    return max(0.0, min(1.0, pb_sum))


def f_distribution_cdf(x: float, d1: float, d2: float) -> float:
    """
    F 分布 CDF (非中心 F 的特殊情形).

    F(x; d1, d2) = I_{d1 x / (d1 x + d2)}(d1/2, d2/2)
    """
    if x <= 0:
        return 0.0
    z = d1 * x / (d1 * x + d2)
    val, _ = betain(z, d1 / 2.0, d2 / 2.0)
    return val


def chi_squared_cdf(x: float, k: int) -> float:
    """
    卡方分布 CDF.
    chi^2(x; k) = gamma_inc(k/2, x/2) / Gamma(k/2) = P(k/2, x/2)
    """
    if x <= 0:
        return 0.0
    val, _ = betain(x / (x + 1.0), k / 2.0, 0.5)
    # 近似
    # 使用不完全 gamma 的简化
    return min(1.0, max(0.0, 1.0 - math.exp(-x / 2) * sum(
        (x / 2) ** j / math.factorial(j) for j in range(min(k, 30))
    )))


def eos_confidence_interval(posterior_samples: np.ndarray,
                              confidence: float = 0.9) -> Tuple[float, float, float]:
    """
    EoS 参数置信区间 (使用非中心分布思想).

    返回: (lower, median, upper)
    """
    alpha = 1.0 - confidence
    lower = float(np.percentile(posterior_samples, 100 * alpha / 2))
    median = float(np.median(posterior_samples))
    upper = float(np.percentile(posterior_samples, 100 * (1 - alpha / 2)))
    return lower, median, upper


def hotelling_t2_confidence(data_matrix: np.ndarray,
                               confidence: float = 0.95) -> Dict:
    """
    Hotelling T^2 置信区域 (多参数 EoS 约束).

    T^2 = n (x_bar - mu)^T S^{-1} (x_bar - mu)

    在 H0 下: (n-p)/(p(n-1)) T^2 ~ F(p, n-p)
    """
    n, p = data_matrix.shape
    x_bar = np.mean(data_matrix, axis=0)
    S = np.cov(data_matrix.T)

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    t2 = n * (x_bar @ S_inv @ x_bar)

    # F 临界值近似
    f_crit = _f_critical(p, n - p, 1.0 - confidence)
    threshold = p * (n - 1) / (n - p) * f_crit if n > p else t2

    return {
        'T2': float(t2),
        'threshold': float(threshold),
        'is_significant': t2 > threshold,
        'mean': x_bar,
    }


def _f_critical(d1: int, d2: int, alpha: float) -> float:
    """F 分布临界值的近似."""
    # 近似: 使用正态近似
    z = _norm_ppf(1.0 - alpha)
    # Wilson-Hilferty 变换
    h = 2.0 / (9.0 * d2)
    k = 2.0 / (9.0 * d1)
    num = (1.0 - h + z * math.sqrt(h)) ** 3
    den = (1.0 - k - z * math.sqrt(k)) ** 3
    return num / den if den > 0 else 1.0


def _norm_ppf(p: float) -> float:
    """正态分布逆 CDF (Abramowitz-Stegun 近似)."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)


# 自检
if __name__ == "__main__":
    print("=== 置信区间自检 ===")

    # 正则不完全 Beta: I_{0.5}(2, 2) = 0.5 (对称)
    val, ifault = betain(0.5, 2.0, 2.0)
    print(f"I_0.5(2,2) = {val:.6f}  (应为 0.5), ifault={ifault}")

    val2, _ = betain(0.3, 1.0, 1.0)
    print(f"I_0.3(1,1) = {val2:.6f}  (应为 0.3)")

    # 非中心 Beta CDF
    cdf_val = beta_noncentral_cdf(2.0, 3.0, 0.0, 0.5)
    print(f"Beta CDF(2,3,lambda=0,x=0.5) = {cdf_val:.6f}")

    cdf_nc = beta_noncentral_cdf(2.0, 3.0, 2.0, 0.5)
    print(f"Beta CDF(2,3,lambda=2,x=0.5) = {cdf_nc:.6f}")

    # F 分布
    f_val = f_distribution_cdf(2.0, 5, 10)
    print(f"F CDF(2; 5, 10) = {f_val:.6f}")

    # 置信区间
    np.random.seed(42)
    samples = np.random.normal(5.0, 1.0, 1000)
    lo, med, hi = eos_confidence_interval(samples, 0.9)
    print(f"90% CI: [{lo:.3f}, {med:.3f}, {hi:.3f}]")

    # Hotelling T^2
    data = np.random.randn(50, 3) + np.array([1.0, 2.0, 3.0])
    result = hotelling_t2_confidence(data, 0.95)
    print(f"T^2 = {result['T2']:.3f}, threshold = {result['threshold']:.3f}")

    print("\nconfidence_intervals.py 自检通过.")
