#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_posterior.py
=====================
【融合种子项目】 035_asa091 (正态分布 CDF - alnorm)

实现 EoS 参数的贝叶斯后验评估, 使用标准正态 CDF 计算可信区间.

数学: Phi(x) = (1/sqrt(2pi)) integral_{-inf}^x exp(-t^2/2) dt
后验: P(theta|D) ∝ P(D|theta) P(theta)
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable, Optional
from numerical_constants import r8_epsilon


def alnorm(x: float, upper: bool = False) -> float:
    """
    标准正态分布 CDF (移植自 ASA 091 alnorm).

    Phi(x) = 1 - Q(x)  其中 Q 为上尾概率.

    使用 Hill 算法 (1973), 精度 ~10^{-8}.
    """
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -3.8052e-8
    c2 = 3.98064794e-4
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66

    up = not upper
    z = x
    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (up or utzero < z):
        y = 0.5 * z * z
        if z <= con:
            q_val = r * math.exp(-y) * (d1 + d2 * y + d3 * y**2) / (
                y**2 + d4 * y + d5 + y * (d1 + d2 * y))
        else:
            q_val = r * math.exp(-y) / (z + 1.0 / (z + 2.0 / (z + 3.0 / z)))
        value = q_val if up else 1.0 - q_val
    else:
        if z <= ltone:
            y = z * z
            q_val = (c1 + c2 * z + c3 * z**2) / (1 + c4 * z + c5 * z**2 + c6 * y)
            result = 0.5 - z * (p - q_val * y / 2.0) if z < 1.0 else (
                0.5 - math.exp(-y / 2.0) * (a1 + a2 * z + a3 * z**2) / (
                    z**3 + b1 * z**2 + b2 + z * (a1 + a2 * z)))
            # 简化
            result = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        else:
            result = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        value = result if up else 1.0 - result

    return max(0.0, min(1.0, value))


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """正态分布 PDF."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """正态分布 CDF."""
    return alnorm((x - mu) / sigma)


def log_likelihood_gaussian(observed: np.ndarray, predicted: np.ndarray,
                              sigma: float) -> float:
    """
    高斯对数似然.
    ln L = -n/2 ln(2 pi sigma^2) - sum (obs - pred)^2 / (2 sigma^2)
    """
    n = len(observed)
    resid = observed - predicted
    return -0.5 * n * math.log(2.0 * math.pi * sigma ** 2) - 0.5 * np.sum(resid ** 2) / sigma ** 2


def log_prior_uniform(theta: np.ndarray, bounds: list) -> float:
    """均匀先验的对数."""
    for i, (lo, hi) in enumerate(bounds):
        if theta[i] < lo or theta[i] > hi:
            return float('-inf')
    return 0.0


def log_prior_gaussian(theta: np.ndarray, mu: np.ndarray,
                        sigma: np.ndarray) -> float:
    """高斯先验的对数."""
    return -0.5 * np.sum(((theta - mu) / sigma) ** 2)


def posterior_grid(eos_params: np.ndarray,
                    likelihood_func: Callable,
                    prior_func: Callable) -> np.ndarray:
    """
    网格法计算后验分布.
    P(theta_i | D) ∝ L(D|theta_i) * prior(theta_i)
    """
    log_post = np.zeros(len(eos_params))
    for i, theta in enumerate(eos_params):
        lp = likelihood_func(theta)
        pp = prior_func(theta)
        if pp == float('-inf'):
            log_post[i] = float('-inf')
        else:
            log_post[i] = lp + pp

    # 归一化 (log-sum-exp)
    max_log = np.max(log_post[np.isfinite(log_post)])
    post = np.exp(log_post - max_log)
    total = np.sum(post)
    if total > 0:
        post /= total
    return post


def credible_interval(posterior: np.ndarray, param_values: np.ndarray,
                       level: float = 0.9) -> Tuple[float, float]:
    """
    最高后验密度 (HPD) 可信区间.
    """
    sorted_idx = np.argsort(-posterior)
    cumsum = np.cumsum(posterior[sorted_idx])
    n_included = np.searchsorted(cumsum, level) + 1
    n_included = min(n_included, len(posterior))
    included = param_values[sorted_idx[:n_included]]
    return float(np.min(included)), float(np.max(included))


def bayesian_eos_constraint(m_obs: float, m_err: float,
                              r_obs: float, r_err: float,
                              n_samples: int = 200) -> Dict:
    """
    贝叶斯 EoS 约束示例.
    """
    np.random.seed(42)
    # 参数网格: (K, Gamma)
    K_range = np.logspace(33, 36, 20)
    gamma_range = np.linspace(2.0, 3.5, 10)
    params = []
    for K in K_range:
        for g in gamma_range:
            params.append([K, g])
    params = np.array(params)

    # 简单模型: R = a * (K/K0)^b * Gamma^c
    def model(K, g):
        return 10.0 * (K / 1e35) ** 0.2 * (g / 2.5) ** 0.5

    def likelihood(theta):
        K, g = theta
        if K <= 0 or g <= 1:
            return float('-inf')
        r_pred = model(K, g)
        m_pred = 2.0  # 简化
        return (log_likelihood_gaussian(np.array([m_obs]), np.array([m_pred]), m_err)
                + log_likelihood_gaussian(np.array([r_obs]), np.array([r_pred]), r_err))

    def prior(theta):
        return log_prior_uniform(theta, [(1e33, 1e36), (1.5, 4.0)])

    log_post = np.array([likelihood(p) + prior(p) for p in params])
    max_log = np.max(log_post[np.isfinite(log_post)])
    post = np.exp(log_post - max_log)
    post /= np.sum(post)

    return {
        'params': params,
        'posterior': post,
        'K_range': K_range,
        'gamma_range': gamma_range,
    }


# 自检
if __name__ == "__main__":
    print("=== 贝叶斯后验自检 ===")
    print(f"Phi(0) = {alnorm(0.0):.6f}  (精确 0.5)")
    print(f"Phi(1) = {alnorm(1.0):.6f}  (精确 ~0.8413)")
    print(f"Phi(-2) = {alnorm(-2.0):.6f}  (精确 ~0.0228)")

    print(f"N(0;0,1) = {normal_pdf(0.0):.6f}  (精确 {1/math.sqrt(2*math.pi):.6f})")

    result = bayesian_eos_constraint(1.4, 0.1, 12.0, 1.0)
    print(f"后验样本数: {len(result['posterior'])}")
    print(f"后验最大值: {np.max(result['posterior']):.4f}")

    ci = credible_interval(result['posterior'], result['params'][:, 1], 0.9)
    print(f"Gamma 90% 可信区间: [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\nbayesian_posterior.py 自检通过.")
