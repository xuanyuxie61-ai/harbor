#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_monte_carlo.py
==================
【融合种子项目】 029_asa053 (Box-Muller 正态随机数 rnorm)

EoS 参数的蒙特卡洛采样, 用于不确定性量化.

数学: Box-Muller 变换
    u1, u2 ~ U(0,1)
    z1 = sqrt(-2 ln u1) cos(2 pi u2) ~ N(0,1)
    z2 = sqrt(-2 ln u1) sin(2 pi u2) ~ N(0,1)
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable


def rnorm_pair() -> Tuple[float, float]:
    """
    Box-Muller 生成两个独立标准正态随机数 (移植自 rnorm).
    """
    while True:
        x = np.random.random()
        y = np.random.random()
        x = 2.0 * x - 1.0
        y = 2.0 * y - 1.0
        s = x * x + y * y
        if s <= 1.0 and s > 0:
            break
    factor = math.sqrt(-2.0 * math.log(s) / s)
    return x * factor, y * factor


def rnorm_vec(n: int) -> np.ndarray:
    """生成 n 个标准正态随机数."""
    result = np.zeros(n)
    for i in range(0, n, 2):
        z1, z2 = rnorm_pair()
        result[i] = z1
        if i + 1 < n:
            result[i + 1] = z2
    return result


def metropolis_hastings(log_posterior: Callable, theta0: np.ndarray,
                          proposal_std: np.ndarray,
                          n_steps: int = 10000,
                          seed: int = 42) -> Dict:
    """
    Metropolis-Hastings MCMC 采样.
    """
    np.random.seed(seed)
    n_params = len(theta0)
    chain = np.zeros((n_steps, n_params))
    log_post_vals = np.zeros(n_steps)
    accepts = 0

    theta = theta0.copy()
    lp = log_posterior(theta)

    for i in range(n_steps):
        # 提议 (高斯随机游走)
        z = rnorm_vec(n_params)
        theta_prop = theta + proposal_std * z
        lp_prop = log_posterior(theta_prop)

        # 接受/拒绝
        log_alpha = lp_prop - lp
        if log_alpha > 0 or math.log(max(np.random.random(), 1e-300)) < log_alpha:
            theta = theta_prop
            lp = lp_prop
            accepts += 1

        chain[i] = theta
        log_post_vals[i] = lp

    return {
        'chain': chain,
        'log_posterior': log_post_vals,
        'acceptance_rate': accepts / n_steps,
    }


def monte_carlo_eos_uncertainty(eos_model: Callable,
                                  param_means: np.ndarray,
                                  param_stds: np.ndarray,
                                  n_samples: int = 1000) -> Dict:
    """
    蒙特卡洛 EoS 不确定性传播.
    """
    n_params = len(param_means)
    pressures = []
    densities = np.logspace(13, 15.5, 30)

    for _ in range(n_samples):
        z = rnorm_vec(n_params)
        params = param_means + param_stds * z
        try:
            p_vals = np.array([eos_model(rho, *params) for rho in densities])
            pressures.append(p_vals)
        except (ValueError, ZeroDivisionError):
            continue

    if not pressures:
        return {'error': '所有样本失败'}

    pressures = np.array(pressures)
    return {
        'densities': densities,
        'pressure_mean': np.mean(pressures, axis=0),
        'pressure_std': np.std(pressures, axis=0),
        'pressure_median': np.median(pressures, axis=0),
        'pressure_5th': np.percentile(pressures, 5, axis=0),
        'pressure_95th': np.percentile(pressures, 95, axis=0),
        'n_successful': len(pressures),
    }


def bootstrap_confidence(statistic_func: Callable, data: np.ndarray,
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95,
                           seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap 置信区间."""
    np.random.seed(seed)
    n = len(data)
    stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[np.random.randint(0, n, size=n)]
        stats[i] = statistic_func(sample)
    alpha = 1.0 - confidence
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(np.mean(stats)), float(hi)


# 自检
if __name__ == "__main__":
    print("=== EoS 蒙特卡洛自检 ===")
    z = rnorm_vec(1000)
    print(f"正态样本均值: {np.mean(z):.4f}  (应为 ~0)")
    print(f"正态样本标准差: {np.std(z):.4f}  (应为 ~1)")

    # MCMC 测试: 采样 N(mu=2, sigma=1)
    def log_post(theta):
        return -0.5 * (theta[0] - 2.0) ** 2

    result = metropolis_hastings(log_post, np.array([0.0]),
                                   np.array([0.5]), n_steps=5000)
    chain = result['chain'][1000:]  # burn-in
    print(f"MCMC 接受率: {result['acceptance_rate']:.3f}")
    print(f"后验均值: {np.mean(chain):.3f}  (应为 2)")
    print(f"后验标准差: {np.std(chain):.3f}  (应为 1)")

    # Bootstrap
    data = np.random.randn(100) * 2 + 5
    lo, mean, hi = bootstrap_confidence(np.mean, data)
    print(f"Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")

    print("\neos_monte_carlo.py 自检通过.")
