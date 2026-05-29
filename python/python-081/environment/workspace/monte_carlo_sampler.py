"""
蒙特卡洛采样与统计验证模块
==========================
基于种子项目:
  - 779_monty_hall_simulation: 蒙特卡洛随机模拟

科学背景:
  蒙特卡洛方法(Monte Carlo, MC)是不确定性量化的基准验证工具。
  通过大量随机采样，可估计响应量的统计矩、概率密度及失效概率。
  本模块实现：
  1. 基于截断正态分布的材料参数随机采样
  2. 有限元响应的蒙特卡洛估计
  3. 统计收敛性分析与置信区间估计

关键公式:
  - 样本均值: μ̂ = (1/N) Σ y_i
  - 样本方差: σ̂^2 = (1/(N-1)) Σ (y_i - μ̂)^2
  - 标准误: SE = σ̂ / sqrt(N)
  - 95% 置信区间: [μ̂ - 1.96 SE, μ̂ + 1.96 SE]
  - 失效概率: P_f = (1/N) Σ I(g(x_i) < 0)
  - Monte Carlo 收敛率: O(N^{-1/2})
"""

import numpy as np
from typing import Callable, Tuple, Optional
from uncertainty_quantification import truncated_normal_sample


def monte_carlo_simulation(model_func: Callable[[np.ndarray], float],
                            mu_params: np.ndarray,
                            sigma_params: np.ndarray,
                            bounds: np.ndarray,
                            n_samples: int = 1000,
                            seed: int = 42) -> dict:
    """
    对具有截断正态不确定参数的模型执行蒙特卡洛模拟。

    参数:
        model_func: 模型函数，输入为参数向量，输出为标量响应
        mu_params: (n_param,) 参数均值
        sigma_params: (n_param,) 参数标准差
        bounds: (n_param, 2) 参数截断区间 [[a1,b1], [a2,b2], ...]
        n_samples: 蒙特卡洛采样数
        seed: 随机种子

    返回:
        results: 包含统计结果的字典
    """
    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    responses = np.zeros(n_samples, dtype=np.float64)
    param_samples = np.zeros((n_samples, n_param), dtype=np.float64)

    for i in range(n_samples):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        param_samples[i] = sample
        try:
            responses[i] = model_func(sample)
        except Exception:
            responses[i] = np.nan

    # 边界处理: 移除NaN值
    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask]
    n_valid = len(valid_responses)

    if n_valid == 0:
        raise ValueError("所有蒙特卡洛样本均产生NaN，模型不稳定")

    mean_val = float(np.mean(valid_responses))
    std_val = float(np.std(valid_responses, ddof=1))
    se = std_val / np.sqrt(n_valid)
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se

    # 分位数
    q25 = float(np.percentile(valid_responses, 25))
    q50 = float(np.percentile(valid_responses, 50))
    q75 = float(np.percentile(valid_responses, 75))

    return {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "mean": mean_val,
        "std": std_val,
        "se": se,
        "ci_95": (ci_lower, ci_upper),
        "q25": q25,
        "median": q50,
        "q75": q75,
        "min": float(np.min(valid_responses)),
        "max": float(np.max(valid_responses)),
        "responses": valid_responses,
        "param_samples": param_samples[valid_mask],
    }


def estimate_failure_probability(model_func: Callable[[np.ndarray], float],
                                  threshold: float,
                                  mu_params: np.ndarray,
                                  sigma_params: np.ndarray,
                                  bounds: np.ndarray,
                                  n_samples: int = 5000,
                                  seed: int = 42) -> dict:
    """
    使用蒙特卡洛方法估计失效概率 P_f = P(model_func(params) < threshold)。

    参数:
        model_func: 模型函数
        threshold: 失效阈值
        mu_params, sigma_params, bounds: 参数分布
        n_samples: 采样数
        seed: 随机种子

    返回:
        包含失效概率估计和置信区间的字典
    """
    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    failures = 0
    responses = []

    for _ in range(n_samples):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        try:
            y = model_func(sample)
            responses.append(y)
            if y < threshold:
                failures += 1
        except Exception:
            pass

    pf = failures / n_samples if n_samples > 0 else 0.0
    # 二项分布近似置信区间 (Wilson score interval)
    z = 1.96
    n = n_samples
    denom = 1 + z**2 / n
    centre = (pf + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((pf * (1 - pf) + z**2 / (4 * n)) / n) / denom

    return {
        "failure_probability": pf,
        "pf_ci_lower": max(0.0, centre - margin),
        "pf_ci_upper": min(1.0, centre + margin),
        "n_failures": failures,
        "n_samples": n_samples,
    }


def convergence_analysis(model_func: Callable[[np.ndarray], float],
                          mu_params: np.ndarray,
                          sigma_params: np.ndarray,
                          bounds: np.ndarray,
                          sample_sizes: Optional[np.ndarray] = None,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    分析蒙特卡洛估计随样本量增加的收敛行为。

    返回:
        Ns: 样本量数组
        means: 对应均值估计数组
        stds: 对应标准差估计数组
    """
    if sample_sizes is None:
        sample_sizes = np.array([50, 100, 200, 500, 1000, 2000], dtype=np.int32)

    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    means = []
    stds = []

    max_n = int(np.max(sample_sizes))
    all_samples = np.zeros((max_n, n_param), dtype=np.float64)
    all_responses = np.zeros(max_n, dtype=np.float64)

    for i in range(max_n):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        all_samples[i] = sample
        try:
            all_responses[i] = model_func(sample)
        except Exception:
            all_responses[i] = np.nan

    for N in sample_sizes:
        vals = all_responses[:N]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            means.append(float(np.mean(valid)))
            stds.append(float(np.std(valid, ddof=1)))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return sample_sizes, np.array(means), np.array(stds)
