"""
uncertainty_quantify.py — 数值计算的不确定度量化
============================================================

本模块使用贝叶斯方法量化数值计算的不确定度:

1. 先验分布: 数值参数的先验知识
    - 磁场 B ~ N(B₀, σ_B)
    - 网格间距 h ~ Uniform(h_min, h_max)
    - 有限差分阶数 p ~ Discrete

2. 似然函数: 数值解对参数的敏感度
    L(data|params) ∝ exp(-χ²/2)

3. 后验分布: 给定数据后的参数分布
    P(params|data) ∝ L(data|params) · P(params)

4. 预测分布: 未来计算的预测区间
    P(E_new|data) = ∫ P(E_new|params) P(params|data) dparams

在量子霍尔数值计算中, 不确定度来源:
    - 有限格点截断误差: δE ~ h^p
    - 磁场不确定性: δB/B ~ 10⁻³
    - 边界条件近似: 有限尺寸效应
    - 数值精度: 浮点运算舍入

参考文献:
    [1] Gelman, A. et al. "Bayesian Data Analysis" (CRC, 2013)
    [2] Smith, R. "Uncertainty Quantification" (SIAM, 2013)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Callable


def gaussian_prior(mu: float, sigma: float,
                   x: np.ndarray) -> np.ndarray:
    """高斯先验分布

    P(x) = (1/σ√2π) exp(-(x-μ)²/(2σ²))

    用于磁场 B 的不确定度建模:
        B = B₀ ± δB, δB ~ N(0, σ_B)

    Args:
        mu: 均值
        sigma: 标准差
        x: 采样点
    Returns:
        概率密度
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def polynomial_chaos_expansion(coeffs: np.ndarray,
                                xi: np.ndarray) -> np.ndarray:
    """多项式混沌展开 (Polynomial Chaos Expansion)

    将随机输出展开为正交多项式基:
        Y(ξ) = Σ_{k=0}^P c_k · Φ_k(ξ)

    其中 Φ_k 是 Hermite 多项式 (对高斯输入):
        Φ_0 = 1, Φ_1 = ξ, Φ_2 = (ξ²-1)/√2, ...

    用于不确定度传播:
        E[Y] = c_0
        Var[Y] = Σ_{k>0} c_k²

    Args:
        coeffs: PCE 系数
        xi: 标准正态随机变量
    Returns:
        随机输出的样本
    """
    from scipy.special import eval_hermitenorm

    result = np.zeros_like(xi)
    for k, c in enumerate(coeffs):
        result += c * eval_hermitenorm(k, xi)
    return result


def monte_carlo_uncertainty(model: Callable, param_ranges: Dict[str, Tuple],
                            n_samples: int = 1000,
                            seed: int = 42) -> Dict[str, Any]:
    """Monte Carlo 不确定度传播

    1. 从参数先验中采样: {B, h, fd_order, ...}
    2. 对每组参数运行模型: E_i = model(params_i)
    3. 统计输出的分布: mean, std, percentiles

    Args:
        model: 数值模型函数 (接受参数字典, 返回能量)
        param_ranges: 参数范围 {name: (mean, std) 或 (low, high)}
        n_samples: 采样数
        seed: 随机种子
    Returns:
        不确定度分析结果
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_samples):
        params = {}
        for name, range_spec in param_ranges.items():
            if len(range_spec) == 2:
                low, high = range_spec
                params[name] = rng.uniform(low, high)
            elif len(range_spec) == 3:
                mu, sigma, dist = range_spec
                if dist == 'normal':
                    params[name] = rng.normal(mu, sigma)
                elif dist == 'uniform':
                    params[name] = rng.uniform(mu - sigma, mu + sigma)
            else:
                params[name] = range_spec[0]

        try:
            result = model(params)
            results.append(result)
        except Exception:
            results.append(np.nan)

    results = np.array(results)
    valid = ~np.isnan(results)

    return {
        'n_samples': n_samples,
        'n_valid': np.sum(valid),
        'mean': np.mean(results[valid]) if np.any(valid) else np.nan,
        'std': np.std(results[valid]) if np.any(valid) else np.nan,
        'median': np.median(results[valid]) if np.any(valid) else np.nan,
        'ci_95': (np.percentile(results[valid], 2.5),
                  np.percentile(results[valid], 97.5)) if np.any(valid) else (np.nan, np.nan),
        'all_results': results,
    }


def sensitivity_analysis(model: Callable, base_params: Dict[str, float],
                         param_name: str,
                         variations: np.ndarray) -> Tuple[np.ndarray, float]:
    """单参数敏感度分析

    计算输出对单个参数的偏导数:
        ∂E/∂p ≈ (E(p+δ) - E(p-δ)) / (2δ)

    以及归一化敏感度:
        S = (p/E) · (∂E/∂p)

    Args:
        model: 模型函数
        base_params: 基准参数
        param_name: 要分析灵敏度的参数名
        variations: 参数变化范围
    Returns:
        (energies, sensitivity): 能量数组和敏感度
    """
    energies = []
    for val in variations:
        params = base_params.copy()
        params[param_name] = val
        try:
            E = model(params)
            energies.append(E)
        except Exception:
            energies.append(np.nan)

    energies = np.array(energies)

    # 有限差分估计敏感度
    valid = ~np.isnan(energies)
    if np.sum(valid) >= 2:
        dE = np.gradient(energies[valid], variations[valid])
        E_base = energies[valid][len(energies[valid])//2]
        p_base = variations[valid][len(variations[valid])//2]
        sensitivity = (p_base / E_base) * np.mean(dE) if abs(E_base) > 1e-10 else 0.0
    else:
        sensitivity = 0.0

    return energies, sensitivity


def convergence_extrapolation(h_values: np.ndarray,
                               E_values: np.ndarray,
                               order: int = 4) -> Tuple[float, float]:
    """Richardson 外推估计精确值

    假设 E_h = E_exact + C·h^p:
        使用多个 h 值的计算结果拟合 E_exact 和 p.

    对于 4 阶有限差分:
        E_h = E_exact + C·h⁴ + O(h⁵)

    双网格外推:
        E_exact ≈ (2^p · E_{h/2} - E_h) / (2^p - 1)

    Args:
        h_values: 网格间距数组
        E_values: 对应能量数组
        order: 预期收敛阶数
    Returns:
        (E_extrapolated, estimated_order)
    """
    valid = ~np.isnan(E_values) & ~np.isinf(E_values)
    h_valid = h_values[valid]
    E_valid = E_values[valid]

    if len(h_valid) < 2:
        return np.nan, np.nan

    # 对数-对数线性拟合
    log_h = np.log(h_valid)
    # 估计 E_exact (最小 E 作为参考)
    E_ref = np.min(E_valid)
    dE = E_valid - E_ref
    dE = dE[dE > 1e-15]
    log_dE = np.log(dE)
    log_h_fit = log_h[valid][:(len(dE))]

    if len(log_h_fit) >= 2 and len(log_dE) >= 2:
        min_len = min(len(log_h_fit), len(log_dE))
        slope, intercept = np.polyfit(log_h_fit[:min_len], log_dE[:min_len], 1)
        est_order = slope
        E_exact = E_ref - np.exp(intercept)  # 近似
    else:
        est_order = order
        E_exact = E_valid[0]

    return E_exact, est_order


def bootstrap_confidence_interval(data: np.ndarray,
                                   statistic: Callable = np.mean,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95,
                                   seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap 置信区间

    从原始数据中有放回采样, 计算统计量的分布,
    然后取分位数作为置信区间.

    Args:
        data: 原始数据
        statistic: 统计量函数
        n_bootstrap: bootstrap 采样数
        confidence: 置信水平
        seed: 随机种子
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = statistic(sample)

    point_est = statistic(data)
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return point_est, lower, upper
