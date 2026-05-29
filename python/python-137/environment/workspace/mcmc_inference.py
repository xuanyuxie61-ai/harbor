# -*- coding: utf-8 -*-
"""
mcmc_inference.py

博士级贝叶斯参数推断库（DREAM MCMC）

融合原项目算法：
- 319_dream 的 Differential Evolution Adaptive Metropolis (DREAM) 算法

科学应用场景：
结晶动力学参数（k_g0, E_g, k_b, b_exp 等）通常无法直接测量，
需要通过拟合实验观测的 CSD 时序数据来反演。

DREAM 算法结合差分进化 (DE) 提议和自适应 Metropolis 接受准则，
能够高效地探索高维参数空间的后验分布。

数学模型：
1. 似然函数（高斯噪声假设）：
    L(θ|D) = ∏_{i=1}^{N_data} (1/√(2πσ²)) · exp(-(D_i - M_i(θ))²/(2σ²))
    log L = -0.5·N_data·ln(2πσ²) - 0.5·Σ_i (D_i - M_i(θ))²/σ²

2. 先验分布（对数正态，保证参数正定性）：
    log π(θ) = -0.5·Σ_j [(ln θ_j - μ_j)²/σ_j²] - Σ_j ln θ_j + const

3. DE 提议分布：
    z_p = z_{current} + (1+η)·γ·Σ_{pairs} (z_a - z_b) + ε
    其中 γ 为跳跃率，η ~ Uniform(-c, c)，ε ~ N(0, 10^{-10})

4. Metropolis-Hastings 接受准则：
    α = min(1, exp[(log L_new + log π_new) - (log L_old + log π_old)])

5. Gelman-Rubin 收敛诊断：
    R̂ = √[((n-1)/n · W + 1/n · B) / W]
    其中 W 为链内方差，B 为链间方差。
"""

import numpy as np


def gelman_rubin_diagnostic(chains):
    """
    计算 Gelman-Rubin R̂ 收敛诊断统计量。

    参数：
        chains : ndarray, shape (n_chains, n_samples, n_params)

    返回：
        R_hat : ndarray, shape (n_params,)
    """
    n_chains, n_samples, n_params = chains.shape
    if n_samples < 2:
        return np.full(n_params, np.inf)

    # 链均值
    chain_means = np.mean(chains, axis=1)  # (n_chains, n_params)
    # 总体均值
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # 链间方差 B
    B = n_samples * np.var(chain_means, axis=0, ddof=1)  # (n_params,)

    # 链内方差 W
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)  # (n_params,)

    # 估计方差
    var_estimate = ((n_samples - 1) / n_samples) * W + B / n_samples

    # R_hat
    R_hat = np.sqrt(var_estimate / np.where(W < 1e-30, 1e-30, W))
    return R_hat


def log_prior_lognormal(theta, mu_ln, sigma_ln):
    """
    对数正态先验的对数值。

    参数：
        theta : ndarray
            参数向量（必须 > 0）
        mu_ln : ndarray
            对数均值
        sigma_ln : ndarray
            对数标准差

    返回：
        log_pi : float
    """
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0):
        return -np.inf
    log_pi = -0.5 * np.sum(((np.log(theta) - mu_ln) / sigma_ln) ** 2)
    log_pi -= np.sum(np.log(theta))
    # 归一化常数省略（MCMC 中抵消）
    return log_pi


def log_likelihood_gaussian(theta, model_func, data, sigma_noise):
    """
    高斯似然函数的对数值。

    参数：
        theta : ndarray
        model_func : callable
            model_func(theta) -> predictions ndarray
        data : ndarray
        sigma_noise : float

    返回：
        log_L : float
    """
    try:
        predictions = model_func(theta)
        predictions = np.asarray(predictions, dtype=float)
        residuals = data - predictions
        n = data.size
        log_L = -0.5 * n * np.log(2.0 * np.pi * sigma_noise ** 2)
        log_L -= 0.5 * np.sum(residuals ** 2) / (sigma_noise ** 2)
        return log_L
    except Exception:
        return -np.inf


def dream_mcmc(log_posterior, n_params, n_chains=3, n_generations=2000,
               bounds=None, init_scale=0.1, rng=None,
               gr_threshold=1.01, burnin_fraction=0.5,
               de_pairs=3, gamma_base=2.38 / np.sqrt(2)):
    """
    DREAM (Differential Evolution Adaptive Metropolis) MCMC 算法。

    参数：
        log_posterior : callable
            log_posterior(theta) -> float
        n_params : int
            参数维数
        n_chains : int
            马尔可夫链数（至少 3 条）
        n_generations : int
            迭代代数
        bounds : ndarray, shape (2, n_params), optional
            [lower, upper] 边界
        init_scale : float
            初始分布的缩放
        rng : numpy.random.Generator
        gr_threshold : float
            Gelman-Rubin 收敛阈值
        burnin_fraction : float
            预烧期比例
        de_pairs : int
            DE 差分对数
        gamma_base : float
            基础跳跃率

    返回：
        samples : ndarray, shape (n_chains, n_samples, n_params)
            后验样本（去除预烧期）
        logpost : ndarray, shape (n_chains, n_samples)
            对数后验值
        R_hat : ndarray, shape (n_params,)
            最终收敛诊断
        acceptance_rate : float
    """
    if rng is None:
        rng = np.random.default_rng()
    if bounds is None:
        bounds = np.zeros((2, n_params))
        bounds[0, :] = 1e-12
        bounds[1, :] = 1e6

    n_chains = max(n_chains, 3)
    n_burnin = int(n_generations * burnin_fraction)
    n_keep = n_generations - n_burnin

    # 初始化链
    chains = np.zeros((n_chains, n_generations, n_params), dtype=float)
    logpost_values = np.zeros((n_chains, n_generations), dtype=float)

    for c in range(n_chains):
        # 在对数尺度上均匀采样初始化
        log_low = np.log(bounds[0, :])
        log_high = np.log(bounds[1, :])
        chains[c, 0, :] = np.exp(log_low + rng.random(n_params) * (log_high - log_low))
        logpost_values[c, 0] = log_posterior(chains[c, 0, :])

    accepted = 0
    total_proposals = 0

    max_pairs = (n_chains - 1) // 2
    actual_pairs = min(de_pairs, max_pairs)
    for gen in range(1, n_generations):
        for c in range(n_chains):
            # DE 提议
            # 随机选择 de_pairs 对不同的链
            other_chains = [i for i in range(n_chains) if i != c]
            selected = rng.choice(other_chains, size=2 * actual_pairs, replace=False)

            diff_sum = np.zeros(n_params)
            for p in range(actual_pairs):
                a = chains[selected[2 * p], gen - 1, :]
                b = chains[selected[2 * p + 1], gen - 1, :]
                diff_sum += (a - b)

            # 自适应跳跃率
            gamma = gamma_base
            # 以 10% 概率使用大跳跃以改善混合
            if rng.random() < 0.1:
                gamma = 1.0

            # 子空间采样：随机选择部分维度进行更新
            n_update = rng.integers(1, n_params + 1)
            update_dims = rng.choice(n_params, size=n_update, replace=False)

            proposal = chains[c, gen - 1, :].copy()
            eta = rng.uniform(-0.1, 0.1, n_update)
            epsilon = rng.normal(0, 1e-10, n_update)

            proposal[update_dims] += ((1.0 + eta) * gamma * diff_sum[update_dims] + epsilon)

            # 边界反射
            for j in range(n_params):
                if proposal[j] < bounds[0, j]:
                    proposal[j] = bounds[0, j] + (bounds[0, j] - proposal[j])
                    if proposal[j] < bounds[0, j]:
                        proposal[j] = bounds[0, j]
                elif proposal[j] > bounds[1, j]:
                    proposal[j] = bounds[1, j] - (proposal[j] - bounds[1, j])
                    if proposal[j] > bounds[1, j]:
                        proposal[j] = bounds[1, j]

            # 确保正值
            proposal = np.maximum(proposal, bounds[0, :])
            proposal = np.minimum(proposal, bounds[1, :])

            # 计算后验
            logpost_prop = log_posterior(proposal)
            logpost_curr = logpost_values[c, gen - 1]

            # Metropolis-Hastings 接受
            log_alpha = logpost_prop - logpost_curr
            if np.isnan(log_alpha):
                log_alpha = -np.inf

            if np.log(rng.random()) < log_alpha:
                chains[c, gen, :] = proposal
                logpost_values[c, gen] = logpost_prop
                accepted += 1
            else:
                chains[c, gen, :] = chains[c, gen - 1, :]
                logpost_values[c, gen] = logpost_curr
            total_proposals += 1

    # 去除预烧期
    samples = chains[:, n_burnin:, :]
    logpost_keep = logpost_values[:, n_burnin:]

    # Gelman-Rubin 诊断
    R_hat = gelman_rubin_diagnostic(samples)

    acceptance_rate = accepted / max(total_proposals, 1)

    return samples, logpost_keep, R_hat, acceptance_rate


def estimate_parameters_summary(samples):
    """
    从 MCMC 样本中提取参数估计的摘要统计。

    参数：
        samples : ndarray, shape (n_chains, n_samples, n_params)

    返回：
        summary : dict
            {'mean': ..., 'median': ..., 'std': ..., 'ci_95': ...}
    """
    flat = samples.reshape(-1, samples.shape[-1])
    mean = np.mean(flat, axis=0)
    median = np.median(flat, axis=0)
    std = np.std(flat, axis=0, ddof=1)
    ci_95 = np.percentile(flat, [2.5, 97.5], axis=0)

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'ci_lower': ci_95[0, :],
        'ci_upper': ci_95[1, :]
    }
