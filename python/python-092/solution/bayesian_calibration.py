"""
bayesian_calibration.py
声学参数贝叶斯反演与不确定性量化
基于 dream (Differential Evolution Adaptive Metropolis MCMC) 核心算法重构

声学工程应用：
从实测（或模拟）的房间脉冲响应中提取混响时间 T60，
使用 DREAM MCMC 方法反演墙面吸声系数 α 的后验分布：

贝叶斯定理：
    p(α | D) ∝ p(D | α) * p(α)

其中：
    - α = [α_floor, α_ceiling, α_wall] 为待反演参数
    - D = {T60_meas(f_i)} 为观测数据
    - 似然函数假设为高斯：p(D|α) = N(T60_meas - T60_pred(α), σ²)

DREAM 算法核心（Vrugt et al., 2009）：
    1. 多条链并行演化
    2. 差分进化生成候选：z_p = z_i + (1+γ) * J * Σ(z_a - z_b) + ε
    3. Metropolis-Hastings 接受/拒绝
    4. 自适应交叉概率 CR
    5. Gelman-Rubin R 统计量诊断收敛
"""

import numpy as np


def sample_limits(params, lower, upper):
    """
    将参数限制在边界内（折叠法）。
    来自 dream 的 sample_limits 函数。
    """
    params = np.asarray(params, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    range_vals = upper - lower
    range_vals = np.where(range_vals < 1e-14, 1e-14, range_vals)
    # 折叠
    params = np.mod(params - lower, 2.0 * range_vals)
    mask = params > range_vals
    params[mask] = 2.0 * range_vals[mask] - params[mask]
    params = params + lower
    # 硬截断
    params = np.clip(params, lower, upper)
    return params


def diff_compute(chains, n_pairs=2):
    """
    计算差分进化向量：Σ(z_a - z_b) 从随机选取的链对。
    来自 dream 的 diff_compute 函数。
    """
    n_chains, n_params = chains.shape
    diff = np.zeros(n_params, dtype=float)
    available = list(range(n_chains))
    for _ in range(n_pairs):
        if len(available) < 2:
            break
        idx = np.random.choice(available, size=2, replace=False)
        diff += chains[idx[0]] - chains[idx[1]]
        available.remove(idx[0])
        available.remove(idx[1])
    return diff


def gr_compute(chains_history):
    """
    计算 Gelman-Rubin R 统计量用于收敛诊断。
    来自 dream 的 gr_compute 函数。

    R = sqrt((W + B/n) / W)
    其中 W 为链内方差，B 为链间方差。
    当 R < 1.2 时认为收敛。
    """
    n_gen, n_chains, n_params = chains_history.shape
    if n_gen < 2:
        return np.ones(n_params) * np.inf

    # 链均值
    chain_means = np.mean(chains_history, axis=0)
    # 总体均值
    overall_mean = np.mean(chain_means, axis=0)
    # 链间方差 B
    B = n_gen * np.var(chain_means, axis=0, ddof=1)
    # 链内方差 W
    W = np.mean(np.var(chains_history, axis=0, ddof=1), axis=0)
    W = np.where(W < 1e-14, 1e-14, W)
    R = np.sqrt((W + B / n_gen) / W)
    return R


def log_likelihood_gaussian(pred, obs, sigma):
    """
    高斯似然函数的对数：
        log L = -0.5 * Σ((pred_i - obs_i)/σ)² - N/2 * log(2πσ²)
    """
    residuals = pred - obs
    n = len(obs)
    log_l = -0.5 * np.sum((residuals / sigma) ** 2) - 0.5 * n * np.log(2.0 * np.pi * sigma ** 2)
    return log_l


def log_prior_uniform(params, lower, upper):
    """
    均匀先验的对数。
    """
    if np.any(params < lower) or np.any(params > upper):
        return -np.inf
    return 0.0  # 均匀先验的对数常数


def dream_mcmc(log_posterior_func, n_params, lower, upper,
               n_chains=5, n_generations=500,
               burn_in=100, gamma=1.0):
    """
    DREAM (Differential Evolution Adaptive Metropolis) MCMC 的简化实现。
    基于 dream 核心算法（Vrugt et al., 2009）。

    参数:
        log_posterior_func: 函数，输入参数向量，返回对数后验
        n_params: 参数维度
        lower, upper: 参数边界
        n_chains: 并行链数
        n_generations: 迭代代数
        burn_in:  burn-in 代数
        gamma: 差分进化缩放因子

    返回:
        chains: 所有链的采样 (n_generations, n_chains, n_params)
        log_probs: 对应的对数后验值
        R_stats: Gelman-Rubin 统计量历史
    """
    # 初始化链
    chains = np.zeros((n_generations, n_chains, n_params), dtype=float)
    log_probs = np.zeros((n_generations, n_chains), dtype=float)
    R_stats = []

    # 从先验中采样初始值
    for j in range(n_chains):
        chains[0, j, :] = lower + np.random.rand(n_params) * (upper - lower)
        log_probs[0, j] = log_posterior_func(chains[0, j, :])

    for g in range(1, n_generations):
        for j in range(n_chains):
            # 差分进化生成候选
            other_chains = np.delete(chains[g - 1], j, axis=0)
            diff = diff_compute(other_chains, n_pairs=1)
            # 添加噪声
            noise = np.random.randn(n_params) * 1e-6
            proposal = chains[g - 1, j, :] + gamma * diff + noise
            proposal = sample_limits(proposal, lower, upper)

            # Metropolis-Hastings 比率
            log_p_prop = log_posterior_func(proposal)
            log_ratio = log_p_prop - log_probs[g - 1, j]

            if np.log(np.random.rand()) < log_ratio:
                chains[g, j, :] = proposal
                log_probs[g, j] = log_p_prop
            else:
                chains[g, j, :] = chains[g - 1, j, :]
                log_probs[g, j] = log_probs[g - 1, j]

        # 每10代计算 R 统计量
        if g % 10 == 0 and g > 10:
            history = chains[max(0, g - 50):g + 1]
            R = gr_compute(history)
            R_stats.append(R.copy())

    R_stats = np.array(R_stats) if R_stats else np.ones((1, n_params))
    return chains, log_probs, R_stats


def calibrate_absorption_coefficients(T60_observed, T60_std,
                                       surface_areas, room_volume,
                                       n_chains=5, n_generations=300,
                                       burn_in=100):
    """
    反演房间表面吸声系数。
    参数模型：3个参数 [α_floor_and_ceiling, α_front_back, α_left_right]
    Sabine 公式：T60 = 0.161 * V / Σ(A_i * α_i)
    """
    area_floor = surface_areas.get('floor', 80.0)
    area_ceiling = surface_areas.get('ceiling', 80.0)
    area_front = surface_areas.get('front_wall', 50.0)
    area_back = surface_areas.get('back_wall', 50.0)
    area_left = surface_areas.get('left_wall', 40.0)
    area_right = surface_areas.get('right_wall', 40.0)

    def predict_t60(params):
        """根据参数预测 T60。"""
        alpha_fc = params[0]
        alpha_fb = params[1]
        alpha_lr = params[2]
        total_abs = (area_floor + area_ceiling) * alpha_fc + \
                    (area_front + area_back) * alpha_fb + \
                    (area_left + area_right) * alpha_lr
        total_abs = max(total_abs, 1e-10)
        return 0.161 * room_volume / total_abs

    def log_posterior(params):
        """对数后验 = 对数似然 + 对数先验。"""
        lp = log_prior_uniform(params, np.array([0.01, 0.01, 0.01]),
                               np.array([0.99, 0.99, 0.99]))
        if lp == -np.inf:
            return -np.inf
        T60_pred = predict_t60(params)
        ll = log_likelihood_gaussian(np.array([T60_pred]),
                                     np.array([T60_observed]),
                                     T60_std)
        return ll + lp

    lower = np.array([0.01, 0.01, 0.01])
    upper = np.array([0.99, 0.99, 0.99])
    chains, log_probs, R_stats = dream_mcmc(
        log_posterior, 3, lower, upper,
        n_chains=n_chains, n_generations=n_generations
    )

    # 去除 burn-in
    post_samples = chains[burn_in:, :, :].reshape(-1, 3)
    post_means = np.mean(post_samples, axis=0)
    post_stds = np.std(post_samples, axis=0)

    result = {
        'posterior_mean': post_means,
        'posterior_std': post_stds,
        'chains': chains,
        'R_final': R_stats[-1] if len(R_stats) > 0 else np.ones(3),
        'predicted_t60': predict_t60(post_means)
    }
    return result


def uncertainty_propagation(chains, surface_areas, room_volume, n_samples=100):
    """
    通过后验采样传播不确定性，计算 T60 的预测分布。
    """
    samples = chains.reshape(-1, chains.shape[2])
    if len(samples) > n_samples:
        idx = np.random.choice(len(samples), n_samples, replace=False)
        samples = samples[idx]

    area_floor = surface_areas.get('floor', 80.0)
    area_ceiling = surface_areas.get('ceiling', 80.0)
    area_front = surface_areas.get('front_wall', 50.0)
    area_back = surface_areas.get('back_wall', 50.0)
    area_left = surface_areas.get('left_wall', 40.0)
    area_right = surface_areas.get('right_wall', 40.0)

    t60_samples = []
    for params in samples:
        total_abs = (area_floor + area_ceiling) * params[0] + \
                    (area_front + area_back) * params[1] + \
                    (area_left + area_right) * params[2]
        total_abs = max(total_abs, 1e-10)
        t60 = 0.161 * room_volume / total_abs
        t60_samples.append(t60)

    return {
        'mean': float(np.mean(t60_samples)),
        'std': float(np.std(t60_samples)),
        'ci_95': (float(np.percentile(t60_samples, 2.5)),
                  float(np.percentile(t60_samples, 97.5)))
    }
