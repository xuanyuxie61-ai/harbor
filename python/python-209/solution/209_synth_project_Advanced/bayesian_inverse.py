"""
bayesian_inverse.py — 贝叶斯反问题与模拟推断 (SBI)
=====================================================

本模块实现随机 PDE 参数的贝叶斯推断:
    给定观测 d_obs, 推断后验 p(ξ|d_obs)

核心方法:
  1. 先验分布: p(ξ) = N(0, I) 或 Uniform
  2. 似然函数: p(d|ξ) ∝ exp(-||d - G(ξ)||²/(2σ²))
  3. 模拟推断: 通过前向模拟近似后验
  4. 后验验证指标: C2ST, MMD, KSD

数学框架:
    贝叶斯定理: p(ξ|d) = p(d|ξ)·p(ξ) / p(d)
    MAP 估计: ξ_MAP = argmax p(ξ|d)
    后验均值: E[ξ|d] = ∫ ξ·p(ξ|d) dξ

映射种子项目:
  - 1111_sbi-benchmark_results: SBI 基准框架, C2ST/MMD/KSD 指标
"""

import numpy as np


# ============================================================
# 第1部分: 先验分布
# ============================================================

class GaussianPrior:
    """高斯先验: ξ ~ N(μ₀, Σ₀)"""

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.dim = len(mean)
        self.cov_inv = np.linalg.inv(self.cov + 1e-10 * np.eye(self.dim))
        self.log_det = np.linalg.slogdet(self.cov)[1]

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def log_pdf(self, xi):
        diff = xi - self.mean
        return -0.5 * (self.dim * np.log(2 * np.pi) + self.log_det +
                        diff @ self.cov_inv @ diff)

    def pdf(self, xi):
        return np.exp(self.log_pdf(xi))


class UniformPrior:
    """均匀先验: ξ ~ U([a,b]^d)"""

    def __init__(self, low, high):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.dim = len(low)
        self.volume = np.prod(self.high - self.low)

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def log_pdf(self, xi):
        if np.all(xi >= self.low) and np.all(xi <= self.high):
            return -np.log(self.volume)
        return -np.inf

    def pdf(self, xi):
        if np.all(xi >= self.low) and np.all(xi <= self.high):
            return 1.0 / self.volume
        return 0.0


# ============================================================
# 第2部分: 似然函数
# ============================================================

def gaussian_log_likelihood(observation, forward_output, noise_std=0.1):
    """
    高斯似然:
        log p(d|ξ) = -||d - G(ξ)||²/(2σ²) - (dim/2)·log(2πσ²)
    """
    diff = np.asarray(observation) - np.asarray(forward_output)
    dim = len(diff)
    return -np.sum(diff ** 2) / (2 * noise_std ** 2) - \
           (dim / 2) * np.log(2 * np.pi * noise_std ** 2)


# ============================================================
# 第3部分: 模拟推断 ( rejection sampling / ABC )
# ============================================================

def rejection_sampling_inference(prior, forward_model, observation,
                                 noise_std=0.1, n_proposals=5000,
                                 threshold=None, seed=42):
    """
    拒绝采样后验推断:
        1. 从先验采样 ξⁱ ~ p(ξ)
        2. 计算前向模拟 dⁱ = G(ξⁱ)
        3. 接受: ||dⁱ - d_obs|| < ε (ABC)
        或按权重: wⁱ = p(d_obs|ξⁱ) (重要性采样)

    返回:
        samples: (M, d) 后验样本
        weights: (M,) 重要性权重
        acceptance_rate: 接受率
    """
    rng = np.random.default_rng(seed)
    proposals = prior.sample(n_proposals, seed=seed)
    log_weights = np.zeros(n_proposals)

    for i in range(n_proposals):
        try:
            d_sim = forward_model(proposals[i])
            log_weights[i] = gaussian_log_likelihood(observation, d_sim, noise_std)
        except Exception:
            log_weights[i] = -1e10

    # 归一化权重
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    weights /= np.sum(weights) + 1e-30

    # 有效样本量
    ess = 1.0 / np.sum(weights ** 2)

    # 重采样
    n_resample = min(1000, n_proposals)
    indices = rng.choice(n_proposals, size=n_resample, p=weights)
    samples = proposals[indices]

    # 接受率 (ABC 意义)
    if threshold is None:
        threshold = np.median([np.sqrt(np.sum(
            (forward_model(proposals[i]) - observation) ** 2))
            for i in range(min(100, n_proposals))])
    accepted = sum(1 for i in range(n_proposals)
                   if np.sqrt(np.sum((forward_model(proposals[i]) - observation) ** 2))
                   < threshold)
    acceptance_rate = accepted / n_proposals

    return samples, weights, acceptance_rate, ess


# ============================================================
# 第4部分: MAP 估计
# ============================================================

def map_estimation(prior, forward_model, observation, noise_std=0.1,
                   n_starts=5, max_iter=100, lr=0.01, seed=42):
    """
    最大后验 (MAP) 估计:
        ξ_MAP = argmax log p(ξ|d) = argmax [log p(d|ξ) + log p(ξ)]

    使用多起点梯度上升:
        ξ^{k+1} = ξ^k + lr · ∇_ξ [log p(d|ξ) + log p(ξ)]

    返回:
        xi_map: MAP 估计
        log_posterior: MAP 处的后验值
    """
    rng = np.random.default_rng(seed)
    dim = prior.dim if hasattr(prior, 'dim') else 1

    best_xi = None
    best_logp = -np.inf

    for start in range(n_starts):
        if isinstance(prior, GaussianPrior):
            xi = prior.mean + 0.5 * rng.standard_normal(dim)
        else:
            xi = prior.sample(1, seed=seed + start)[0]

        for it in range(max_iter):
            # 数值梯度
            eps = 1e-5
            grad = np.zeros(dim)
            try:
                d0 = forward_model(xi)
                lp0 = gaussian_log_likelihood(observation, d0, noise_std) + prior.log_pdf(xi)
            except Exception:
                break

            for k in range(dim):
                xi_plus = xi.copy()
                xi_plus[k] += eps
                try:
                    d_plus = forward_model(xi_plus)
                    lp_plus = gaussian_log_likelihood(observation, d_plus, noise_std) + \
                              prior.log_pdf(xi_plus)
                    grad[k] = (lp_plus - lp0) / eps
                except Exception:
                    grad[k] = 0.0

            xi = xi + lr * grad
            # 梯度衰减
            lr *= 0.999

        try:
            d_final = forward_model(xi)
            logp = gaussian_log_likelihood(observation, d_final, noise_std) + prior.log_pdf(xi)
        except Exception:
            logp = -np.inf

        if logp > best_logp:
            best_logp = logp
            best_xi = xi.copy()

    return best_xi, best_logp


# ============================================================
# 第5部分: 后验验证指标
# (映射自 1111_sbi-benchmark_results: C2ST, MMD, KSD)
# ============================================================

def c2st_metric(samples_ref, samples_test, n_folds=3, seed=42):
    """
    Classifier Two-Sample Test (C2ST):
        训练分类器区分两组样本, 准确率 ≈ 0.5 表示分布一致。

    实现: 简单二次判别 (QDA-like)

    返回:
        accuracy: 分类准确率 (0.5 = 一致, 1.0 = 完全不同)
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(samples_ref), len(samples_test)
    if samples_ref.ndim == 1:
        samples_ref = samples_ref.reshape(-1, 1)
    if samples_test.ndim == 1:
        samples_test = samples_test.reshape(-1, 1)
    dim = samples_ref.shape[1]

    X = np.vstack([samples_ref, samples_test])
    y = np.array([0] * n1 + [1] * n2)

    # 标准化
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-10
    X_norm = (X - mu) / sigma

    # K-fold CV
    accuracies = []
    perm = rng.permutation(len(X))
    fold_size = len(X) // n_folds

    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)

        # 简单线性分类器
        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_test, y_test = X_norm[test_idx], y[test_idx]

        # 类条件均值
        mu0 = np.mean(X_train[y_train == 0], axis=0)
        mu1 = np.mean(X_train[y_train == 1], axis=0)
        # 决策边界: 中点投影
        w = mu1 - mu0
        w_norm = w / (np.linalg.norm(w) + 1e-10)
        threshold = 0.5 * (mu0 + mu1) @ w_norm

        preds = (X_test @ w_norm > threshold).astype(int)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)

    return np.mean(accuracies)


def mmd_metric(samples_ref, samples_test, bandwidth=None):
    """
    Maximum Mean Discrepancy (MMD):
        MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    使用高斯核: k(x,y) = exp(-||x-y||²/(2h²))

    返回:
        mmd_squared: MMD² 值 (越小越一致)
    """
    if samples_ref.ndim == 1:
        samples_ref = samples_ref.reshape(-1, 1)
    if samples_test.ndim == 1:
        samples_test = samples_test.reshape(-1, 1)

    if bandwidth is None:
        # 中位数启发式
        dists = np.sqrt(np.sum((samples_ref[0:1] - samples_test) ** 2, axis=1))
        bandwidth = max(np.median(dists), 1e-5)

    def gram_matrix(X, Y, h):
        n, m = len(X), len(Y)
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = np.exp(-np.sum((X[i] - Y[j]) ** 2) / (2 * h ** 2))
        return K

    K_xx = gram_matrix(samples_ref, samples_ref, bandwidth)
    K_yy = gram_matrix(samples_test, samples_test, bandwidth)
    K_xy = gram_matrix(samples_ref, samples_test, bandwidth)

    mmd2 = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    return max(mmd2, 0.0)


def ksd_metric(samples, score_func, bandwidth=None):
    """
    Kernelized Stein Discrepancy (KSD):
        KSD² = E_x[E_{x'}[k_p(x,x')]]
    其中 k_p 是 Steinfeld 核:
        k_p(x,y) = ∇_x log p(x)ᵀ k(x,y) ∇_y log p(y) + ...

    简化版: 使用有限差分近似 score。

    返回:
        ksd_squared: KSD² 值
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n, dim = samples.shape
    if bandwidth is None:
        bandwidth = 1.0

    # 计算 score (∇log p) 近似
    scores = np.zeros((n, dim))
    for i in range(n):
        s = score_func(samples[i])
        scores[i] = s if s is not None else np.zeros(dim)

    # 简化 KSD
    ksd2 = 0.0
    count = 0
    for i in range(min(n, 100)):
        for j in range(min(n, 100)):
            diff = samples[i] - samples[j]
            r2 = np.sum(diff ** 2)
            k = np.exp(-r2 / (2 * bandwidth ** 2))
            # Steinfeld 核近似
            term1 = np.dot(scores[i], scores[j]) * k
            term2 = np.dot(scores[i], diff) * k / bandwidth ** 2
            term3 = np.dot(scores[j], -diff) * k / bandwidth ** 2
            term4 = k * (dim / bandwidth ** 2 - r2 / bandwidth ** 4)
            ksd2 += term1 + term2 + term3 + term4
            count += 1
    return max(ksd2 / max(count, 1), 0.0)


def compute_all_posterior_metrics(samples_ref, samples_test, score_func=None):
    """
    计算全部后验验证指标 (映射自 sbi-benchmark results.py):
      - C2ST: 分类器二样本检验
      - MMD: 最大均值差异
      - Median Distance: 中位距离
    """
    c2st = c2st_metric(samples_ref, samples_test)
    mmd = mmd_metric(samples_ref, samples_test)
    if samples_ref.ndim == 1:
        samples_ref = samples_ref.reshape(-1, 1)
    if samples_test.ndim == 1:
        samples_test = samples_test.reshape(-1, 1)
    med_dist = np.median([np.min(np.sqrt(np.sum((samples_test - r) ** 2, axis=1)))
                          for r in samples_ref[:min(50, len(samples_ref))]])
    results = {
        'c2st_accuracy': c2st,
        'mmd_squared': mmd,
        'median_distance': med_dist
    }
    return results
