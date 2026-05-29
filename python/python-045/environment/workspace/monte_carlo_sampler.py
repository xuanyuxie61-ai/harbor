#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monte_carlo_sampler.py
高维参数空间蒙特卡洛采样器

融合种子项目：
  - 566_hypersphere_monte_carlo: 超球面均匀采样
  - 561_hypercube_surface_distance: 超立方体表面采样与距离统计
  - 178_circle_distance: 圆上随机点距离统计

核心功能：
  1. 单位超球面均匀采样（用于正则化参数搜索方向）
  2. 超立方体表面采样（用于模型参数边界约束）
  3. 圆/球面上随机距离统计（用于阻抗误差分析）
  4. MCMC Metropolis-Hastings 采样器（用于贝叶斯反演）
  5. 自适应协方差采样
"""

import numpy as np


def hypersphere01_sample(m, n):
    """
    在单位 m 维超球面上均匀采样 n 个点

    融合种子项目 566_hypersphere_monte_carlo 的核心算法。

    Algorithm: 生成 m 维标准正态分布随机向量，然后归一化到单位长度。
    """
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms[norms == 0.0] = 1.0  # 避免除零
    x = x / norms
    return x


def hypersphere01_monomial_integral(m, e):
    """
    计算超球面上单项式的精确积分

    ∫_{S^{m-1}} x_1^{e_1} ... x_m^{e_m} dS

    当任一 e_i 为奇数时，积分为 0。
    否则：
        I = 2 * Γ((e_1+1)/2) ... Γ((e_m+1)/2) / Γ((m + Σe_i)/2)
    """
    from math import gamma
    e = np.asarray(e, dtype=np.int32)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    if np.any(e % 2 == 1):
        return 0.0

    num = 1.0
    for ei in e:
        num *= gamma((ei + 1) / 2.0)
    denom = gamma((m + np.sum(e)) / 2.0)
    return 2.0 * num / denom


def hypercube_surface_sample(n_points, d):
    """
    在单位 d 维超立方体表面上均匀采样 n_points 个点

    融合种子项目 561_hypercube_surface_distance 的核心算法。
    """
    p = np.random.rand(n_points, d)
    # 选择每个点落在哪个维度面上
    i = np.random.randint(0, d, size=n_points)
    # 选择落在该维度的下表面 (0) 还是上表面 (1)
    s = np.random.randint(0, 2, size=n_points)
    for idx in range(n_points):
        p[idx, i[idx]] = float(s[idx])
    return p


def circle_unit_sample():
    """
    在单位圆上均匀采样一个点

    融合种子项目 178_circle_distance 的核心算法。
    """
    theta = 2.0 * np.pi * np.random.rand()
    return np.array([np.cos(theta), np.sin(theta)])


def circle_distance_stats(n_samples):
    """
    估计单位圆上两随机点之间距离的统计量

    理论值: E[d] = 4/π ≈ 1.2732
    """
    distances = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        p = circle_unit_sample()
        q = circle_unit_sample()
        distances[i] = np.linalg.norm(p - q)
    mu = np.mean(distances)
    var = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    return mu, var


def hypercube_surface_distance_stats(n_samples, d):
    """
    估计单位超立方体表面上两随机点之间距离的统计量
    """
    p1 = hypercube_surface_sample(n_samples, d)
    p2 = hypercube_surface_sample(n_samples, d)
    distances = np.linalg.norm(p1 - p2, axis=1)
    mu = np.mean(distances)
    var = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    return mu, var


class MetropolisHastingsSampler:
    """
    Metropolis-Hastings MCMC 采样器

    用于贝叶斯大地电磁反演中，从后验分布 p(m|d) ∝ p(d|m) * p(m) 中采样。
    """

    def __init__(self, log_target, proposal_cov, bounds=None):
        """
        Parameters
        ----------
        log_target : callable
            对数目标函数 log π(m)
        proposal_cov : ndarray
            提议分布的协方差矩阵
        bounds : list of tuple or None
            各参数的下界和上界 [(lb, ub), ...]
        """
        self.log_target = log_target
        self.proposal_cov = np.asarray(proposal_cov, dtype=np.float64)
        self.dim = self.proposal_cov.shape[0]
        self.bounds = bounds

    def _proposal(self, current):
        """高斯随机游走提议"""
        proposal = np.random.multivariate_normal(current, self.proposal_cov)
        if self.bounds is not None:
            for i, (lb, ub) in enumerate(self.bounds):
                proposal[i] = np.clip(proposal[i], lb, ub)
        return proposal

    def sample(self, initial, n_samples, burn_in=1000, thinning=10):
        """
        执行 MCMC 采样

        Parameters
        ----------
        initial : ndarray
            初始参数向量
        n_samples : int
            采样的样本数
        burn_in : int
            预烧期迭代次数
        thinning : int
            稀释间隔

        Returns
        -------
        samples : ndarray, shape (n_samples, dim)
        acceptance_rate : float
        """
        current = np.asarray(initial, dtype=np.float64)
        current_log = self.log_target(current)

        # 预烧期
        for _ in range(burn_in):
            proposal = self._proposal(current)
            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))
            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log

        # 采样
        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        accepted = 0
        total = 0
        idx = 0
        step = 0

        while idx < n_samples:
            proposal = self._proposal(current)
            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))
            total += 1
            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log
                accepted += 1

            step += 1
            if step % thinning == 0:
                samples[idx] = current
                idx += 1

        acceptance_rate = accepted / total if total > 0 else 0.0
        return samples, acceptance_rate


class AdaptiveCovarianceSampler:
    """
    自适应协方差 MCMC 采样器 (Haario et al., 2001)

    在采样过程中自适应调整提议分布的协方差，提高效率。
    """

    def __init__(self, log_target, initial_cov, bounds=None,
                 adapt_interval=100, target_rate=0.234):
        self.log_target = log_target
        self.cov = np.array(initial_cov, dtype=np.float64, copy=True)
        self.dim = self.cov.shape[0]
        self.bounds = bounds
        self.adapt_interval = adapt_interval
        self.target_rate = target_rate
        self.scale = 2.4 ** 2 / self.dim
        self._chain = []

    def sample(self, initial, n_samples, burn_in=1000):
        current = np.asarray(initial, dtype=np.float64)
        current_log = self.log_target(current)
        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        accepted = 0

        for step in range(-burn_in, n_samples):
            if step >= 0 and len(self._chain) > self.dim:
                # 使用链的经验协方差
                emp_cov = np.cov(np.array(self._chain).T)
                # 保证正定性
                emp_cov += 1e-8 * np.eye(self.dim)
                prop_cov = self.scale * emp_cov
            else:
                prop_cov = self.cov

            proposal = np.random.multivariate_normal(current, prop_cov)
            if self.bounds is not None:
                for i, (lb, ub) in enumerate(self.bounds):
                    proposal[i] = np.clip(proposal[i], lb, ub)

            prop_log = self.log_target(proposal)
            alpha = min(1.0, np.exp(prop_log - current_log))

            if np.random.rand() < alpha:
                current = proposal
                current_log = prop_log
                if step >= 0:
                    accepted += 1

            if step >= 0:
                samples[step] = current
                self._chain.append(current.copy())

        acceptance_rate = accepted / n_samples if n_samples > 0 else 0.0
        return samples, acceptance_rate


if __name__ == "__main__":
    # 自检
    x = hypersphere01_sample(3, 1000)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    print(f"超球面采样范数均值: {np.mean(norms):.6f} (应为 1.0)")

    mu, var = circle_distance_stats(10000)
    print(f"圆上距离统计: 均值={mu:.4f}, 方差={var:.4f} (理论均值≈1.2732)")

    mu_h, var_h = hypercube_surface_distance_stats(5000, 3)
    print(f"立方体表面距离: 均值={mu_h:.4f}, 方差={var_h:.4f}")

    # 简单 MCMC 测试
    def log_target(x):
        return -0.5 * np.sum(x ** 2)

    sampler = MetropolisHastingsSampler(log_target, 0.5 * np.eye(2))
    samples, rate = sampler.sample(np.zeros(2), 500, burn_in=200, thinning=5)
    print(f"MCMC 采样均值: {np.mean(samples, axis=0)}, 接受率: {rate:.3f}")
