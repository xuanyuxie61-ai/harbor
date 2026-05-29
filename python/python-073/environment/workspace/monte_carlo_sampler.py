# -*- coding: utf-8 -*-
"""
monte_carlo_sampler.py
高维参数空间蒙特卡洛采样与不确定性量化

核心算法来源：
- hyperball_positive_distance: 高维超球随机采样与距离统计
- high_card_simulation: 最优停止策略（序贯采样）

物理背景：
高超声速边界层转捩位置对以下参数高度敏感：
    - 马赫数 Ma
    - 单位雷诺数 Re_1 = ρu/μ
    - 壁温比 Tw/Te
    - 表面粗糙度 k_s/δ
    - 自由流扰动强度 Tu

本模块在多维参数空间中执行拉丁超立方采样 (LHS) 与自适应蒙特卡洛，
量化参数不确定性对转捩位置预测的累积效应。
"""

import numpy as np
from math import sqrt, exp, pi, gamma as gamma_func


class HypersonicParameterSampler:
    """
    高超声速边界层参数空间采样器。
    """

    def __init__(self, Ma_range=(5.0, 8.0), Re_range=(1e5, 1e7),
                 Tw_Te_range=(0.5, 2.0), Tu_range=(0.001, 0.02)):
        """
        参数:
            Ma_range (tuple): 马赫数范围
            Re_range (tuple): 雷诺数范围
            Tw_Te_range (tuple): 壁温比范围
            Tu_range (tuple): 自由流湍流度范围 [%]
        """
        self.Ma_range = Ma_range
        self.Re_range = Re_range
        self.Tw_Te_range = Tw_Te_range
        self.Tu_range = Tu_range

    def lhs_sampling(self, n_samples):
        """
        拉丁超立方采样 (LHS)。

        将每个参数区间等分为 n_samples 个子区间，
        在每个子区间内随机采样，保证空间填充性。

        参数:
            n_samples (int): 样本数

        返回:
            np.ndarray: shape (n_samples, 4)，列对应 [Ma, Re, Tw/Te, Tu]
        """
        n_params = 4
        samples = np.zeros((n_samples, n_params))

        for p in range(n_params):
            perm = np.random.permutation(n_samples)
            u = (perm + np.random.rand(n_samples)) / n_samples
            if p == 0:
                samples[:, p] = self.Ma_range[0] + u * (self.Ma_range[1] - self.Ma_range[0])
            elif p == 1:
                log_min, log_max = np.log10(self.Re_range[0]), np.log10(self.Re_range[1])
                samples[:, p] = 10.0 ** (log_min + u * (log_max - log_min))
            elif p == 2:
                samples[:, p] = self.Tw_Te_range[0] + u * (self.Tw_Te_range[1] - self.Tw_Te_range[0])
            else:
                samples[:, p] = self.Tu_range[0] + u * (self.Tu_range[1] - self.Tu_range[0])

        return samples

    def hyperball_uniform_sample(self, m, n):
        """
        基于 hyperball_positive_sample 的高维正超球均匀采样。

        在单位正超球 {x ∈ R^m : ||x|| ≤ 1, x_i ≥ 0} 内生成 n 个随机点。
        算法：先生成指数分布样本，再归一化。

        参数:
            m (int): 空间维度
            n (int): 样本数

        返回:
            np.ndarray: shape (n, m)
        """
        X = np.random.exponential(scale=1.0, size=(n, m))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        radii = np.random.rand(n, 1) ** (1.0 / m)
        samples = radii * X / np.maximum(norms, 1e-15)
        return samples

    def parameter_distance_stats(self, samples):
        """
        基于 hyperball_positive_distance_stats 的样本距离统计。

        计算参数空间中样本对的欧氏距离均值与方差，
        用于评估采样均匀性。

        参数:
            samples (np.ndarray): shape (n, m)

        返回:
            tuple: (mu, var) 距离均值与方差
        """
        n = samples.shape[0]
        if n < 2:
            return 0.0, 0.0

        # 归一化到单位超球
        mins = np.min(samples, axis=0)
        maxs = np.max(samples, axis=0)
        ranges = np.maximum(maxs - mins, 1e-15)
        normalized = (samples - mins) / ranges

        # 计算所有样本对的距离
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(normalized[i] - normalized[j]))

        dists = np.array(dists)
        mu = np.mean(dists)
        var = np.var(dists, ddof=1) if len(dists) > 1 else 0.0
        return mu, var

    def sequential_optimal_sampling(self, n_total, n_skip=None):
        """
        基于 high_card_simulation 最优停止思想的序贯采样。

        在大量候选样本中，先观察 n_skip 个样本建立先验，
        随后选择第一个超过先验最大值的样本。

        类比秘书问题：在 n_total 个候选参数组合中，
        通过序贯策略最大化找到"最优"转捩敏感参数组合的概率。

        参数:
            n_total (int): 总候选数
            n_skip (int): 初始观察数，默认 n_total // e

        返回:
            dict: 最优样本索引与相关统计
        """
        if n_skip is None:
            n_skip = max(1, int(n_total / exp(1)))

        # 生成候选参数
        candidates = self.lhs_sampling(n_total)

        # 定义"价值"函数：参数组合的转捩敏感指数
        # 简化为参数偏离设计中心点的距离
        center = np.array([6.5, 1e6, 1.0, 0.01])
        scales = np.array([1.0, 1e6, 0.5, 0.01])
        values = np.linalg.norm((candidates - center) / scales, axis=1)

        # 最优停止策略
        if n_skip >= n_total:
            best_idx = np.argmax(values)
            return {'best_idx': best_idx, 'best_value': values[best_idx], 'strategy': 'random'}

        skip_max = np.max(values[:n_skip])
        best_idx = n_total - 1
        for i in range(n_skip, n_total):
            if values[i] > skip_max:
                best_idx = i
                break

        success = values[best_idx] == np.max(values)
        return {
            'best_idx': best_idx,
            'best_value': values[best_idx],
            'global_max': np.max(values),
            'success': success,
            'strategy': 'optimal_stop'
        }

    def uncertainty_propagation(self, transition_model, n_samples=500):
        """
        蒙特卡洛不确定性传播。

        对采样参数评估转捩位置，统计均值、方差与置信区间。

        参数:
            transition_model (callable): Re_t = f(Ma, Re, Tw_Te, Tu)
            n_samples (int): 样本数

        返回:
            dict: 统计结果
        """
        samples = self.lhs_sampling(n_samples)
        Re_t = np.zeros(n_samples)

        for i in range(n_samples):
            Ma, Re, Tw_Te, Tu = samples[i]
            try:
                Re_t[i] = transition_model(Ma, Re, Tw_Te, Tu)
            except Exception:
                Re_t[i] = np.nan

        valid = Re_t[~np.isnan(Re_t)]
        if len(valid) == 0:
            return {'mean': np.nan, 'std': np.nan, 'ci95': (np.nan, np.nan)}

        mean_val = np.mean(valid)
        std_val = np.std(valid, ddof=1)
        ci_low = np.percentile(valid, 2.5)
        ci_high = np.percentile(valid, 97.5)

        return {
            'mean': mean_val,
            'std': std_val,
            'ci95': (ci_low, ci_high),
            'samples': samples,
            'Re_t': Re_t
        }


def random_transition_model(Ma, Re, Tw_Te, Tu):
    """
    简化的转捩雷诺数经验模型（用于测试不确定性传播）。

    基于 Mack 第二模态与 Abu-Ghanam & Shaw 关联式的混合模型:
        Re_θ,t = C1 * Ma^{-C2} * (Tw/Te)^{C3} * Tu^{-C4}
        Re_x,t = Re_θ,t^2 / 常数

    参数:
        Ma (float): 马赫数
        Re (float): 单位雷诺数
        Tw_Te (float): 壁温比
        Tu (float): 湍流度

    返回:
        float: 转捩雷诺数 Re_x,t
    """
    C1 = 200.0
    C2 = 0.8
    C3 = 0.4
    C4 = 0.7

    Re_theta_t = C1 * (Ma ** (-C2)) * (Tw_Te ** C3) * (Tu ** (-C4))
    # 近似 Re_x ≈ Re_θ^2 * 常数
    Re_xt = (Re_theta_t ** 2) * 2.5 + np.random.normal(0, 1e4)
    return max(Re_xt, 1e4)
