# -*- coding: utf-8 -*-
"""
sampling_engine.py
==================
参数空间采样与不确定性量化模块。

融合原始项目:
  - 649_latin_center: 拉丁超立方中心采样（高维空间均匀覆盖）

核心数学公式
------------
1. 拉丁超立方采样 (Latin Hypercube Sampling, LHS):
   对 d 维空间，每维分成 n 等份，
   在每个子区间 [ (k-1)/n, k/n ) 中恰好放置一个样本点。
   中心采样取: x_{ij} = (2·π_j(i) - 1) / (2n)
   其中 π_j 为第 j 维的随机排列。

2. Latin Center 采样:
   在每个子区间中心放置点，保证投影到任一维上均匀分布。
   x(i,j) = (2·perm(i) - 1) / (2·n)
   其中 perm 为 1..n 的随机排列。

3. Monte Carlo 积分误差估计:
   对积分 I = ∫ f(x) dx，N 点 MC 估计的方差:
   Var(Î_N) = σ²_f / N
   其中 σ²_f = Var(f(X))。

4. 拉丁超立方的方差缩减:
   相比纯随机采样，LHS 的方差约为 O(N^{-2/3}) 到 O(N^{-1})，
   具体取决于目标函数的可加性结构。

5. 参数敏感性分析（Sobol 指数的一阶近似）:
   S_i ≈ Var(E[f | X_i]) / Var(f)
   通过 Latin 采样计算条件期望的方差。
"""

import numpy as np


class LatinCenterSampler:
    """
    拉丁中心采样器（源自 649_latin_center）。
    """

    @staticmethod
    def sample(dim_num, point_num):
        """
        生成 Latin Center 采样点。
        参数:
            dim_num   : int, 维度数
            point_num : int, 每维点数（总样本数 = point_num）
        返回:
            x : ndarray, shape (point_num, dim_num)
        """
        x = np.zeros((point_num, dim_num), dtype=np.float64)
        for j in range(dim_num):
            perm = np.random.permutation(point_num)
            for i in range(point_num):
                x[i, j] = (2.0 * perm[i] + 1.0) / (2.0 * point_num)
        return x

    @staticmethod
    def sample_scaled(dim_num, point_num, bounds):
        """
        在指定边界内生成 Latin Center 采样点。
        bounds : list of (low, high) tuples, length dim_num
        """
        x_unit = LatinCenterSampler.sample(dim_num, point_num)
        x_scaled = np.zeros_like(x_unit)
        for j in range(dim_num):
            low, high = bounds[j]
            x_scaled[:, j] = low + x_unit[:, j] * (high - low)
        return x_scaled


class UncertaintyQuantification:
    """
    不确定性量化工具，基于 Latin 采样的统计量估计。
    """

    def __init__(self, sampler=None):
        if sampler is None:
            sampler = LatinCenterSampler()
        self.sampler = sampler

    def estimate_mean_variance(self, func, dim, n_samples, bounds=None):
        """
        估计函数在参数空间上的均值与方差。
        参数:
            func      : callable, f(x) where x is ndarray shape (dim,)
            dim       : int, 参数维度
            n_samples : int, 样本数
            bounds    : list of tuples, 参数边界
        返回:
            mean, variance : float
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim
        samples = self.sampler.sample_scaled(dim, n_samples, bounds)
        vals = np.array([func(s) for s in samples])
        mean = np.mean(vals)
        var = np.var(vals, ddof=1)
        return mean, var

    def estimate_sensitivity_indices(self, func, dim, n_samples, bounds=None):
        """
        一阶敏感性指数的 Monte Carlo 估计（近似 Sobol 指数）。
        使用 Latin 采样矩阵 A, B 及混合矩阵 C_i 估计:
        S_i ≈ Var(E[f | X_i]) / Var(f)

        数值实现:
        1. 生成两个独立 LHS 矩阵 A, B
        2. 构造 C_i = [B 的第 i 列替换为 A 的第 i 列]
        3. 计算 f(A), f(B), f(C_i)
        4. S_i ≈ (1/N) Σ f(B)(f(C_i) - f(A)) / Var(f)
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim

        A = self.sampler.sample_scaled(dim, n_samples, bounds)
        B = self.sampler.sample_scaled(dim, n_samples, bounds)

        fA = np.array([func(a) for a in A])
        fB = np.array([func(b) for b in B])

        var_f = np.var(np.concatenate([fA, fB]), ddof=1)
        if var_f < 1e-14:
            return np.zeros(dim)

        S1 = np.zeros(dim)
        for i in range(dim):
            C = B.copy()
            C[:, i] = A[:, i]
            fC = np.array([func(c) for c in C])
            S1[i] = np.mean(fB * (fC - fA)) / var_f

        S1 = np.clip(S1, 0.0, 1.0)
        return S1

    def monte_carlo_integral(self, func, dim, n_samples, bounds=None):
        """
        Monte Carlo 积分估计。
        I ≈ V · (1/N) Σ f(x_i)
        其中 V 为积分区域体积。
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * dim
        samples = self.sampler.sample_scaled(dim, n_samples, bounds)
        vals = np.array([func(s) for s in samples])
        volume = 1.0
        for low, high in bounds:
            volume *= (high - low)
        return volume * np.mean(vals)
