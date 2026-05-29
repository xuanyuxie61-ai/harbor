#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
truncated_distribution.py
=========================

基于种子项目 1360_truncated_normal 的截断正态分布评分模型。

科学背景
--------
推荐系统中的评分通常是有界的（例如 1-5 星）。直接使用正态分布
N(μ, σ²) 建模会导致概率质量溢出边界。截断正态分布（Truncated Normal）
将有界区间的概率重新归一化，是更物理合理的模型。

定义:
    X ~ TN(μ, σ², a, b)
    
    概率密度函数:
        f(x) = φ((x-μ)/σ) / [σ · (Φ(α_b) - Φ(α_a))],  x ∈ [a, b]
        
    其中:
        α_a = (a - μ) / σ
        α_b = (b - μ) / σ
        φ(z) = (1/√(2π)) exp(-z²/2)   标准正态密度
        Φ(z) = ∫_{-∞}^z φ(t) dt       标准正态分布函数
        
    均值:
        E[X] = μ + σ · (φ(α_a) - φ(α_b)) / (Φ(α_b) - Φ(α_a))
        
    方差:
        Var[X] = σ² · [1 + (α_a·φ(α_a) - α_b·φ(α_b)) / (Φ(α_b) - Φ(α_a))
                       - ((φ(α_a) - φ(α_b)) / (Φ(α_b) - Φ(α_a)))²]
                       
拒绝采样:
    采样方法:
        1. 生成 U ~ U[0, 1]
        2. 计算 ξ_cdf = Φ(α_a) + U · (Φ(α_b) - Φ(α_a))
        3. ξ = Φ^{-1}(ξ_cdf)
        4. X = μ + σ · ξ
"""

import numpy as np
from scipy.stats import norm


class TruncatedNormalRatingModel:
    """
    截断正态分布评分模型，适用于有界评分预测。
    """
    
    def __init__(self, mu=3.0, sigma=1.2, a=1.0, b=5.0):
        """
        参数:
            mu    : 父正态分布均值
            sigma : 父正态分布标准差
            a, b  : 截断下界和上界
        """
        self.mu = float(mu)
        self.sigma = max(float(sigma), 1e-6)
        self.a = float(a)
        self.b = float(b)
        
        # 标准化边界
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        
        # 预计算 CDF 值
        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        
        # 归一化常数
        self.Z = self.beta_cdf - self.alpha_cdf
        if self.Z < 1e-12:
            self.Z = 1e-12
    
    def pdf(self, x):
        """
        截断正态概率密度函数。
        
        f(x) = φ((x-μ)/σ) / [σ · Z]
        """
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        density = norm.pdf(z) / (self.sigma * self.Z)
        # 截断
        density = np.where((x >= self.a) & (x <= self.b), density, 0.0)
        return density
    
    def mean(self):
        """
        截断正态均值。
        
        E[X] = μ + σ · (φ(α) - φ(β)) / Z
        """
        return self.mu + self.sigma * (self.alpha_pdf - self.beta_pdf) / self.Z
    
    def variance(self):
        """
        截断正态方差。
        
        Var[X] = σ² · [1 + (α·φ(α) - β·φ(β))/Z - ((φ(α) - φ(β))/Z)²]
        """
        term1 = 1.0
        term2 = (self.alpha * self.alpha_pdf - self.beta * self.beta_pdf) / self.Z
        term3 = ((self.alpha_pdf - self.beta_pdf) / self.Z) ** 2
        return self.sigma ** 2 * (term1 + term2 - term3)
    
    def sample(self, size=1):
        """
        截断正态拒绝采样。
        
        算法:
            1. U ~ Uniform(0, 1)
            2. xi_cdf = Φ(α) + U · (Φ(β) - Φ(α))
            3. xi = Φ^{-1}(xi_cdf)
            4. X = μ + σ · xi
            
        边界保护:
            - 结果严格截断在 [a, b]
        """
        rng = np.random.RandomState(42)
        size = int(size)
        if size <= 0:
            return np.array([])
        
        u = rng.rand(size)
        xi_cdf = self.alpha_cdf + u * (self.beta_cdf - self.alpha_cdf)
        xi = norm.ppf(xi_cdf)
        samples = self.mu + self.sigma * xi
        
        # 数值保护：严格截断
        samples = np.clip(samples, self.a, self.b)
        return samples
    
    def expected_rating(self, predicted_mean, clip=True):
        """
        将预测均值转换为截断正态期望评分。
        
        当预测均值靠近边界时，截断效应显著。
        """
        # 临时更新 mu 为预测均值
        old_mu = self.mu
        self.mu = predicted_mean
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        self.Z = max(self.beta_cdf - self.alpha_cdf, 1e-12)
        
        e = self.mean()
        
        # 恢复
        self.mu = old_mu
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        self.alpha_cdf = norm.cdf(self.alpha)
        self.beta_cdf = norm.cdf(self.beta)
        self.alpha_pdf = norm.pdf(self.alpha)
        self.beta_pdf = norm.pdf(self.beta)
        self.Z = max(self.beta_cdf - self.alpha_cdf, 1e-12)
        
        if clip:
            e = np.clip(e, self.a, self.b)
        return float(e)
