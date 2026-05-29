"""
stochastic_processes.py

随机过程与统计工具库

基于种子项目:
  - 1006_random_data: Brownian运动生成、方向均匀采样
  - 819_normal01_multivariate_distance: 多元正态距离统计

科学应用:
  1. Brownian运动: 为连续控制策略提供随机探索噪声:
     a_t = μ_θ(s_t) + σ·W_t, 其中 W_t 为d维Brown运动增量.
  2. 多元正态距离: 用于状态空间相似性度量与核方法:
     k(s,s') = exp(-||s-s'||^2 / (2 E[||Z||^2])), Z~N(0,I_d)
     其中 E[||Z||] 通过 Monte Carlo 估计.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Brownian 运动与随机游走
# ---------------------------------------------------------------------------

def direction_uniform_nd(d: int) -> np.ndarray:
    """
    在单位球面 S^{d-1} 上均匀采样一个方向向量.

    数学原理:
        若 Z ~ N(0, I_d), 则 U = Z / ||Z|| 在 S^{d-1} 上均匀分布.
    """
    if d < 1:
        raise ValueError("direction_uniform_nd: d must be positive")
    z = np.random.randn(d)
    norm = np.linalg.norm(z)
    if norm < 1.0e-12:
        return direction_uniform_nd(d)
    return z / norm


def brownian_motion(n: int, d: int, sigma: float = 1.0) -> np.ndarray:
    """
    生成 n 步 d 维 Brownian 运动轨迹.

    数学定义:
        W_0 = 0,
        W_{k} = W_{k-1} + σ·R_k·U_k,
        其中 R_k ~ |N(0,1)|, U_k ~ Uniform(S^{d-1})

    物理背景:
        在 Langevin 动力学中, Brownian运动描述介质分子对粒子的随机碰撞:
            m dv/dt = -γ v + ξ(t),  <ξ_i(t) ξ_j(t')> = 2γ k_B T δ_{ij} δ(t-t')
        离散化后得到上述随机游走.

    参数:
        n: 步数
        d: 空间维度
        sigma: 扩散系数

    返回:
        X: n×d 轨迹矩阵
    """
    if n < 1 or d < 1:
        raise ValueError("brownian_motion: n and d must be positive")
    X = np.zeros((n, d))
    for i in range(1, n):
        r = abs(np.random.randn())
        direction = direction_uniform_nd(d)
        X[i, :] = X[i - 1, :] + sigma * r * direction
    return X


def ornstein_uhlenbeck_process(n: int, d: int, theta: float = 0.15,
                                sigma: float = 0.2, dt: float = 0.01) -> np.ndarray:
    """
    Ornstein-Uhlenbeck 过程 —— 均值回归的随机过程.

    随机微分方程:
        dX_t = -θ X_t dt + σ dW_t

    在策略梯度中的应用 (OU noise for exploration):
        连续控制中常用 OU 过程作为相关探索噪声,
        相比独立高斯噪声, OU 噪声具有时间相关性,
        更符合物理动量系统的惯性特征.
    """
    X = np.zeros((n, d))
    X[0, :] = np.random.randn(d) * sigma / np.sqrt(2.0 * theta)
    for i in range(1, n):
        dW = np.random.randn(d) * np.sqrt(dt)
        X[i, :] = X[i - 1, :] - theta * X[i - 1, :] * dt + sigma * dW
    return X


# ---------------------------------------------------------------------------
# 多元正态距离统计
# ---------------------------------------------------------------------------

def multivariate_normal_distance_stats(m: int, n_samples: int = 10000) -> Tuple[float, float]:
    """
    估计 d 维标准正态空间中两点欧氏距离的期望与方差.

    理论值:
        若 Z_1, Z_2 ~ N(0, I_d) 独立, 则 D = ||Z_1 - Z_2||.
        D^2 ~ χ^2(d), 故 E[D^2] = d, Var(D^2) = 2d.
        E[D] = sqrt(2) Γ((d+1)/2) / Γ(d/2).

    参数:
        m: 空间维度 d
        n_samples: Monte Carlo 样本数

    返回:
        (mu, var) 距离的样本均值与方差
    """
    if m < 1:
        raise ValueError("multivariate_normal_distance_stats: m must be positive")
    t = np.zeros(n_samples)
    for i in range(n_samples):
        p = np.random.randn(m)
        q = np.random.randn(m)
        t[i] = np.linalg.norm(p - q)
    mu = float(np.mean(t))
    if n_samples > 1:
        var = float(np.var(t, ddof=1))
    else:
        var = 0.0
    return mu, var


def theoretical_chi_mean(d: int) -> float:
    """
    E[||Z||] 的理论值, Z~N(0,I_d).

    公式:
        E[||Z||] = sqrt(2) · Γ((d+1)/2) / Γ(d/2)
    """
    from math import gamma, sqrt
    return sqrt(2.0) * gamma((d + 1) / 2.0) / gamma(d / 2.0)


# ---------------------------------------------------------------------------
# 高斯核与状态相似性
# ---------------------------------------------------------------------------

def gaussian_kernel_matrix(states: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    计算状态集合的高斯核矩阵.

    核函数:
        K_{ij} = exp( -||s_i - s_j||^2 / (2 σ^2) )

    策略梯度应用:
        核矩阵用于核化优势估计 (KAE) 与核化自然梯度,
        将有限样本推广到连续状态空间.
    """
    n = states.shape[0]
    K = np.zeros((n, n))
    if sigma is None:
        # 使用中位数启发式
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(states[i] - states[j]))
        if len(dists) == 0:
            sigma = 1.0
        else:
            sigma = float(np.median(dists)) + 1.0e-8
    for i in range(n):
        for j in range(n):
            d2 = np.sum((states[i] - states[j]) ** 2)
            K[i, j] = np.exp(-d2 / (2.0 * sigma ** 2))
    return K
