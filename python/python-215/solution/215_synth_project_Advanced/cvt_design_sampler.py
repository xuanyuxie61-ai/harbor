#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cvt_design_sampler.py — CVT 最优采样与设计空间探索

对应种子项目:
  - 246_cvt_1d_sampling: Lloyd 算法的采样版 CVT
  - 910_prime: 素数筛法 (用于 Halton 低差异序列)
  - 448_fresnel: Fresnel 积分 (用于波场设计空间的相位映射)

核心数学公式:
  Centroidal Voronoi Tessellation (CVT):
      给定生成点 {z_i}_{i=1}^k, Voronoi 区域:
          V_i = {x ∈ Ω : ||x - z_i|| ≤ ||x - z_j||, ∀j}
      CVT 条件: z_i = Centroid(V_i) = ∫_{V_i} x ρ(x) dV / ∫_{V_i} ρ(x) dV

  Lloyd 迭代 (采样版):
      1. 在 Ω 中生成 N 个随机样本
      2. 将每个样本分配到最近的生成点
      3. 更新生成点: z_i = mean({样本 x : x ∈ V_i})
      4. 重复至收敛

  能量泛函:
      E({z_i}) = Σ_{i=1}^k ∫_{V_i} ρ(x) ||x - z_i||² dx

  Halton 低差异序列 (基于素数):
      H_n^(p) = Σ_{j=0}^{∞} d_j(n) p^{-(j+1)}
      其中 n = Σ d_j(n) p^j 是 n 的 p 进制展开

  Fresnel 积分 (波场相位):
      C(x) = ∫_0^x cos(π t²/2) dt
      S(x) = ∫_0^x sin(π t²/2) dt
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 素数筛法 (来自 910)
# ---------------------------------------------------------------------------
def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    """
    Eratosthenes 筛法求 ≤ limit 的所有素数.

    算法:
        1. 创建 [2, limit] 的布尔数组, 初始全 True
        2. 从 p=2 开始, 标记 p 的所有倍数为 False
        3. 找下一个 True, 重复
        4. 直到 p² > limit

    复杂度: O(n log log n)

    Parameters
    ----------
    limit : int

    Returns
    -------
    primes : ndarray of int
    """
    if limit < 2:
        return np.array([], dtype=int)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[p]:
            is_prime[p * p:limit + 1:p] = False
    return np.where(is_prime)[0].astype(int)


def radical_inverse(n: int, base: int) -> float:
    """
    Radical inverse 函数 — Halton 序列的基础.

    φ_p(n) = Σ_{j=0}^{K} d_j · p^{-(j+1)}
    其中 n = Σ d_j · p^j (p 进制展开)

    Parameters
    ----------
    n    : int
    base : int (素数)

    Returns
    -------
    float ∈ [0, 1)
    """
    result = 0.0
    f = 1.0 / base
    i = n
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def halton_sequence(n_points: int, dimension: int) -> np.ndarray:
    """
    Halton 低差异序列 — 基于不同素数基的 radical inverse.

    H_n = (φ_{p_1}(n), φ_{p_2}(n), ..., φ_{p_d}(n))

    其中 p_i 为第 i 个素数.

    Parameters
    ----------
    n_points  : int
    dimension : int

    Returns
    -------
    points : shape (n_points, dimension)
    """
    primes = sieve_of_eratosthenes(max(100, 10 * dimension))
    if len(primes) < dimension:
        raise ValueError(f"需要 {dimension} 个素数, 但只有 {len(primes)} 个")

    points = np.zeros((n_points, dimension))
    for d in range(dimension):
        base = primes[d]
        for n in range(n_points):
            points[n, d] = radical_inverse(n + 1, base)
    return points


# ---------------------------------------------------------------------------
# Fresnel 积分 (来自 448)
# ---------------------------------------------------------------------------
def fresnel_cos(x: np.ndarray) -> np.ndarray:
    """
    Fresnel 余弦积分:
        C(x) = ∫_0^x cos(π t²/2) dt

    级数展开:
        C(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n} x^{4n+1}
                           / ((2n)! (4n+1))
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    for n in range(25):
        sign = (-1) ** n
        num = (np.pi / 2) ** (2 * n) * x ** (4 * n + 1)
        from math import factorial
        den = float(factorial(2 * n) * (4 * n + 1))
        term = sign * num / den
        result += term
        if np.all(np.abs(term) < 1e-15 * np.maximum(np.abs(result), 1e-15)):
            break
    return result


def fresnel_sin(x: np.ndarray) -> np.ndarray:
    """
    Fresnel 正弦积分:
        S(x) = ∫_0^x sin(π t²/2) dt

    级数展开:
        S(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n+1} x^{4n+3}
                           / ((2n+1)! (4n+3))
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    for n in range(25):
        sign = (-1) ** n
        from math import factorial
        num = (np.pi / 2) ** (2 * n + 1) * x ** (4 * n + 3)
        den = float(factorial(2 * n + 1) * (4 * n + 3))
        term = sign * num / den
        result += term
        if np.all(np.abs(term) < 1e-15 * np.maximum(np.abs(result), 1e-15)):
            break
    return result


def fresnel_phase(x: np.ndarray) -> np.ndarray:
    """
    Fresnel 复振幅相位:
        A(x) = √(C(x)² + S(x)²)
        φ(x) = arctan(S(x) / C(x))

    用于波场优化中的相位编码.
    """
    C = fresnel_cos(x)
    S = fresnel_sin(x)
    amplitude = np.sqrt(C ** 2 + S ** 2)
    phase = np.arctan2(S, C)
    return amplitude, phase


# ---------------------------------------------------------------------------
# CVT 采样 (来自 246)
# ---------------------------------------------------------------------------
def cvt_lloyd_sampling(n_generators: int,
                       n_samples: int = 5000,
                       n_iterations: int = 50,
                       dimension: int = 1,
                       domain: str = 'unit',
                       seed: int = 42
                       ) -> Tuple[np.ndarray, float]:
    """
    Lloyd 算法的采样版本 CVT (来自 246).

    算法:
        初始化: 用 Halton 序列生成 k 个初始生成点
        迭代:
            1. 在 Ω 内生成 N 个随机样本
            2. 将样本分配给最近的生成点 (Voronoi)
            3. 更新生成点: z_i = mean(V_i 中的样本)
            4. 计算能量: E = Σ_i Σ_{x ∈ V_i} ||x - z_i||²

    收敛: 能量单调递减.

    Parameters
    ----------
    n_generators : int  — 生成点数量
    n_samples    : int  — 每次迭代的采样数
    n_iterations : int  — Lloyd 迭代次数
    dimension    : int  — 空间维度
    domain       : str  — 'unit' = [0,1]^d
    seed         : int

    Returns
    -------
    (generators, energy_history)
    """
    rng = np.random.default_rng(seed)

    # 初始化: Halton 序列 (低差异, 比随机更均匀)
    if dimension <= 10:
        generators = halton_sequence(n_generators, dimension)
    else:
        generators = rng.uniform(0, 1, (n_generators, dimension))

    energy_history = []

    for it in range(n_iterations):
        # 1. 生成随机样本
        samples = rng.uniform(0, 1, (n_samples, dimension))

        # 2. 分配样本到最近生成点
        # distances: shape (n_samples, n_generators)
        dists = np.zeros((n_samples, n_generators))
        for g in range(n_generators):
            diff = samples - generators[g]
            dists[:, g] = np.sum(diff ** 2, axis=1)
        assignments = np.argmin(dists, axis=1)

        # 3. 计算能量
        energy = 0.0
        for g in range(n_generators):
            mask = assignments == g
            if np.sum(mask) > 0:
                diff = samples[mask] - generators[g]
                energy += np.sum(diff ** 2)
        energy_history.append(energy)

        # 4. 更新生成点
        for g in range(n_generators):
            mask = assignments == g
            if np.sum(mask) > 0:
                generators[g] = np.mean(samples[mask], axis=0)

    return generators, energy_history


# ---------------------------------------------------------------------------
# 设计空间采样器
# ---------------------------------------------------------------------------
def sample_design_space(n_designs: int,
                        n_elements: int,
                        method: str = 'halton',
                        seed: int = 42) -> np.ndarray:
    """
    在拓扑设计空间中采样候选设计.

    每个设计是 n_elements 维向量, ρ ∈ [0,1]^n.

    方法:
      - 'halton': 低差异 Halton 序列 (更均匀)
      - 'random': 均匀随机采样
      - 'cvt': CVT 采样后映射

    Returns
    -------
    designs : shape (n_designs, n_elements)
    """
    if method == 'halton':
        # 需要足够多的素数
        n_primes_needed = n_elements
        primes = sieve_of_eratosthenes(max(100, 10 * n_primes_needed))
        if len(primes) < n_elements:
            # 退化到 random
            rng = np.random.default_rng(seed)
            return rng.uniform(0.1, 0.9, (n_designs, n_elements))
        designs = np.zeros((n_designs, n_elements))
        for d in range(min(n_elements, len(primes))):
            base = primes[d]
            for n in range(n_designs):
                designs[n, d] = radical_inverse(n + 1, base)
        # 收缩到 [0.1, 0.9] 避免边界
        designs = 0.1 + 0.8 * designs
        return designs
    elif method == 'random':
        rng = np.random.default_rng(seed)
        return rng.uniform(0.1, 0.9, (n_designs, n_elements))
    elif method == 'cvt':
        # CVT 在 1D 采样后扩展到 n_elements
        gen, _ = cvt_lloyd_sampling(n_elements, n_samples=1000,
                                    n_iterations=20, dimension=1, seed=seed)
        gen = np.sort(gen.ravel())
        # 用生成点作为基础, 添加扰动生成多个设计
        rng = np.random.default_rng(seed)
        designs = np.zeros((n_designs, n_elements))
        for d in range(n_designs):
            perturbation = rng.normal(0, 0.05, n_elements)
            designs[d] = np.clip(gen + perturbation, 0.1, 0.9)
        return designs
    else:
        raise ValueError(f"未知采样方法: {method}")
