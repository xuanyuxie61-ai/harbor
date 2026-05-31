"""
utils.py
================================================================================
通用数学工具：多维积分、正态化、简单形采样与特殊函数。

融合来源：
  - 1209_test_int_nd（N 维盒形 Monte Carlo 积分）
  - 587_imshow_numeric（数值数组归一化）
  - 1312_triangle_monte_carlo（三角形 Monte Carlo）

物理背景：
  量子退火中常需要计算高维积分，如：
    - 路径积分中的配分函数 Z = ∫ D[s(τ)] exp(-S_E[s])
    - 量子态重叠 ⟨ψ|φ⟩ = ∫ ψ*(x) φ(x) dx
    - 观测量的系综平均

  对于低维（d≤6），Monte Carlo 积分是一种鲁棒的数值方法：
      ∫_Ω f(x) dV ≈ V_Ω / N Σ_{k=1}^N f(x_k)

  误差估计（中心极限定理）：
      σ_I ≈ V_Ω / sqrt(N) * sqrt( Var[f] )
"""

import numpy as np
from math import gamma as math_gamma
from typing import Tuple, Callable


def normalize_array_to_range(arr: np.ndarray, target_min: float = 0.0,
                             target_max: float = 1.0) -> np.ndarray:
    """
    将数值数组线性映射到目标区间 [target_min, target_max]。

    来自 imshow_numeric 的数值归一化思想，用于量子态振幅归一化。
    """
    arr = np.asarray(arr, dtype=float)
    a_min = arr.min()
    a_max = arr.max()
    if np.isclose(a_min, a_max):
        return np.full_like(arr, (target_min + target_max) / 2.0)
    scaled = target_min + (arr - a_min) / (a_max - a_min) * (target_max - target_min)
    return scaled


def unit_simplex_volume(dim: int) -> float:
    """
    标准 d 维单形 {x_i ≥ 0, Σ x_i ≤ 1} 的体积：
        V_d = 1 / d!
    """
    if dim < 0:
        raise ValueError("dim must be non-negative")
    if dim == 0:
        return 1.0
    return 1.0 / math_gamma(dim + 1)


def sample_unit_simplex(dim: int, n_points: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    在单位单形内均匀采样 n_points 个点。

    算法：生成 d+1 个指数分布随机变量 E_i ~ Exp(1)，
    归一化 x_i = E_i / Σ E_j。
    """
    if dim <= 0 or n_points <= 0:
        raise ValueError("dim and n_points must be positive")
    E = rng.exponential(scale=1.0, size=(n_points, dim + 1))
    S = E.sum(axis=1, keepdims=True)
    return E[:, :dim] / S


def monte_carlo_box_integral(dim: int, n_points: int,
                              integrand: Callable,
                              box_a: np.ndarray, box_b: np.ndarray,
                              rng: np.random.Generator) -> Tuple[float, float]:
    """
    d 维盒形区域 Monte Carlo 积分。

        I ≈ V * (1/N) Σ f(x_k)
        σ ≈ V / sqrt(N) * sqrt( (1/(N-1)) Σ (f_k - f̄)^2 )

    来自 test_int_nd 的 p00_box_mc 算法。
    """
    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)
    if box_a.shape != box_b.shape:
        raise ValueError("box_a and box_b must have same shape")
    if box_a.ndim != 1 or box_a.size != dim:
        raise ValueError("box bounds must be 1D arrays of length dim")
    volume = np.prod(box_b - box_a)
    samples = rng.uniform(box_a, box_b, size=(n_points, dim))
    values = np.array([integrand(x) for x in samples], dtype=float)
    mean_val = values.mean()
    std_val = values.std(ddof=1)
    integral = volume * mean_val
    error = volume * std_val / np.sqrt(max(n_points - 1, 1))
    return float(integral), float(error)


def triangle_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算二维三角形面积（鞋带公式）：
        A = 0.5 | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) |
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    if p1.shape != (2,) or p2.shape != (2,) or p3.shape != (2,):
        raise ValueError("points must be 2D vectors")
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1]) +
        p2[0] * (p3[1] - p1[1]) +
        p3[0] * (p1[1] - p2[1])
    )
    return float(area)


def monte_carlo_triangle_integral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                   n_points: int, integrand: Callable,
                                   rng: np.random.Generator) -> Tuple[float, float]:
    """
    三角形区域上的 Monte Carlo 积分。

    来自 triangle_monte_carlo：在参考三角形 (0,0),(1,0),(0,1) 上采样，
    通过仿射变换映射到物理三角形。
    """
    area = triangle_area_2d(p1, p2, p3)
    # 在参考三角形上采样：u,v ~ uniform with u+v ≤ 1
    u = rng.random(n_points)
    v = rng.random(n_points)
    # 拒绝采样改为直接变换：u' = 1 - sqrt(1-u), v' = v * sqrt(1-u)
    # 更简单的均匀采样：令 s = sqrt(u), t = v, 若 s+t > 1 则反射
    samples = np.zeros((n_points, 2))
    for k in range(n_points):
        s = np.sqrt(rng.random())
        t = rng.random()
        if s + t > 1.0:
            s = 1.0 - s
            t = 1.0 - t
        samples[k] = [s, t]
    # 仿射变换到物理三角形
    # x = p1 + s*(p2-p1) + t*(p3-p1)
    dp2 = p2 - p1
    dp3 = p3 - p1
    vals = []
    for s, t in samples:
        pt = p1 + s * dp2 + t * dp3
        vals.append(integrand(pt))
    vals = np.array(vals, dtype=float)
    integral = area * vals.mean()
    error = area * vals.std(ddof=1) / np.sqrt(max(n_points - 1, 1))
    return float(integral), float(error)


def log_sum_exp(log_vals: np.ndarray) -> float:
    """
    数值稳定的 log-sum-exp：
        log( Σ exp(a_i) ) = a_max + log( Σ exp(a_i - a_max) )
    """
    log_vals = np.asarray(log_vals, dtype=float)
    a_max = log_vals.max()
    shifted = log_vals - a_max
    shifted = np.clip(shifted, -700, 0)
    return float(a_max + np.log(np.sum(np.exp(shifted))))


def quantum_state_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    计算两个量子态矢量之间的保真度：
        F = |⟨ψ1|ψ2⟩|^2 / (⟨ψ1|ψ1⟩ ⟨ψ2|ψ2⟩)
    """
    psi1 = np.asarray(psi1, dtype=complex)
    psi2 = np.asarray(psi2, dtype=complex)
    overlap = np.vdot(psi1, psi2)
    norm1 = np.vdot(psi1, psi1).real
    norm2 = np.vdot(psi2, psi2).real
    if norm1 <= 0 or norm2 <= 0:
        raise ValueError("Quantum states must have positive norm")
    fidelity = (abs(overlap) ** 2) / (norm1 * norm2)
    return float(np.clip(fidelity, 0.0, 1.0))


def density_matrix_purity(rho: np.ndarray) -> float:
    """
    密度矩阵的纯度 Tr(ρ²)。

    对于纯态 Tr(ρ²) = 1；对于最大混合态 Tr(ρ²) = 1/d。
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    purity = np.trace(rho @ rho).real
    return float(np.clip(purity, 0.0, 1.0))


def entanglement_entropy_singular_values(sv: np.ndarray) -> float:
    """
    由 Schmidt 分解的奇异值计算冯·诺依曼纠缠熵：
        S = - Σ_i λ_i^2 log(λ_i^2)
    """
    sv = np.asarray(sv, dtype=float)
    sv = np.clip(sv, 0.0, 1.0)
    probs = sv ** 2
    probs = probs[probs > 1e-15]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)
