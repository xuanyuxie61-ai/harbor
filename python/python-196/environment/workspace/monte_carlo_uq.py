"""
monte_carlo_uq.py
蒙特卡洛不确定性量化模块

包含：
- 椭圆区域蒙特卡洛采样（源自 ellipse_sample, ellipse_area1, uniform_in_sphere01_map）
- 超球体蒙特卡洛积分（源自 hypersphere01_monomial_integral, hypersphere01_sample）
- 超立方体距离统计（源自 hypercube_distance_stats）
- 高维参数空间探索与方差缩减

科学背景：
热-电耦合模拟中，材料参数（导热系数k、电阻率rho、热源密度q）
存在显著不确定性。假设参数不确定性可用高维椭球置信域描述：
    (p - p0)^T Sigma^{-1} (p - p0) <= R^2

我们需要在椭球内采样参数向量，通过FEM前向模拟计算输出统计量：
    E[T_max] ≈ (1/N) sum_{i=1}^N T_max(p_i)
    Var[T_max] ≈ (1/(N-1)) sum (T_max(p_i) - E[T_max])^2
"""

import numpy as np
from utils import cholesky_factor, hypersphere_surface_area


def uniform_in_sphere01_map(dim_num, n, rng=None):
    """
    在单位球内生成均匀随机点。
    源自 uniform_in_sphere01_map。

    算法：
        1) 生成标准正态向量 z ~ N(0, I)
        2) 归一化到球面: u = z / ||z||
        3) 径向采样: r = U^{1/d}, U~Uniform(0,1)
        4) 最终点: x = r * u

    参数:
        dim_num: int, 维度
        n: int, 点数
        rng: np.random.Generator

    返回:
        x: ndarray, shape (dim_num, n)
    """
    if rng is None:
        rng = np.random.default_rng()
    exponent = 1.0 / dim_num
    x = np.zeros((dim_num, n), dtype=float)
    for j in range(n):
        z = rng.standard_normal(dim_num)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-15:
            z_norm = 1.0
        u = z / z_norm
        r = rng.random() ** exponent
        x[:, j] = r * u
    return x


def ellipse_sample(n, A, r, rng=None):
    """
    在椭圆内生成均匀随机点。
    源自 ellipse_sample。

    椭圆定义: X^T A X <= r^2, A 对称正定。
    算法:
        1) Cholesky分解 A = U^T U
        2) 在单位球内采样 Y
        3) 解 U X = Y，即 X = U^{-1} Y

    参数:
        n: int, 采样数
        A: ndarray, shape (2,2), 正定矩阵
        r: float, 半径
        rng: np.random.Generator

    返回:
        x: ndarray, shape (2, n)
    """
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2")
    U = cholesky_factor(A)  # A = U^T U, U 上三角
    y = uniform_in_sphere01_map(2, n, rng=rng) * r
    # 解 U x = y, 即 x = U^{-1} y
    x = np.linalg.solve(U, y)
    return x


def ellipse_area(A, r):
    """
    椭圆面积（源自 ellipse_area1）。

    Area = pi * r^2 / sqrt(det(A))
    """
    det_a = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det_a <= 0:
        raise ValueError("A must be positive definite")
    return np.pi * r * r / np.sqrt(det_a)


def hypersphere01_monomial_integral(dim, expon):
    """
    单位超球面上的单项式积分。
    源自 hypersphere01_monomial_integral。

    I = integral_{S^{d-1}} prod_i x_i^{expon_i} dS

    若任一 expon_i 为奇数，则 I = 0。
    否则:
        I = 2 * prod_i Gamma((expon_i+1)/2) / Gamma((d + sum expon_i)/2)
    """
    expon = np.array(expon, dtype=int)
    if np.any(expon < 0):
        raise ValueError("exponents must be nonnegative")
    if np.any(expon % 2 == 1):
        return 0.0
    from math import gamma as math_gamma
    num = 2.0
    for e in expon:
        num *= math_gamma((e + 1) / 2.0)
    den = math_gamma((dim + np.sum(expon)) / 2.0)
    return float(num / den)


def hypersphere_monte_carlo_integral(dim, n_samples, func, rng=None):
    """
    超球面上蒙特卡洛积分。

    I ≈ (Area / N) * sum func(x_i)
    """
    if rng is None:
        rng = np.random.default_rng()
    area = hypersphere_surface_area(dim)
    # 在球面上采样: 归一化标准正态向量
    samples = rng.standard_normal((dim, n_samples))
    norms = np.linalg.norm(samples, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    samples = samples / norms
    vals = np.array([func(samples[:, i]) for i in range(n_samples)])
    return float(area * np.mean(vals)), float(area * np.std(vals) / np.sqrt(max(n_samples, 1)))


def hypercube_distance_stats(dim, n_samples, rng=None):
    """
    超立方体内两随机点距离的均值与方差统计。
    源自 hypercube_distance_stats。

    在任务调度中，这用于度量高维任务特征空间中的"任务相似度"：
        d_{ij} = ||f_i - f_j||_2
    其中 f_i 是任务i的特征向量（flops, intensity, memory）。

    参数:
        dim: int, 特征维度
        n_samples: int, 采样对数

    返回:
        mu, var: float, 距离均值与方差
    """
    if rng is None:
        rng = np.random.default_rng()
    p1 = rng.random((dim, n_samples))
    p2 = rng.random((dim, n_samples))
    dists = np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
    mu = float(np.mean(dists))
    if n_samples > 1:
        var = float(np.sum((dists - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def antithetic_variates_integral(dim, n_pairs, func, rng=None):
    """
    对偶变量方差缩减技术。

    利用对称性: 若 U ~ Uniform, 则 1-U 也是 Uniform。
    I ≈ (1/(2N)) sum [f(x_i) + f(x_i')]
    其中 x_i' 是 x_i 的对偶点。
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random((dim, n_pairs))
    x1 = u
    x2 = 1.0 - u
    f1 = np.array([func(x1[:, i]) for i in range(n_pairs)])
    f2 = np.array([func(x2[:, i]) for i in range(n_pairs)])
    estimates = (f1 + f2) / 2.0
    return float(np.mean(estimates)), float(np.std(estimates) / np.sqrt(max(n_pairs, 1)))
