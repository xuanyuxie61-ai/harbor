"""
stochastic_field.py
===================
随机场生成与 Monte Carlo 采样, 用于随机变分不等式 (SVI).

数学背景
--------
随机变分不等式 (SVI): 求 x* ∈ K 使得
    E_ω[⟨F(x*, ω), y - x*⟩] ≥ 0,  ∀ y ∈ K

其中 ω 为随机参数, F(x, ω) = M(ω)x + q(ω) + Φ(x, ω).

本模块实现:
    1. 基于 Karhunen-Loève 展开的随机场生成
    2. 基于 Matérn 协方差的随机场采样
    3. 圆上 Monte Carlo 积分 (遍历采样)
    4. 离散 CDF 逆变换采样
    5. 高维超立方体距离统计

关键公式
--------
KL 展开:
    Z(x, ω) = Σ_{k=1}^M √λ_k · φ_k(x) · ξ_k(ω)

其中 λ_k, φ_k 为协方差算子的特征值/特征函数,
ξ_k ~ N(0,1) 独立标准正态.

Monte Carlo 估计:
    E[g(x, ω)] ≈ (1/N) Σ_{i=1}^N g(x, ω_i)

估计误差: O(1/√N)

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import eigh


class KarhunenLoeveExpansion:
    """
    Karhunen-Loève 展开: 随机场 Z(x, ω) 的低秩近似.

    Z(x, ω) ≈ Σ_{k=1}^M √λ_k · φ_k(x) · ξ_k(ω)

    其中:
        λ_1 ≥ λ_2 ≥ ... ≥ λ_M > 0 为协方差矩阵的特征值
        φ_k 为对应的特征向量 (离散化的特征函数)
        ξ_k ~ i.i.d. N(0,1)

    截断误差:
        ε_M^2 = Σ_{k=M+1}^∞ λ_k / Σ_{k=1}^∞ λ_k
    """

    def __init__(self, covariance_matrix: np.ndarray, energy_threshold: float = 0.99):
        """
        Parameters
        ----------
        covariance_matrix : ndarray (N, N)
            空间离散点的协方差矩阵
        energy_threshold : float
            能量阈值, 选择最小 M 使得 Σ_{k=1}^M λ_k / Σ λ_k ≥ threshold
        """
        N = covariance_matrix.shape[0]
        # 对称化 (数值稳定)
        C = 0.5 * (covariance_matrix + covariance_matrix.T)
        # 特征分解 (升序)
        eigvals, eigvecs = eigh(C)
        # 转为降序
        idx = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]

        # 处理负特征值 (数值误差)
        self.eigvals = np.maximum(self.eigvals, 0.0)

        # 确定截断阶数
        total_energy = np.sum(self.eigvals)
        if total_energy < 1e-30:
            self.M = 1
            self.truncation_error = 1.0
        else:
            cumulative = np.cumsum(self.eigvals) / total_energy
            self.M = int(np.searchsorted(cumulative, energy_threshold) + 1)
            self.M = min(self.M, N)
            self.truncation_error = 1.0 - cumulative[self.M - 1]

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        生成随机场样本.

        Returns
        -------
        Z : ndarray (N, n_samples)
            N 个空间点 × n_samples 个样本
        """
        if rng is None:
            rng = np.random.default_rng()
        # 标准正态随机变量
        xi = rng.standard_normal((self.M, n_samples))
        # KL 展开: Z = Φ_M · diag(√λ_M) · ξ
        sqrt_eigvals = np.sqrt(self.eigvals[:self.M])
        Phi_M = self.eigvecs[:, :self.M]
        Z = Phi_M @ np.diag(sqrt_eigvals) @ xi
        return Z

    def explained_variance_ratio(self) -> float:
        """前 M 个特征值解释的方差比例."""
        total = np.sum(self.eigvals)
        if total < 1e-30:
            return 0.0
        return float(np.sum(self.eigvals[:self.M]) / total)


class MaternCovarianceBuilder:
    """
    基于 Matérn 协方差函数构建协方差矩阵.

    Matérn 协方差:
        C(r) = σ² · (2√ν · r/ρ)^ν · K_ν(2√ν · r/ρ) / (Γ(ν) · 2^{ν-1})

    物理应用:
        - 地下水流中的渗透率随机场 (ν=0.5)
        - 材料科学中的弹性模量随机场 (ν=1.5)
        - 流体力学中的初始条件不确定性 (ν=2.5)
    """

    def __init__(self, length_scale: float = 1.0, nu: float = 2.5,
                 sigma_sq: float = 1.0):
        if length_scale <= 0:
            raise ValueError(f"相关长度必须为正, 得到 {length_scale}")
        if nu <= 0:
            raise ValueError(f"光滑性参数必须为正, 得到 {nu}")
        self.length_scale = length_scale
        self.nu = nu
        self.sigma_sq = sigma_sq

    def build_from_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        从空间位置构建协方差矩阵.

        Parameters
        ----------
        positions : ndarray (N, d)
            N 个点的空间坐标

        Returns
        -------
        C : ndarray (N, N)
            协方差矩阵
        """
        from nonlocal_operator import MaternCorrelationKernel
        kernel = MaternCorrelationKernel(
            lambda_corr=self.length_scale,
            nu=self.nu,
            sigma_sq=self.sigma_sq,
            dimension=positions.shape[1] if positions.ndim > 1 else 1,
        )
        return kernel.evaluate_matrix(positions)

    def build_from_1d_grid(self, x_grid: np.ndarray) -> np.ndarray:
        """从一维网格构建协方差矩阵."""
        N = len(x_grid)
        diff = x_grid[:, np.newaxis] - x_grid[np.newaxis, :]
        r = np.abs(diff)
        return self._evaluate_matern_1d(r)

    def _evaluate_matern_1d(self, r: np.ndarray) -> np.ndarray:
        """一维 Matérn 协方差."""
        from scipy.special import kv as besselk
        from scipy.special import gamma as gamma_fn

        rho = self.length_scale
        nu = self.nu
        scaled = 2.0 * np.sqrt(nu) * r / rho
        scaled_safe = np.maximum(scaled, 1e-14)

        bess = besselk(nu, scaled_safe)
        norm = gamma_fn(nu) * (2.0 ** (nu - 1.0))
        c = (scaled_safe ** nu) * bess / norm
        c = np.where(r < 1e-14, 1.0, c)
        return self.sigma_sq * c


class CircleMonteCarloSampler:
    """
    圆上 Monte Carlo 采样 (用于 SVI 的随机参数生成).

    单位圆上的均匀采样:
        θ_i ~ U[0, 2π),  x_i = (cos θ_i, sin θ_i)

    遍历采样 (低差异序列):
        θ_k = 2π · k · φ,  φ = (√5 - 1) / 2  (黄金角)

    积分估计:
        ∫_{S^1} f(θ) dθ / (2π) ≈ (1/N) Σ f(θ_k)

    误差: O(1/N) 随机, O(1/N²) 遍历 (对光滑被积函数)
    """

    GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0

    @staticmethod
    def random_sample(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        随机均匀采样.

        Returns
        -------
        points : ndarray (n, 2)
            圆上的 n 个采样点
        """
        if rng is None:
            rng = np.random.default_rng()
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        points = np.column_stack([np.cos(theta), np.sin(theta)])
        return points

    @staticmethod
    def ergodic_sample(n: int) -> np.ndarray:
        """
        遍历采样 (黄金角递推).

        θ_k = 2π · k · φ mod 2π
        其中 φ = (√5-1)/2 为黄金分割比.

        Returns
        -------
        points : ndarray (n, 2)
        """
        k = np.arange(n)
        theta = 2.0 * np.pi * (k * CircleMonteCarloSampler.GOLDEN_RATIO) % (2.0 * np.pi)
        points = np.column_stack([np.cos(theta), np.sin(theta)])
        return points

    @staticmethod
    def integrate_function(
        func_values: np.ndarray,
        method: str = 'mean'
    ) -> float:
        """
        在圆上积分函数.

        ∫ f dμ ≈ (1/N) Σ f(θ_k)  (均匀测度)
        """
        if method == 'mean':
            return float(np.mean(func_values))
        elif method == 'trapezoidal':
            n = len(func_values)
            h = 2.0 * np.pi / n
            return float(0.5 * h * (func_values[0] + func_values[-1]
                                     + 2.0 * np.sum(func_values[1:-1])))
        else:
            raise ValueError(f"未知积分方法: {method}")


class DiscreteCDFSampler:
    """
    离散 CDF 逆变换采样.

    给定二维离散 PDF p(i,j), 计算 CDF:
        F(i,j) = Σ_{a≤i, b≤j} p(a,b)

    逆变换采样:
        1. 生成 u ~ U[0,1]
        2. 找 (i,j) 使得 F(i-1,j-1) < u ≤ F(i,j)

    应用: 从离散分布生成随机场景 (用于 SVI 的 Monte Carlo).
    """

    def __init__(self, pdf_matrix: np.ndarray):
        """
        Parameters
        ----------
        pdf_matrix : ndarray (M1, M2)
            离散概率密度矩阵, 元素非负且总和为 1.
        """
        pdf = pdf_matrix.copy()
        pdf = np.maximum(pdf, 0.0)
        total = np.sum(pdf)
        if total < 1e-30:
            raise ValueError("PDF 矩阵总和为零")
        pdf = pdf / total
        self.pdf = pdf
        self.M1, self.M2 = pdf.shape
        # 计算 CDF
        self.cdf = np.cumsum(np.cumsum(pdf, axis=0), axis=1)

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        从离散 CDF 生成样本.

        Returns
        -------
        samples : ndarray (n_samples, 2)
            每个样本为 (i, j) 索引对
        """
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(0.0, 1.0, n_samples)
        samples = np.zeros((n_samples, 2), dtype=int)
        for k in range(n_samples):
            # 找第一个 CDF > u 的位置
            mask = self.cdf >= u[k]
            if not np.any(mask):
                samples[k] = [self.M1 - 1, self.M2 - 1]
            else:
                idx = np.argmax(mask.flatten())
                i, j = np.unravel_index(idx, (self.M1, self.M2))
                samples[k] = [i, j]
        return samples

    def expectation(self, values: np.ndarray) -> float:
        """
        计算 E[g(X)] = Σ_{i,j} g(i,j) · p(i,j).
        """
        if values.shape != self.pdf.shape:
            raise ValueError(f"values 形状 {values.shape} 与 PDF {self.pdf.shape} 不匹配")
        return float(np.sum(values * self.pdf))


class HypercubeDistanceStats:
    """
    高维超立方体中两点距离的统计量.

    在 [0,1]^m 中随机取两点 X, Y, 距离 D = ||X - Y||_2.

    理论均值 (精确):
        E[D²] = m / 6
    当 m → ∞, D / √m → 1/√6 (集中现象)

    本模块通过 Monte Carlo 估计 E[D] 和 Var[D].
    """

    @staticmethod
    def estimate(n_samples: int, dimension: int,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Monte Carlo 估计超立方体距离的均值和方差.

        Returns
        -------
        mu, var : float, float
        """
        if rng is None:
            rng = np.random.default_rng()
        X = rng.uniform(0.0, 1.0, (dimension, n_samples))
        Y = rng.uniform(0.0, 1.0, (dimension, n_samples))
        distances = np.sqrt(np.sum((X - Y)**2, axis=0))
        mu = float(np.mean(distances))
        var = float(np.var(distances, ddof=1)) if n_samples > 1 else 0.0
        return mu, var

    @staticmethod
    def theoretical_mean_squared(dimension: int) -> float:
        """E[D²] = m/6 的精确值."""
        return dimension / 6.0
