#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scattering_model.py
水声传播抛物方程模型 — 体积散射、混响与随机路径统计

本模块整合散射模型与随机路径统计，来源于：
- 113_box_distance（3D 盒子内随机点距离统计 → 散射体间距分布）
- 498_hammersley（准随机序列 → QMC 散射积分）
- 942_quad_parfor（并行梯形积分 → 深度/角度积分）

核心物理公式：
1. 体积散射强度（Volume Scattering Strength）：
   S_v = 10·log₁₀(σ_v)   (dB)
   其中 σ_v 为单位体积的散射截面。
   典型海洋微层模型（Clay & Medwin）：
   σ_v(z) = σ₀ · exp(−z/z₀) · [1 + α·sin(2πz/Λ)]
   其中 z₀ 为表面混合层深度，Λ 为内波波长尺度，α 为调制幅度。

2. 双向散射截面（Bistatic Scattering Cross Section）：
   σ_bs(θ_i, θ_s) = σ_v · |f(θ_i, θ_s)|²
   其中 f(θ_i, θ_s) 为散射振幅函数。
   对于 Rayleigh 散射（粒子尺寸 ≪ 波长）：
   f(θ) ∝ k²·a³·(γ_κ − cosθ·γ_ρ) / (1 + cosθ)
   其中 a 为散射体等效半径，γ_κ = ΔK/K，γ_ρ = Δρ/ρ。

3. 混响强度（Reverberation Level）：
   RL = SL − 2·TL + S_v + 10·log₁₀(V)
   其中 V 为散射体积：
   V = c·τ·R²·Δθ·Δφ / 2
   τ 为脉冲长度，R 为距离，Δθ 和 Δφ 为波束立体角。

4. 随机路径距离统计（来自 box_distance）：
   在三维散射体积 [0,a]×[0,b]×[0,c] 内，
   两点间距离 D 的均值 μ_D 的精确公式（Philip, KTH）：
   μ_D = [复杂有理组合 + asinh 项 + asin 项] / (a·b·c)
   此处采用 Monte Carlo 估计：
   μ_D ≈ (1/N) Σ_n ||X_n − Y_n||，X_n,Y_n ∼ U(box)。

5. Hammersley QMC 散射积分：
   计算散射场积分 I = ∫∫∫ σ_v(z)·G(r,z|r',z')·u(r',z') dr'dz'dφ
   使用准随机序列采样 (r',z',φ)，收敛速度 O(1/N) 优于 MC 的 O(1/√N)。

6. 空间相关系数：
   C(Δr, Δz) = ⟨u(r,z)·u*(r+Δr, z+Δz)⟩ / √⟨|u|²⟩⟨|u|²⟩
   利用 Hammersley 采样在统计系综上平均。
"""

import numpy as np
from special_functions import alnorm


# =============================================================================
# 体积散射模型
# =============================================================================

class VolumeScatteringModel:
    """
    海洋体积散射模型，包含深度依赖的散射强度剖面。
    """

    def __init__(self, sigma0=1e-6, z0=50.0, alpha=0.3, Lambda=100.0,
                 particle_radius=0.01, gamma_kappa=0.1, gamma_rho=0.05):
        self.sigma0 = sigma0
        self.z0 = z0
        self.alpha = alpha
        self.Lambda = Lambda
        self.particle_radius = particle_radius
        self.gamma_kappa = gamma_kappa
        self.gamma_rho = gamma_rho

    def scattering_strength_linear(self, z):
        """
        深度依赖的体积散射截面（线性单位，m⁻¹）。
        σ_v(z) = σ₀ · exp(−z/z₀) · [1 + α·sin(2πz/Λ)]
        """
        z = np.asarray(z, dtype=np.float64)
        base = self.sigma0 * np.exp(-z / self.z0)
        modulation = 1.0 + self.alpha * np.sin(2.0 * np.pi * z / self.Lambda)
        return base * modulation

    def scattering_strength_db(self, z):
        """转换为 dB 单位。"""
        sigma = self.scattering_strength_linear(z)
        sigma = np.maximum(sigma, 1e-20)
        return 10.0 * np.log10(sigma)

    def rayleigh_scattering_amplitude(self, k, theta):
        """
        Rayleigh 散射振幅（粒子尺寸 ≪ 波长）：
        f(θ) = k²·a³ · (γ_κ − γ_ρ·cosθ)
        """
        theta = np.asarray(theta, dtype=np.float64)
        return (k ** 2) * (self.particle_radius ** 3) * \
               (self.gamma_kappa - self.gamma_rho * np.cos(theta))

    def bistatic_cross_section(self, k, theta_i, theta_s):
        """
        双向散射截面。
        """
        f = self.rayleigh_scattering_amplitude(k, theta_i)
        # 简化：与散射角独立（各向同性近似）
        return np.abs(f) ** 2


# =============================================================================
# 混响计算
# =============================================================================

class ReverberationModel:
    """
    声纳混响强度计算模型。
    """

    def __init__(self, scattering_model, c_water=1500.0):
        self.scat = scattering_model
        self.c = c_water

    def scattering_volume(self, R, tau_pulse, beamwidth_az=10.0, beamwidth_el=10.0):
        """
        散射体积（m³）：
        V = (c·τ/2) · R² · ΔΩ
        ΔΩ ≈ Δθ_az · Δθ_el（弧度立体角，简化）。
        """
        domega = np.radians(beamwidth_az) * np.radians(beamwidth_el)
        return (self.c * tau_pulse / 2.0) * (R ** 2) * domega

    def reverberation_level(self, R, tau_pulse, SL_db, TL_db,
                            beamwidth_az=10.0, beamwidth_el=10.0,
                            z_scatter=100.0):
        """
        混响级（dB）：
        RL = SL − 2·TL + S_v + 10·log₁₀(V)
        """
        V = self.scattering_volume(R, tau_pulse, beamwidth_az, beamwidth_el)
        Sv_db = self.scat.scattering_strength_db(z_scatter)
        RL = SL_db - 2.0 * TL_db + Sv_db + 10.0 * np.log10(max(V, 1e-20))
        return RL


# =============================================================================
# 随机距离统计（来自 113_box_distance）
# =============================================================================

class BoxDistanceStatistics:
    """
    三维矩形盒子内随机点对距离统计。
    """

    def __init__(self, a, b, c, seed=None):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.rng = np.random.default_rng(seed)

    def sample_points(self, n):
        """在 [0,a]×[0,b]×[0,c] 内均匀采样 n 个点。"""
        x = self.rng.random((n, 3)) * [self.a, self.b, self.c]
        return x

    def mean_distance_monte_carlo(self, n_samples=50000):
        """
        Monte Carlo 估计两点间距离均值。
        μ_D = E[||X − Y||], X,Y ∼ U(box)。
        """
        X = self.sample_points(n_samples)
        Y = self.sample_points(n_samples)
        D = np.linalg.norm(X - Y, axis=1)
        return float(np.mean(D)), float(np.std(D))

    def mean_distance_exact(self):
        """
        精确均值公式（Johan Philip, KTH）。
        公式极其复杂，此处给出主要结构：
        μ = [多项式组合 + r·asinh 项 + 面角 asin 项] / (a·b·c)
        由于实现复杂，当 a=b=c 时使用简化立方体公式。
        """
        a, b, c = self.a, self.b, self.c
        # 立方体简化公式（文献近似）：
        # μ ≈ 0.6617 · L  for cube side L
        if abs(a - b) < 1e-9 and abs(b - c) < 1e-9:
            return 0.661707182 * a
        # 一般长方体使用 MC 估计
        return self.mean_distance_monte_carlo()[0]

    def distance_pdf_histogram(self, n_samples=50000, bins=100):
        """距离分布的直方图估计。"""
        X = self.sample_points(n_samples)
        Y = self.sample_points(n_samples)
        D = np.linalg.norm(X - Y, axis=1)
        hist, edges = np.histogram(D, bins=bins, density=True)
        return hist, edges


# =============================================================================
# Hammersley QMC 散射积分
# =============================================================================

from source_field import hammersley_sequence


def qmc_scattering_integral(integrand_func, bounds, n_samples=4096):
    """
    使用 Hammersley 准随机序列计算多维积分。
    参数:
        integrand_func: callable(x)，x 形状为 (dim,)
        bounds: list of (low, high) tuples
        n_samples: 样本数
    返回:
        积分估计值
    """
    dim = len(bounds)
    seq = hammersley_sequence(0, n_samples, dim, n=n_samples)
    # 映射到积分区间
    for j, (low, high) in enumerate(bounds):
        seq[:, j] = low + seq[:, j] * (high - low)
    values = np.array([integrand_func(seq[i, :]) for i in range(n_samples)])
    volume = 1.0
    for low, high in bounds:
        volume *= (high - low)
    return volume * np.mean(values)


# =============================================================================
# 空间相关函数
# =============================================================================

class SpatialCorrelation:
    """
    声场的空间相关系数计算。
    """

    def __init__(self, U, r_grid, z_grid):
        self.U = np.asarray(U, dtype=np.complex128)
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)

    def correlation_1d(self, axis='r', lag_index=1):
        """
        计算一维空间相关系数。
        axis='r': 固定深度，沿水平方向相关。
        axis='z': 固定距离，沿深度方向相关。
        """
        if axis == 'r':
            # 对每个深度计算水平相关
            nr, nz = self.U.shape
            C = np.zeros(nz, dtype=np.float64)
            for j in range(nz):
                u1 = self.U[:-lag_index, j]
                u2 = self.U[lag_index:, j]
                num = np.mean(u1 * np.conj(u2))
                den = np.sqrt(np.mean(np.abs(u1) ** 2) * np.mean(np.abs(u2) ** 2))
                if abs(den) > 1e-20:
                    C[j] = np.abs(num) / den
            return C
        else:
            nr, nz = self.U.shape
            C = np.zeros(nr, dtype=np.float64)
            for i in range(nr):
                if nz <= lag_index:
                    continue
                u1 = self.U[i, :-lag_index]
                u2 = self.U[i, lag_index:]
                num = np.mean(u1 * np.conj(u2))
                den = np.sqrt(np.mean(np.abs(u1) ** 2) * np.mean(np.abs(u2) ** 2))
                if abs(den) > 1e-20:
                    C[i] = np.abs(num) / den
            return C

    def mean_correlation_length(self, axis='r'):
        """
        估计平均相关长度（相关系数降至 1/e 的距离）。
        """
        # 简化：返回相关系数均值的特征衰减尺度
        if axis == 'r':
            C = self.correlation_1d('r', lag_index=1)
            # 使用指数拟合的近似
            if len(C) > 0 and np.mean(C) > 0:
                return -1.0 / np.log(max(np.mean(C), 1e-6))
        return 0.0
