# -*- coding: utf-8 -*-
"""
band_sampling.py — 能带采样与统计
====================================
核心科学问题: 从第一性原理/紧束缚数据中采样能带结构,
构建统计分布用于输运计算.

融合种子项目:
  - 542_histogram_pdf_2d_sample: 2D 离散 CDF 采样
  - 626_knapsack_random: 组合优化采样思想
"""

import numpy as np


class DiscreteCDF2D:
    """
    二维离散 CDF 构建与采样.
    融合 542_histogram_pdf_2d_sample.

    将 (kx, ky) → E(k) 映射为离散概率分布.
    """

    def __init__(self, kx_grid, ky_grid, E_grid):
        """
        参数
        ----
        kx_grid, ky_grid : ndarray, shape (Nx, Ny)
            k 空间网格
        E_grid : ndarray, shape (Nx, Ny)
            能带能量 E(kx,ky)
        """
        self.kx = kx_grid
        self.ky = ky_grid
        self.E = E_grid
        self.Nx, self.Ny = kx_grid.shape

        # 构建 2D CDF
        self._build_cdf()

    def _build_cdf(self):
        """构建 1D 离散 CDF (展平 2D)."""
        # 概率正比于态密度 (能带平坦处态密度高)
        # 近似: p ∝ |det(Hessian(E))|^{-1/2}
        # 简化: 使用梯度大小作为权重
        dkx = self.kx[1, 0] - self.kx[0, 0] if self.Nx > 1 else 1e7
        dky = self.ky[0, 1] - self.ky[0, 0] if self.Ny > 1 else 1e7

        # 梯度
        dE_dkx = np.gradient(self.E, axis=0) / max(dkx, 1e-20)
        dE_dky = np.gradient(self.E, axis=1) / max(dky, 1e-20)
        grad_mag = np.sqrt(dE_dkx**2 + dE_dky**2)

        # 权重: 反比于梯度 (态密度 ∝ 1/|v_g|)
        weights = 1.0 / np.maximum(grad_mag, 1e-10)
        weights /= np.sum(weights)

        # 展平并构建 CDF
        self.weights_flat = weights.ravel()
        self.cdf = np.cumsum(self.weights_flat)
        self.cdf /= self.cdf[-1]

    def sample(self, n_samples, seed=42):
        """
        逆变换采样:
            u ~ Uniform(0,1)
            idx = CDF^{-1}(u)
            (kx, ky, E) = grid[idx]

        融合 542_histogram_pdf_2d_sample 的离散 CDF 采样.
        """
        rng = np.random.RandomState(seed)
        u = rng.uniform(0, 1, n_samples)

        # 查找 CDF 索引
        indices = np.searchsorted(self.cdf, u)
        indices = np.clip(indices, 0, self.Nx * self.Ny - 1)

        # 转换回 2D 索引
        idx_x = indices // self.Ny
        idx_y = indices % self.Ny

        kx_samples = self.kx.ravel()[indices]
        ky_samples = self.ky.ravel()[indices]
        E_samples = self.E.ravel()[indices]

        return kx_samples, ky_samples, E_samples


class BandStructureSampler:
    """
    能带结构采样器.
    融合 626_knapsack_random 的子集选择思想.
    """

    def __init__(self, k_points, energies):
        self.k_points = k_points
        self.energies = energies
        self.n_k = len(k_points)

    def importance_sample(self, n_samples, energy_window=None, seed=42):
        """
        重要性采样: 在指定能量窗口内密集采样.
        """
        rng = np.random.RandomState(seed)

        if energy_window is not None:
            E_min, E_max = energy_window
            mask = (self.energies >= E_min) & (self.energies <= E_max)
            k_valid = self.k_points[mask]
            E_valid = self.energies[mask]
        else:
            k_valid = self.k_points
            E_valid = self.energies

        if len(k_valid) == 0:
            return np.array([]), np.array([])

        # 均匀采样
        indices = rng.randint(0, len(k_valid), n_samples)
        return k_valid[indices], E_valid[indices]

    def stratified_sample(self, n_strata, samples_per_stratum, seed=42):
        """
        分层采样: 将能量范围分为 n_strata 层,
        每层独立采样.
        """
        rng = np.random.RandomState(seed)
        E_min, E_max = np.min(self.energies), np.max(self.energies)
        strata_edges = np.linspace(E_min, E_max, n_strata + 1)

        k_all = []
        E_all = []

        for i in range(n_strata):
            mask = ((self.energies >= strata_edges[i]) &
                    (self.energies < strata_edges[i + 1]))
            if not np.any(mask):
                continue

            k_valid = self.k_points[mask]
            E_valid = self.energies[mask]

            idx = rng.randint(0, len(k_valid), samples_per_stratum)
            k_all.extend(k_valid[idx])
            E_all.extend(E_valid[idx])

        return np.array(k_all), np.array(E_all)

    def subset_enumeration(self, n_select):
        """
        子集枚举 (融合 626_knapsack_random).
        从 n_k 个 k 点中选择 n_select 个.
        """
        if n_select >= self.n_k:
            return np.arange(self.n_k)

        # Gray code 风格枚举
        indices = np.zeros(n_select, dtype=int)
        for i in range(n_select):
            indices[i] = i * self.n_k // n_select

        return indices
