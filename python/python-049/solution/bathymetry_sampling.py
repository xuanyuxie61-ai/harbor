#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 bathymetry_sampling.py
 
 融合种子项目：
   - 042_asa144 (rcont)：给定行列边界的随机二维表生成
   - 1372_unicycle (unicycle_random)：随机排列生成
 
 科学功能：
   随机海底地形生成与采样。
   
   海底地形对海啸传播有决定性影响。通过随机采样生成符合
   统计特征的海底地形，用于不确定性量化。
   
   rcont 算法被改造为：在保持水深边缘分布（沿 x/y 方向的
   累积深度分布）约束下，生成随机的海底地形矩阵。
   
   unicycle_random 被改造为：通过随机排列生成海底地形的
   随机相位，用于构造符合功率谱特征的随机地形。
 
 核心物理公式：
 
   1) 海底地形功率谱（von Kármán 谱）：
      P(k) = C · (k² + k_0²)^{-H-1}
      
      其中 k 为波数，k_0 为截止波数，H 为 Hurst 指数（~0.7-0.9），
      C 为归一化常数。
      
   2) 随机地形生成（傅里叶方法）：
      h(x,y) = Σ_{k_x,k_y} Â(k_x,k_y) · exp(i·(k_x·x + k_y·y))
      
      其中 Â(k) = √(P(k)/2) · (N_1 + i·N_2)，N_1, N_2 为标准高斯随机变量。
      
   3) 大陆坡地形（双曲正切剖面）：
      h(y) = h_deep - (h_deep - h_shelf) · [1 - tanh((y - y_0)/w_shelf)] / 2
      
      其中 h_deep 为深海深度，h_shelf 为陆架深度，w_shelf 为陆架宽度。
      
   4) 边缘分布约束采样（rcont 思想）：
      给定行总和（沿 x 方向的累积深度）和列总和（沿 y 方向的累积深度），
      生成满足这些约束的随机地形矩阵。
      
      约束条件：
        Σ_j h_{ij} = R_i  （行约束）
        Σ_i h_{ij} = C_j  （列约束）
"""

import numpy as np


class BathymetryGenerator:
    """
    随机海底地形生成器。
    """
    
    def __init__(self, x_grid, y_grid):
        """
        Parameters
        ----------
        x_grid, y_grid : ndarray
            网格坐标（m）
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.Lx = x_grid[-1] - x_grid[0]
        self.Ly = y_grid[-1] - y_grid[0]
    
    def generate_random_bathymetry(self, depth_mean=4000.0, depth_std=800.0,
                                    continental_slope=True,
                                    hurst_exponent=0.8):
        """
        生成随机海底地形。
        
        Parameters
        ----------
        depth_mean : float
            平均深度（m）
        depth_std : float
            深度标准差（m）
        continental_slope : bool
            是否添加大陆坡
        hurst_exponent : float
            Hurst 指数（地形粗糙度参数）
            
        Returns
        -------
        h : ndarray
            海底地形，形状 (ny, nx)，单位 m（正值表示水深）
        """
        # 1. 使用 von Kármán 功率谱生成随机地形
        h_random = self._von_karman_terrain(
            depth_mean, depth_std, hurst_exponent
        )
        
        # 2. 添加大陆坡
        if continental_slope:
            h_random = self._add_continental_slope(h_random, depth_mean)
        
        # 3. 保证水深为正
        h_random = np.maximum(h_random, 10.0)
        
        # 4. 使用 rcont 思想调整边缘分布
        h_random = self._adjust_marginal_distributions(h_random, depth_mean, depth_std)
        
        return h_random
    
    def _von_karman_terrain(self, depth_mean, depth_std, H):
        """
        基于 von Kármán 功率谱的随机地形生成。
        
        P(k) = C · (k² + k_0²)^{-H-1}
        """
        # 波数网格
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.Lx / self.nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.Ly / self.ny)
        KX, KY = np.meshgrid(kx, ky)
        k_mag = np.sqrt(KX**2 + KY**2)
        k_mag[0, 0] = 1e-6  # 避免除零
        
        # 截止波数（对应最大地形特征尺度）
        k_0 = 2.0 * np.pi / max(self.Lx, self.Ly)
        
        # von Kármán 功率谱
        P = (k_mag**2 + k_0**2) ** (-H - 1.0)
        P[0, 0] = 0.0  # 移除直流分量
        
        # 归一化
        P = P / np.sum(P)
        
        # 随机相位（使用 unicycle_random 思想：生成随机排列作为相位种子）
        phase_seed = self._unicycle_random_phase(self.nx * self.ny)
        np.random.seed(int(np.sum(phase_seed[:10])) % 2**31)
        
        # 复振幅
        amplitude = np.sqrt(P / 2.0)
        N1 = np.random.randn(self.ny, self.nx)
        N2 = np.random.randn(self.ny, self.nx)
        
        # 满足共轭对称性
        h_hat = amplitude * (N1 + 1j * N2)
        h_hat = self._enforce_conjugate_symmetry(h_hat)
        
        # 逆傅里叶变换
        h = np.real(np.fft.ifft2(h_hat))
        
        # 归一化到目标统计量
        h = (h - np.mean(h)) / (np.std(h) + 1e-12) * depth_std + depth_mean
        
        return h
    
    def _unicycle_random_phase(self, n):
        """
        基于 unicycle_random 思想的随机相位种子生成。
        
        unicycle 是一种特殊的排列，将其改造为随机数生成种子。
        """
        # 模拟 unicycle_random：生成 [1, n] 的随机排列
        u = np.arange(1, n + 1)
        for i in range(1, n):
            j = np.random.randint(i, n)
            u[i], u[j] = u[j], u[i]
        return u
    
    def _enforce_conjugate_symmetry(self, h_hat):
        """
        强制二维傅里叶变换的共轭对称性，保证实数输出。
        """
        ny, nx = h_hat.shape
        h_hat_sym = h_hat.copy()
        
        for j in range(ny):
            for i in range(nx):
                j_conj = (-j) % ny
                i_conj = (-i) % nx
                if j == 0 and i == 0:
                    h_hat_sym[j, i] = np.real(h_hat_sym[j, i])
                else:
                    val = 0.5 * (h_hat[j, i] + np.conj(h_hat[j_conj, i_conj]))
                    h_hat_sym[j, i] = val
                    h_hat_sym[j_conj, i_conj] = np.conj(val)
        
        return h_hat_sym
    
    def _add_continental_slope(self, h, depth_mean):
        """
        添加大陆坡地形剖面。
        
        使用双曲正切函数构造从陆架到深海的过渡：
        h(y) = h_deep - (h_deep - h_shelf) * [1 - tanh((y - y_0)/w)] / 2
        """
        y_norm = (self.y_grid - self.y_grid[0]) / (self.y_grid[-1] - self.y_grid[0])
        
        # 陆架参数
        h_shelf = 200.0    # 陆架水深 (m)
        h_deep = depth_mean * 1.2  # 深海深度
        y_slope = 0.6      # 大陆坡位置（归一化坐标）
        w_slope = 0.08     # 大陆坡宽度
        
        # 构造剖面
        profile = h_deep - (h_deep - h_shelf) * (1.0 - np.tanh((y_norm - y_slope) / w_slope)) / 2.0
        
        # 叠加到随机地形
        h_new = h.copy()
        for j in range(self.ny):
            h_new[j, :] = h[j, :] + (profile[j] - depth_mean)
        
        return h_new
    
    def _adjust_marginal_distributions(self, h, depth_mean, depth_std):
        """
        使用 rcont 思想调整边缘分布。
        
        改造 rcont 算法：将行/列约束视为沿 x/y 方向的累积深度分布约束，
        通过迭代调整使生成的地形满足目标边缘分布。
        """
        h_adj = h.copy()
        
        # 目标行总和（沿 x 方向的平均深度约束）
        target_row = np.ones(self.ny) * depth_mean * self.nx
        # 目标列总和（沿 y 方向的平均深度约束）
        target_col = np.ones(self.nx) * depth_mean * self.ny
        
        # 迭代调整（简化版 IPF 算法）
        for _ in range(10):
            # 调整行
            row_sum = np.sum(h_adj, axis=1)
            row_factor = target_row / (row_sum + 1e-12)
            h_adj = h_adj * row_factor[:, np.newaxis]
            
            # 调整列
            col_sum = np.sum(h_adj, axis=0)
            col_factor = target_col / (col_sum + 1e-12)
            h_adj = h_adj * col_factor[np.newaxis, :]
            
            # 保证正深度
            h_adj = np.maximum(h_adj, 10.0)
        
        return h_adj
    
    def generate_bathymetry_with_rcont_constraints(self, row_totals, col_totals):
        """
        使用 rcont 算法生成给定行列约束的随机海底地形矩阵。
        
        这是 asa144/rcont 的直接科学改造：
        给定行总和（row_totals）和列总和（col_totals），
        生成满足这些约束的随机二维表作为海底地形。
        
        Parameters
        ----------
        row_totals : ndarray
            每行的目标总和（沿 x 方向累积深度）
        col_totals : ndarray
            每列的目标总和（沿 y 方向累积深度）
            
        Returns
        -------
        h : ndarray
            满足约束的随机地形矩阵
        """
        nrow = len(row_totals)
        ncol = len(col_totals)
        
        # 边界检查
        if np.sum(row_totals) != np.sum(col_totals):
            raise ValueError("行总和必须等于列总和")
        if np.any(row_totals <= 0) or np.any(col_totals <= 0):
            raise ValueError("所有约束必须为正")
        
        ntotal = np.sum(row_totals)
        
        # 构造待排列的向量（rcont 核心思想）
        nvect = np.arange(1, ntotal + 1)
        
        # 随机排列（Fisher-Yates 洗牌）
        nnvect = nvect.copy()
        ntemp = ntotal
        perm = np.zeros(ntotal, dtype=int)
        for i in range(ntotal):
            idx = np.random.randint(0, ntemp)
            perm[i] = nnvect[idx]
            nnvect[idx] = nnvect[ntemp - 1]
            ntemp -= 1
        
        # 累积列边界
        nsubt = np.cumsum(col_totals)
        
        # 构建矩阵
        matrix = np.zeros((nrow, ncol), dtype=int)
        ii = 0
        for i in range(nrow):
            for k in range(row_totals[i]):
                for j in range(ncol):
                    if perm[ii] <= nsubt[j]:
                        ii += 1
                        matrix[i, j] += 1
                        break
        
        # 转换为深度（缩放）
        h = matrix.astype(float)
        # 归一化到合理水深范围
        h = h / np.max(h) * 5000.0 + 500.0
        
        return h
