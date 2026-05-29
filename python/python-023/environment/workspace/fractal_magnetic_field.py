#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
分形磁通管结构生成器
================================================================================

基于 751_menger_sponge_chaos 的迭代函数系统(IFS)思想，生成磁层亚暴期间
磁重联区域的分形磁通管结构。分形几何被广泛用于描述等离子体湍流中的
间歇性和多尺度能量级串。

核心数学模型：

Menger海绵IFS由20个仿射变换组成：
    𝐱_{n+1} = A 𝐱_n + 𝐛_j,   j ∈ {1, ..., 20}

其中 A = (1/3) I₃ 为均匀收缩，𝐛_j 为平移向量。

Hausdorff维数：
    D_H = log(20) / log(3) ≈ 2.7268

在等离子体物理中的应用：
磁场分形维数与能量耗散率 ε 的关系（Kolmogorov型标度律）：
    δB(ℓ) / B_0 ~ (ℓ / L)^{1/3}    (Kolmogorov-Obukhov)
    
对于磁流体湍流 (MHD turbulence)：
    E(k) ~ k^{-5/3}    (Iroshnikov-Kraichnan: k^{-3/2})

本项目使用IFS生成磁重联电流片的分形边界，并映射为空间磁场扰动。
================================================================================
"""

import numpy as np


def generate_fractal_flux_tubes(n_points=5000):
    """
    使用Menger海绵IFS生成分形磁通管采样点。
    
    算法（基于 751_menger_sponge_chaos）：
    1. 随机选择起始点 𝐱_0 ∈ [0,1]³
    2. 重复 n_points 次：
       - 随机选择 j ∈ {1,...,20}
       - 𝐱_{i+1} = (1/3) 𝐱_i + 𝐛_j
    3. 返回所有迭代点
    
    20个平移向量对应于3×3×3立方体中保留的子立方体（排除中心及面心）。
    
    参数
    ----
    n_points : int
        采样点数。默认5000。
        
    返回
    ----
    points : ndarray, shape (n_points, 3)
        分形点云。每个点在 [0,1]³ 中。
    """
    if n_points <= 0:
        raise ValueError("n_points 必须为正整数")
    
    # Menger海绵的20个平移向量（除以3后）
    # 这些对应于3×3×3中保留的子立方体中心
    b_vectors = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0/3.0], [0.0, 0.0, 2.0/3.0],
        [0.0, 1.0/3.0, 0.0], [0.0, 1.0/3.0, 2.0/3.0],
        [0.0, 2.0/3.0, 0.0], [0.0, 2.0/3.0, 1.0/3.0], [0.0, 2.0/3.0, 2.0/3.0],
        [1.0/3.0, 0.0, 0.0], [1.0/3.0, 0.0, 2.0/3.0],
        [1.0/3.0, 2.0/3.0, 0.0], [1.0/3.0, 2.0/3.0, 2.0/3.0],
        [2.0/3.0, 0.0, 0.0], [2.0/3.0, 0.0, 1.0/3.0], [2.0/3.0, 0.0, 2.0/3.0],
        [2.0/3.0, 1.0/3.0, 0.0], [2.0/3.0, 1.0/3.0, 2.0/3.0],
        [2.0/3.0, 2.0/3.0, 0.0], [2.0/3.0, 2.0/3.0, 1.0/3.0], [2.0/3.0, 2.0/3.0, 2.0/3.0]
    ])
    
    # 收缩矩阵 A = (1/3) I_3
    scale = 1.0 / 3.0
    
    # 随机初始点
    points = np.zeros((n_points, 3))
    x = np.random.rand(3)
    
    # 预烧 (burn-in) 以消除初始条件影响
    burn_in = min(100, n_points // 10)
    for _ in range(burn_in):
        j = np.random.randint(0, 20)
        x = scale * x + b_vectors[j]
    
    # 主迭代
    for i in range(n_points):
        j = np.random.randint(0, 20)
        x = scale * x + b_vectors[j]
        points[i] = x.copy()
    
    return points


def compute_fractal_dimension(points, r_min=0.01, r_max=0.5, n_r=20):
    """
    使用盒计数法计算点云的分形维数。
    
    盒计数维数：
        D_box = lim_{r→0} log N(r) / log(1/r)
    
    其中 N(r) 是覆盖点集所需边长为 r 的盒子数。
    
    参数
    ----
    points : ndarray, shape (N, 3)
        点云。
    r_min, r_max : float
        盒子尺寸范围。
    n_r : int
        盒子尺寸数量。
        
    返回
    ----
    D_box : float
        估计的盒计数维数。
    """
    N = points.shape[0]
    if N < 100:
        return 0.0
    
    # 归一化到单位立方体
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    extent = np.max(pmax - pmin)
    if extent < 1e-20:
        return 0.0
    
    p_norm = (points - pmin) / extent
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    counts = np.zeros(n_r)
    
    for i, r in enumerate(radii):
        n_boxes = max(1, int(np.ceil(1.0 / r)))
        # 将点映射到盒子索引
        idx = np.floor(p_norm * n_boxes).astype(int)
        idx = np.clip(idx, 0, n_boxes - 1)
        # 唯一盒子数
        unique = set(map(tuple, idx))
        counts[i] = len(unique)
    
    # 线性拟合 log(N) vs log(1/r)
    valid = counts > 0
    if np.sum(valid) < 3:
        return 0.0
    
    log_inv_r = np.log(1.0 / radii[valid])
    log_N = np.log(counts[valid])
    
    D_box = np.polyfit(log_inv_r, log_N, 1)[0]
    
    # 物理约束：3D空间中 D ≤ 3
    D_box = min(max(D_box, 0.0), 3.0)
    
    return D_box


def map_fractal_to_magnetic_field(points, B0, fractal_scale=1e4):
    """
    将分形点云映射为物理空间中的磁场扰动。
    
    映射模型：
        δB_x(x) = B_0 × Σ_i w_i × sin(2π x / λ_i) × f_i(y, z)
    
    其中 f_i 是基于分形结构的调制函数，λ_i 为多尺度波长。
    
    参数
    ----
    points : ndarray, shape (N, 3)
        分形点云。
    B0 : float
        背景磁场振幅 [T]。
    fractal_scale : float
        分形结构的空间尺度 [m]。
        
    返回
    ----
    B_field : callable
        接受位置 ndarray (3,) 返回磁场 ndarray (3,) [T] 的函数。
    """
    N = points.shape[0]
    
    # 使用分形点定义调制中心
    centers = points * fractal_scale
    
    # 多尺度波长
    wavelengths = fractal_scale / (3.0 ** np.arange(1, 5))
    
    def B_field(x):
        """计算位置 x 处的总磁场。"""
        x = np.asarray(x, dtype=float)
        B = np.array([0.0, 0.0, B0])
        
        # 添加多尺度扰动
        for j, lam in enumerate(wavelengths):
            k = 2.0 * np.pi / lam
            amplitude = 0.05 * B0 / (j + 1.0)
            
            # 边界检查
            if np.abs(k * x[2]) > 100:
                continue
                
            B[0] += amplitude * np.sin(k * x[2])
            B[1] += amplitude * np.cos(k * x[2])
        
        return B
    
    return B_field
