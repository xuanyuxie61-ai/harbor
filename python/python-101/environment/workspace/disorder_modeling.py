"""
disorder_modeling.py
====================
光子晶体制造缺陷与无序模型

融合原项目:
  - 1021_rejection_sample : 拒绝采样 (Chebyshev/CVT 采样)
  - 538_histogram_data_2d_sample : 二维离散 CDF 采样

本模块实现光子晶体制造过程中随机缺陷的统计建模，包括:
  1. 空气孔位置偏离 (positional disorder)
  2. 空气孔半径涨落 (size disorder)
  3. 介电常数不均匀性 (dielectric disorder)
  4. 表面粗糙度模型
"""

import numpy as np


# =============================================================================
# 基于 1021_rejection_sample 的拒绝采样
# =============================================================================

def chebyshev2_rejection_sample(n):
    """
    Chebyshev 型分布的拒绝采样 —— 基于 chebyshev2_sample.m
    
    PDF:
        ρ(x) = (2/π) √(1-x²),  x ∈ [-1, 1]
    
    该分布描述光子晶体孔径随机涨落的边缘效应。
    
    Parameters
    ----------
    n : int
        采样点数
    
    Returns
    -------
    samples : ndarray, shape (n,)
        采样结果
    n_trials : int
        总尝试次数
    """
    if n < 1:
        raise ValueError("采样数必须 >= 1")
    
    pdfmax = 2.0 / np.pi
    samples = np.zeros(n)
    trials = 0
    i = 0
    
    while i < n:
        trials += 1
        x = -1.0 + 2.0 * np.random.rand()
        y = pdfmax * np.random.rand()
        z = (2.0 / np.pi) * np.sqrt(max(1.0 - x ** 2, 0.0))
        
        if y <= z:
            samples[i] = x
            i += 1
    
    return samples, trials


def cvt_1d_rejection_sample(n):
    """
    一维 CVT 密度拒绝采样 —— 基于 cvt_1d_sample.m
    
    PDF:
        ρ(x) ∝ 1/√(1-x²)^(1/6),  x ∈ [-1, 1]
    
    用于模拟 Voronoi 单元尺度上的介电常数涨落。
    
    Parameters
    ----------
    n : int
        采样点数
    
    Returns
    -------
    samples : ndarray, shape (n,)
        采样结果
    """
    if n < 1:
        raise ValueError("采样数必须 >= 1")
    
    pdfmax = 1.0 / np.sqrt(np.pi)
    samples = np.zeros(n)
    i = 0
    
    while i < n:
        x = np.random.rand()  # [0, 1]
        y = pdfmax * np.random.rand()
        z = 1.0 / np.sqrt(np.pi) / (np.sqrt(max(1.0 - x ** 2, 1e-12)) ** (1.0 / 6.0))
        
        if y <= z:
            samples[i] = x
            i += 1
    
    return samples


def gaussian_rejection_sample(n, mu, sigma):
    """
    高斯分布的拒绝采样实现
    
    采用 Box-Muller 变换与拒绝采样结合，用于生成
    孔位置偏离的高斯随机位移。
    
    PDF:
        ρ(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
    
    Parameters
    ----------
    n : int
        采样点数
    mu : float
        均值
    sigma : float
        标准差
    
    Returns
    -------
    samples : ndarray, shape (n,)
        高斯随机样本
    """
    if sigma <= 0:
        raise ValueError("标准差必须为正")
    
    # 使用 Box-Muller 变换 (比拒绝采样更高效)
    samples = np.zeros(n)
    for i in range(0, n, 2):
        u1 = np.random.rand()
        u2 = np.random.rand()
        r = sigma * np.sqrt(-2.0 * np.log(max(u1, 1e-18)))
        theta = 2.0 * np.pi * u2
        samples[i] = mu + r * np.cos(theta)
        if i + 1 < n:
            samples[i + 1] = mu + r * np.sin(theta)
    
    return samples


# =============================================================================
# 光子晶体无序模型
# =============================================================================

def positional_disorder(eps_r, x, y, a, disorder_strength, n_disorder_samples=1):
    """
    空气孔位置偏离无序模型
    
    每个孔的位置从理想位置 (i·a, j·a) 发生随机偏移:
        r' = r + δr
    
    其中 δr 服从高斯分布 N(0, σ²)，σ = disorder_strength · a。
    
    安德森局域化长度估计:
        ξ_loc ≈ a · (Δω/ω)² / (δr/a)²
    
    Parameters
    ----------
    eps_r : ndarray
        理想介电常数分布
    x, y : ndarray
        坐标网格
    a : float
        晶格常数
    disorder_strength : float
        相对位置涨落幅度 (σ/a)
    n_disorder_samples : int
        无序样本数
    
    Returns
    -------
    eps_disordered : list of ndarray
        无序样本列表
    """
    if disorder_strength < 0:
        raise ValueError("无序强度必须非负")
    if disorder_strength < 1e-12:
        return [eps_r.copy()]
    
    nx, ny = eps_r.shape
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    eps_disordered = []
    
    for _ in range(n_disorder_samples):
        # 计算理想结构中的孔位置
        eps_threshold = (np.max(eps_r) + np.min(eps_r)) / 2.0
        
        # 生成偏移场
        dx_shift = gaussian_rejection_sample(nx * ny, 0.0, disorder_strength * a)
        dy_shift = gaussian_rejection_sample(nx * ny, 0.0, disorder_strength * a)
        
        dx_field = dx_shift[:nx * ny].reshape(nx, ny)
        dy_field = dy_shift[:nx * ny].reshape(nx, ny)
        
        # 应用偏移 (使用双线性插值近似)
        X_shifted = X + dx_field
        Y_shifted = Y + dy_field
        
        # 简化的偏移模型: 重新判断每个网格点是否在偏移后的孔内
        eps_new = np.full_like(eps_r, np.max(eps_r))
        
        # 找到所有孔中心
        hole_centers = []
        for i in range(nx):
            for j in range(ny):
                if eps_r[i, j] < eps_threshold:
                    hole_centers.append([X[i, j], Y[i, j]])
        
        if len(hole_centers) == 0:
            eps_disordered.append(eps_r.copy())
            continue
        
        # 计算平均孔半径
        r_hole_est = a * 0.3  # 估计值
        
        # 对每个孔应用随机偏移
        for hc in hole_centers:
            dx = np.random.normal(0, disorder_strength * a)
            dy = np.random.normal(0, disorder_strength * a)
            hc_new = [hc[0] + dx, hc[1] + dy]
            dist = np.sqrt((X - hc_new[0]) ** 2 + (Y - hc_new[1]) ** 2)
            eps_new[dist < r_hole_est] = np.min(eps_r)
        
        eps_disordered.append(eps_new)
    
    return eps_disordered


def size_disorder(eps_r, x, y, a, r_hole, relative_variation, n_samples=1):
    """
    空气孔半径涨落无序模型
    
    每个孔的半径发生随机变化:
        r'_hole = r_hole · (1 + δ)
    
    其中 δ 服从截断高斯分布，保证 r'_hole > 0。
    
    带隙宽度变化:
        Δω/ω ≈ Δω₀/ω₀ · (1 - α·⟨δ²⟩)
    
    Parameters
    ----------
    eps_r : ndarray
        理想介电常数分布
    x, y : ndarray
        坐标网格
    a : float
        晶格常数
    r_hole : float
        理想孔半径
    relative_variation : float
        半径相对涨落标准差
    n_samples : int
        样本数
    
    Returns
    -------
    eps_samples : list of ndarray
        无序样本
    """
    if relative_variation < 0 or r_hole <= 0:
        raise ValueError("参数超出允许范围")
    
    nx, ny = eps_r.shape
    X, Y = np.meshgrid(x, y, indexing='ij')
    eps_threshold = (np.max(eps_r) + np.min(eps_r)) / 2.0
    
    # 找到孔中心
    hole_centers = []
    for i in range(nx):
        for j in range(ny):
            if eps_r[i, j] < eps_threshold:
                hole_centers.append([X[i, j], Y[i, j]])
    
    eps_samples = []
    for _ in range(n_samples):
        eps_new = np.full_like(eps_r, np.max(eps_r))
        
        for hc in hole_centers:
            # 截断高斯半径
            delta = np.random.normal(0, relative_variation)
            delta = max(delta, -0.9)  # 保证半径为正
            r_new = r_hole * (1.0 + delta)
            r_new = max(r_new, 1e-9)
            
            dist = np.sqrt((X - hc[0]) ** 2 + (Y - hc[1]) ** 2)
            eps_new[dist < r_new] = np.min(eps_r)
        
        eps_samples.append(eps_new)
    
    return eps_samples


def dielectric_disorder(eps_r, correlation_length, sigma_eps, n_samples=1):
    """
    介电常数空间相关涨落模型
    
    采用高斯随机场模型:
        ε(r) = ε₀(r) + δε(r)
    
    其中 δε(r) 的相关函数:
        ⟨δε(r) δε(r')⟩ = σ²_ε exp(-|r-r'|²/(2L_c²))
    
    L_c 为关联长度。
    
    Parameters
    ----------
    eps_r : ndarray
        理想介电常数分布
    correlation_length : float
        关联长度 (以网格点数为单位)
    sigma_eps : float
        介电常数涨落标准差
    n_samples : int
        样本数
    
    Returns
    -------
    eps_samples : list of ndarray
        无序介电常数样本
    """
    if sigma_eps < 0 or correlation_length < 0:
        raise ValueError("参数必须非负")
    
    nx, ny = eps_r.shape
    eps_samples = []
    
    for _ in range(n_samples):
        # 生成白噪声
        white_noise = np.random.randn(nx, ny)
        
        if correlation_length > 0.5:
            # 高斯滤波生成相关噪声
            from scipy.ndimage import gaussian_filter
            kernel_sigma = correlation_length
            correlated_noise = gaussian_filter(white_noise, sigma=kernel_sigma)
            # 归一化到目标方差
            current_std = np.std(correlated_noise)
            if current_std > 1e-12:
                correlated_noise *= sigma_eps / current_std
        else:
            correlated_noise = white_noise * sigma_eps
        
        eps_new = eps_r + correlated_noise
        # 保证介电常数为正
        eps_new = np.maximum(eps_new, 1.0)
        
        eps_samples.append(eps_new)
    
    return eps_samples


# =============================================================================
# 2D 离散 PDF 缺陷采样
# =============================================================================

def defect_histogram_sampling(nx, ny, defect_density, defect_size_pdf, n_defects):
    """
    基于二维直方图的缺陷位置-尺寸联合采样
    
    将缺陷位置和尺寸建模为二维联合 PDF，通过 CDF 反演采样。
    
    Parameters
    ----------
    nx, ny : int
        空间网格维度
    defect_density : ndarray, shape (nx, ny)
        缺陷概率密度 (位置相关)
    defect_size_pdf : ndarray, shape (nx, ny)
        缺陷尺寸分布
    n_defects : int
        采样缺陷数
    
    Returns
    -------
    defects : list of dict
        缺陷列表，每个缺陷包含位置和尺寸
    """
    if np.any(defect_density < 0):
        raise ValueError("缺陷密度必须非负")
    
    # 联合 PDF: P(x, y, size)
    joint_pdf = defect_density * defect_size_pdf
    joint_pdf = np.maximum(joint_pdf, 0.0)
    
    # 展平为 1D 并构建 CDF
    pdf_flat = joint_pdf.flatten()
    if np.sum(pdf_flat) < 1e-18:
        return []
    
    pdf_flat /= np.sum(pdf_flat)
    cdf = np.cumsum(pdf_flat)
    
    defects = []
    for _ in range(n_defects):
        u = np.random.rand()
        idx = np.searchsorted(cdf, u)
        idx = min(idx, nx * ny - 1)
        
        i = idx % nx
        j = idx // nx
        
        # 尺寸从局部分布采样
        local_size = np.random.exponential(scale=defect_size_pdf[i, j])
        
        defects.append({
            'x_index': i,
            'y_index': j,
            'size': max(local_size, 1e-9)
        })
    
    return defects
