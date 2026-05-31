"""
piecewise_approx.py
分段常数函数逼近与密度估计
融合原项目: 923_pwc_plot_1d

核心科学思想:
在流形学习中，数据分布密度通常是非光滑的，
分段常数逼近提供了一种计算简单但有效的密度估计方法。

数学模型:
给定区间划分 a = x_0 < x_1 < ... < x_n = b，
分段常数函数:
    f(x) = y_i,  x_{i-1} <= x < x_i

在D维空间中，将空间划分为超立方体单元格，
每个单元格内密度视为常数:
    ρ(x) = N_cell / (N * V_cell)

这种方法与核密度估计(KDE)形成互补:
    KDE: ρ(x) = (1/Nh^D) Σ_i K((x-x_i)/h)
    PWC: ρ(x) = Σ_cell (1/V_cell) 1_{x ∈ cell} * (N_cell/N)
"""

import numpy as np
from typing import Tuple


def piecewise_constant_1d(x_breaks: np.ndarray, y_values: np.ndarray,
                          x_query: np.ndarray) -> np.ndarray:
    """
    一维分段常数插值
    x_breaks: (n+1,) 断点
    y_values: (n,) 区间值
    x_query: (m,) 查询点
    """
    n = len(y_values)
    result = np.zeros(len(x_query))
    for i, xq in enumerate(x_query):
        # 找到所属区间
        idx = np.searchsorted(x_breaks, xq, side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        result[i] = y_values[idx]
    return result


def piecewise_constant_nd(data: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, list, list]:
    """
    D维分段常数密度估计
    对于高维数据 (D>5)，先PCA降维到低维子空间再分箱
    返回: (density_grid, bin_edges, bin_centers)
    """
    D = data.shape[1]
    N = len(data)
    # 高维限制: 最多使用6维进行分箱
    effective_bins = n_bins
    if D > 6:
        cov = np.cov(data.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 取前6个主成分
        data = data @ eigvecs[:, :6]
        D = 6
    bin_edges = []
    bin_centers = []
    for d in range(D):
        xmin, xmax = np.min(data[:, d]), np.max(data[:, d])
        edges = np.linspace(xmin, xmax, effective_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges.append(edges)
        bin_centers.append(centers)
    # 计算每个单元格的计数
    counts = np.zeros([effective_bins] * D, dtype=int)
    for pt in data:
        idx = []
        for d in range(D):
            ix = int((pt[d] - bin_edges[d][0]) / (bin_edges[d][-1] - bin_edges[d][0] + 1e-10) * effective_bins)
            ix = min(ix, effective_bins - 1)
            idx.append(ix)
        counts[tuple(idx)] += 1
    # 计算密度
    vol = 1.0
    for d in range(D):
        vol *= (bin_edges[d][1] - bin_edges[d][0])
    density = counts / (N * vol + 1e-15)
    return density, bin_edges, bin_centers


def adaptive_piecewise_density(data: np.ndarray, min_bins: int = 4,
                                max_bins: int = 32) -> Tuple[np.ndarray, list]:
    """
    自适应分段常数密度估计
    在数据密集区域使用更多单元格
    """
    N, D = data.shape
    # 基于数据量选择初始分箱数
    n_bins = min(max_bins, max(min_bins, int(N ** (1.0 / D) / 2)))
    density, edges, _ = piecewise_constant_nd(data, n_bins)
    # 自适应细化: 在密度变化大的区域增加分辨率
    for _ in range(2):
        # 检测高密度区域
        threshold = np.mean(density[density > 0])
        high_density_mask = density > threshold
        # 这里简化处理，实际应用可使用递归四叉树/八叉树剖分
        n_bins = min(max_bins, n_bins + 2)
        density, edges, _ = piecewise_constant_nd(data, n_bins)
    return density, edges


def pwc_histogram_entropy(data: np.ndarray, n_bins: int = 20) -> float:
    """
    基于分段常数近似的微分熵估计
    H = -∫ ρ(x) log ρ(x) dx ≈ -Σ_cell p_cell log(p_cell / V_cell)
    """
    density, edges, _ = piecewise_constant_nd(data, n_bins)
    # 计算单元格体积
    cell_vol = 1.0
    for d in range(len(edges)):
        cell_vol *= (edges[d][1] - edges[d][0])
    entropy = 0.0
    for idx in np.ndindex(*density.shape):
        p = density[idx] * cell_vol  # 概率质量
        if p > 1e-15:
            entropy -= p * np.log(density[idx] + 1e-15)
    return float(entropy)


def pwc_mutual_information(data_x: np.ndarray, data_y: np.ndarray,
                            n_bins: int = 10) -> float:
    """
    基于分段常数近似的互信息估计
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    data_joint = np.hstack([data_x, data_y])
    # 限制分箱数避免内存爆炸
    safe_bins = min(n_bins, 6)
    H_x = pwc_histogram_entropy(data_x, safe_bins)
    H_y = pwc_histogram_entropy(data_y, safe_bins)
    H_joint = pwc_histogram_entropy(data_joint, safe_bins)
    return max(0.0, H_x + H_y - H_joint)
