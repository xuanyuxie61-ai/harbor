"""
domain_analysis.py
==================
磁畴结构识别、统计分析与熵度量模块。
融合来源：
- components_2d（2D 连通分量标记，识别磁畴团簇）
- histogram_display（多列数据直方图统计框架）
- triangle_histogram（子三角形区域分布统计）

物理背景：
    阻挫磁体在低温下通常不形成长程铁磁序，而是发展出短程关联的磁畴结构。
    通过连通分量分析可提取磁畴尺寸分布、分形维数、及构型熵：
        S_config = - Σ_α p_α log p_α
    其中 p_α 为第 α 个磁畴的体积分数。
"""

import numpy as np
from typing import Tuple, Dict, List
from utils import histogram_stats_1d, triangle_area_histogram_2d, EPS_MACHINE
from spin_lattice import connected_components_2d_spin_map


def extract_domain_statistics(
    spin_map: np.ndarray, threshold: float = 0.5
) -> Dict:
    """
    从二维自旋投影图提取磁畴统计信息。

    返回字典包含：
    - n_domains: 磁畴数量
    - domain_sizes: 各磁畴像素数
    - max_domain_size: 最大磁畴尺寸
    - mean_domain_size: 平均尺寸
    - domain_size_entropy: 磁畴尺寸熵
    - histogram: 尺寸分布直方图统计
    """
    labels = connected_components_2d_spin_map(spin_map, threshold)
    max_label = int(labels.max())
    if max_label == 0:
        return {
            "n_domains": 0,
            "domain_sizes": np.array([]),
            "max_domain_size": 0,
            "mean_domain_size": 0.0,
            "domain_size_entropy": 0.0,
            "histogram": {},
        }

    sizes = np.array([np.sum(labels == k) for k in range(1, max_label + 1)], dtype=int)
    total = np.sum(sizes)
    probs = sizes / total
    entropy = -np.sum(probs * np.log(probs + EPS_MACHINE))

    counts, edges, stats = histogram_stats_1d(sizes.astype(float), bins=min(20, max(5, max_label)))

    return {
        "n_domains": int(max_label),
        "domain_sizes": sizes,
        "max_domain_size": int(np.max(sizes)),
        "mean_domain_size": float(np.mean(sizes)),
        "domain_size_entropy": float(entropy),
        "histogram": {
            "counts": counts,
            "edges": edges,
            "stats": stats,
        },
    }


def spin_orientation_histogram(
    spins: np.ndarray, n_bins: int = 18
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    统计自旋相对于 z 轴的极角分布。
    极角 θ = arccos(S_z)，将 [0, π] 分为 n_bins 个区间。

    返回
    ----
    counts : np.ndarray
        各区间计数。
    theta_centers : np.ndarray
        区间中心角度。
    stats : dict
        均值、方差等统计量。
    """
    N = spins.shape[0]
    sz = np.clip(spins[:, 2], -1.0, 1.0)
    theta = np.arccos(sz)
    counts, edges = np.histogram(theta, bins=n_bins, range=(0.0, np.pi))
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    counts_f = counts.astype(float)
    _, _, stats = histogram_stats_1d(theta, bins=n_bins)
    return counts, theta_centers, stats


def radial_distribution_function_2d(
    positions: np.ndarray, spins: np.ndarray, max_r: float = 0.5, n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    二维自旋系统的径向关联函数 g(r)。
    计算不同距离 r 处的自旋关联 <S(0)·S(r)>。

    公式：
        g(r) = (1/N_r) Σ_{|r_i - r_j| ∈ [r, r+dr)} S_i · S_j
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be (N, 2)")
    N = positions.shape[0]
    dr = max_r / n_bins
    g = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(N):
        dx = positions[:, 0] - positions[i, 0]
        dy = positions[:, 1] - positions[i, 1]
        # 周期边界（假设在 [0,1]^2）
        dx = np.minimum(np.abs(dx), 1.0 - np.abs(dx))
        dy = np.minimum(np.abs(dy), 1.0 - np.abs(dy))
        r_vals = np.sqrt(dx * dx + dy * dy)
        for j in range(i + 1, N):
            r = r_vals[j]
            if r >= max_r or r < EPS_MACHINE:
                continue
            idx = int(r / dr)
            if idx >= n_bins:
                continue
            corr = np.dot(spins[i], spins[j])
            g[idx] += corr
            counts[idx] += 1

    mask = counts > 0
    g[mask] /= counts[mask]
    r_centers = np.linspace(0.5 * dr, max_r - 0.5 * dr, n_bins)
    return r_centers, g


def entropy_rate_from_trajectory(mz_trajectory: np.ndarray, delay: int = 1) -> float:
    """
    从磁化强度时间序列估算熵产生率（简化近似）。
    使用一阶差分的 Shannon 熵：
        S_rate ≈ - Σ p(ΔM) log p(ΔM) / τ
    """
    delta = np.diff(mz_trajectory)
    if delta.size == 0:
        return 0.0
    # 简单分箱
    dmin, dmax = delta.min(), delta.max()
    if abs(dmax - dmin) < EPS_MACHINE:
        return 0.0
    bins = 20
    counts, _ = np.histogram(delta, bins=bins)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def analyze_triangle_spin_distribution(
    spins_xy: np.ndarray, n_sub: int = 5
) -> Tuple[np.ndarray, Dict]:
    """
    在单位三角形内分析自旋分布的均匀性。
    融合来源：triangle_histogram（子三角形划分统计）。

    参数
    ----
    spins_xy : np.ndarray, shape (N, 2)
        自旋在 xy 平面的投影分量 (S_x, S_y)，需已归一化到 [0,1] 三角形内。
    """
    # 将单位圆投影映射到单位三角形：使用重心坐标归一化
    pts = spins_xy.copy()
    # 保证在第一象限且 x+y <= 1（近似）
    pts = np.abs(pts)
    s = np.sum(pts, axis=1, keepdims=True) + EPS_MACHINE
    pts = pts / s
    histo, info = triangle_area_histogram_2d(pts, n_sub)
    return histo, info
