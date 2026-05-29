"""
geometric_stats.py
==================
基于种子项目 1300_triangle_distance 的几何统计模块。
在三角化域上计算随机点对距离分布的统计量（均值、方差、PDF），
用于近似 Wasserstein 距离以评估物理信息 GAN 生成样本的几何保真度。

核心数学：
  1. 三角形内均匀采样（Turk's rule #2, Graphics Gems, 1990）：
       生成 r1, r2 ~ U[0,1]；若 r1+r2 > 1 则反射为 (1-r1, 1-r2)；
       重心坐标 (r1, r2, r3=1-r1-r2)；
       映射到笛卡尔坐标：P = r1·V1 + r2·V2 + r3·V3。

  2. 等边三角形内点对距离的精确 PDF（Baesel, 2014）：
       设边长为 s，外接圆半径 r = s/√3。
       距离 d 的定义域为 (0, s]。
       分段定义：
         0 < d ≤ 1.5·r：
           pdf(d) = (2d/s²)·[ (2π/3) - (√3/2)·(d/r)² ]
         1.5·r < d ≤ s：
           pdf(d) = (2d/s²)·[ 2·arcsin( s/(2d) ) - (√3/2)·(d/r)²
                              + √( (d/r)² - 9/4 ) ]

  3. Wasserstein-1 距离近似（基于 Monte Carlo 距离统计）：
       W_1(P, Q) ≈ | μ_d(P) - μ_d(Q) | + λ·| σ_d(P) - σ_d(Q) |
       其中 μ_d, σ_d 分别为域内随机点对距离分布的均值与标准差。
"""

import numpy as np


def triangle_sample(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                    n: int, seed: int = None) -> np.ndarray:
    """
    在三角形内均匀随机采样 n 个点（Turk's rule #2）。

    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (2,) or (3,)
        三角形顶点。
    n : int
        采样点数。
    seed : int, optional
        随机种子。

    Returns
    -------
    pts : np.ndarray, shape (n, dim)
        采样点坐标。
    """
    rng = np.random.default_rng(seed)
    r1 = rng.random(n)
    r2 = rng.random(n)
    # 反射到标准单纯形
    mask = r1 + r2 > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    r3 = 1.0 - r1 - r2
    pts = (r1[:, None] * v1.reshape(1, -1)
           + r2[:, None] * v2.reshape(1, -1)
           + r3[:, None] * v3.reshape(1, -1))
    return pts


def triangle_distance_stats(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                            n_samples: int = 5000, seed: int = None) -> dict:
    """
    通过 Monte Carlo 估计三角形内随机点对距离的均值与方差。

    Returns
    -------
    stats : dict
        {'mean': float, 'variance': float, 'std': float}
    """
    rng = np.random.default_rng(seed)
    pts1 = triangle_sample(v1, v2, v3, n_samples, rng.integers(0, 2**31))
    pts2 = triangle_sample(v1, v2, v3, n_samples, rng.integers(0, 2**31))
    dists = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
    return {
        "mean": float(np.mean(dists)),
        "variance": float(np.var(dists, ddof=1)),
        "std": float(np.std(dists, ddof=1)),
    }


def equilateral_distance_pdf(d: np.ndarray, side: float = 1.0) -> np.ndarray:
    """
    等边三角形内随机点对距离的精确 PDF。

    Parameters
    ----------
    d : np.ndarray
        距离值数组。
    side : float
        边长 s。

    Returns
    -------
    pdf : np.ndarray
        对应 d 的概率密度值。
    """
    d = np.asarray(d, dtype=float)
    s = float(side)
    if s <= 0.0:
        raise ValueError("边长必须为正。")
    r = s / np.sqrt(3.0)
    pdf = np.zeros_like(d)
    # 区域 1：0 < d ≤ 1.5·r
    mask1 = (d > 0.0) & (d <= 1.5 * r)
    pdf[mask1] = (2.0 * d[mask1] / (s * s)) * (
        (2.0 * np.pi / 3.0) - (np.sqrt(3.0) / 2.0) * (d[mask1] / r) ** 2
    )
    # 区域 2：1.5·r < d ≤ s
    mask2 = (d > 1.5 * r) & (d <= s)
    ratio = d[mask2] / r
    term1 = 2.0 * np.arcsin(np.clip(s / (2.0 * d[mask2]), -1.0, 1.0))
    term2 = (np.sqrt(3.0) / 2.0) * ratio ** 2
    term3 = np.sqrt(np.clip(ratio ** 2 - 2.25, 0.0, None))
    pdf[mask2] = (2.0 * d[mask2] / (s * s)) * (term1 - term2 + term3)
    # 边界处理：d = 0 处密度为 0；d > s 处密度为 0
    pdf = np.where(d <= 0.0, 0.0, pdf)
    pdf = np.where(d > s, 0.0, pdf)
    return pdf


def wasserstein_approx_mc(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """
    使用 Monte Carlo 距离统计近似两个点集之间的 Wasserstein-1 距离。

    Parameters
    ----------
    samples_p, samples_q : np.ndarray, shape (N, dim)
        两组样本点。

    Returns
    -------
    w1 : float
        近似 W_1 距离。
    """
    # 计算两组样本内部的随机点对距离统计量
    n = min(samples_p.shape[0], samples_q.shape[0])
    if n < 2:
        return 0.0
    rng = np.random.default_rng()
    idx_p = rng.choice(samples_p.shape[0], size=n, replace=False)
    idx_q = rng.choice(samples_q.shape[0], size=n, replace=False)
    sp = samples_p[idx_p]
    sq = samples_q[idx_q]
    # 内部距离
    dp = np.sqrt(np.sum((sp[:-1] - sp[1:]) ** 2, axis=1))
    dq = np.sqrt(np.sum((sq[:-1] - sq[1:]) ** 2, axis=1))
    mean_p = float(np.mean(dp))
    mean_q = float(np.mean(dq))
    std_p = float(np.std(dp, ddof=1))
    std_q = float(np.std(dq, ddof=1))
    # 组合近似
    w1 = abs(mean_p - mean_q) + 0.5 * abs(std_p - std_q)
    return w1


def mesh_distance_distribution(nodes: np.ndarray, triangles: list,
                               n_samples: int = 2000, seed: int = None) -> dict:
    """
    在整个三角网格上估计随机点对距离分布（按面积加权）。

    Returns
    -------
    stats : dict
        {'mean', 'variance', 'std', 'n_pairs'}
    """
    rng = np.random.default_rng(seed)
    from triangle_quadrature import triangle_area
    areas = []
    for tri in triangles:
        areas.append(triangle_area(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]))
    areas = np.array(areas)
    total_area = np.sum(areas)
    if total_area < 1e-15:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "n_pairs": 0}

    # 按面积比例采样三角形
    probs = areas / total_area
    n_tri_samples = min(len(triangles), max(10, n_samples // 10))
    chosen = rng.choice(len(triangles), size=n_tri_samples, p=probs)

    all_pts = []
    for idx in chosen:
        tri = triangles[idx]
        pts = triangle_sample(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]], 20, rng.integers(0, 2**31))
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)

    # 随机配对
    n = all_pts.shape[0]
    if n < 2:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "n_pairs": 0}
    idx1 = rng.choice(n, size=min(n_samples, n * (n - 1) // 2), replace=False)
    idx2 = rng.choice(n, size=idx1.size, replace=False)
    # 避免自配对
    mask_same = idx1 == idx2
    if np.any(mask_same):
        idx2[mask_same] = (idx2[mask_same] + 1) % n
    dists = np.sqrt(np.sum((all_pts[idx1] - all_pts[idx2]) ** 2, axis=1))
    return {
        "mean": float(np.mean(dists)),
        "variance": float(np.var(dists, ddof=1)),
        "std": float(np.std(dists, ddof=1)),
        "n_pairs": int(dists.size),
    }
