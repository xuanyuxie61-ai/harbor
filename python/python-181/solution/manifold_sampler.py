"""
manifold_sampler.py
各向异性流形采样与几何网格生成
融合原项目: 333_ellipsoid_grid, 115_box_games

在高维数据空间中，数据分布往往呈现各向异性特征。
本项目采用椭球体网格对嵌入空间进行结构化采样，
并引入棋盘格离散化策略对局部邻域进行规则化剖分。

数学模型:
设嵌入空间为 R^D，数据流形 M ⊂ R^D。
局部采样区域由椭球方程定义:
    Σ_{d=1}^{D} ((x_d - c_d) / r_d)^2 <= 1
其中 c 为中心，r 为各轴向半轴长。
"""

import numpy as np
from typing import Tuple, Optional


def ellipsoid_grid_count(n: int, r: np.ndarray) -> int:
    """
    计算椭球体内网格点数量
    n: 最短轴上的子区间数
    r: 半轴长度 (D,)
    """
    D = len(r)
    r_min = np.min(r)
    if r_min < 1e-15:
        raise ValueError("半轴长度必须为正")
    h = 2.0 * r_min / (2.0 * n + 1.0)
    counts = np.ceil(r / r_min * n).astype(int)
    # 估算点数 (使用体积比例)
    volume_ratio = np.prod(r) / (r_min ** D)
    est_points = int(volume_ratio * (2 * n + 1) ** D)
    return est_points


def ellipsoid_grid(n: int, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    生成椭球体内均匀网格点
    n: 最短轴上的子区间数
    r: 半轴长度 (D,)
    c: 中心 (D,)
    返回: (N, D) 网格点坐标
    """
    D = len(r)
    r_min = np.min(r)
    if r_min < 1e-15:
        raise ValueError("半轴长度必须为正")
    h = 2.0 * r_min / (2.0 * n + 1.0)
    ni = np.ceil(r / r_min * n).astype(int)
    points = []
    # 仅在第一象限生成，然后通过反射对称生成全部
    ranges = [range(ni[d] + 1) for d in range(D)]
    # 使用递归或迭代生成网格点
    def gen_points(dim: int, current: list):
        if dim == D:
            x = np.array(current, dtype=np.float64)
            x_scaled = (x - c) / r
            if np.sum(x_scaled ** 2) <= 1.0 + 1e-12:
                # 通过反射生成所有对称点
                p = c + x * h
                # 反射生成 2^D 个象限
                for mask in range(1 << D):
                    q = p.copy()
                    for d in range(D):
                        if (mask >> d) & 1:
                            q[d] = 2.0 * c[d] - q[d]
                    # 只有当原始坐标非零时才产生新点，避免重复
                    valid = True
                    for d in range(D):
                        if current[d] == 0 and ((mask >> d) & 1):
                            valid = False
                            break
                    if valid:
                        points.append(q)
            return
        for i in range(ni[dim] + 1):
            current.append(i)
            gen_points(dim + 1, current)
            current.pop()
    gen_points(0, [])
    if len(points) == 0:
        return np.array([c])
    return np.array(points)


def anisotropic_metric_tensor(data: np.ndarray, center: np.ndarray,
                               bandwidth: float = 1.0) -> np.ndarray:
    """
    基于局部数据协方差结构计算各向异性度量张量
    给定局部数据点，计算黎曼度量:
        g_{ij} = Σ_{k} K(||x_k - c||/h) (x_k - c)_i (x_k - c)_j
    其中 K 为核函数 (高斯核)
    """
    D = data.shape[1]
    diff = data - center
    dist_sq = np.sum(diff ** 2, axis=1)
    # 高斯核权重
    weights = np.exp(-dist_sq / (2.0 * bandwidth ** 2))
    weights = weights / (np.sum(weights) + 1e-15)
    # 加权协方差 = 度量张量
    g = np.zeros((D, D), dtype=np.float64)
    for k in range(len(data)):
        g += weights[k] * np.outer(diff[k], diff[k])
    # 正则化确保正定性
    g += 1e-6 * np.eye(D)
    return g


def board_grid_discretize(bounds: np.ndarray, n_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    棋盘格离散化: 将 D 维矩形区域划分为规则网格
    bounds: (D, 2) 每维的 [min, max]
    n_cells: (D,) 每维的单元格数
    返回: (centers, edges)
        centers: (N, D) 单元格中心
        edges: list of (n_i+1,) 每维的边界
    """
    D = len(n_cells)
    edges = []
    for d in range(D):
        e = np.linspace(bounds[d, 0], bounds[d, 1], n_cells[d] + 1)
        edges.append(e)
    # 生成中心点网格
    centers_list = []
    def gen_centers(dim: int, current: list):
        if dim == D:
            centers_list.append(np.array(current, dtype=np.float64))
            return
        for i in range(n_cells[dim]):
            center = (edges[dim][i] + edges[dim][i + 1]) / 2.0
            current.append(center)
            gen_centers(dim + 1, current)
            current.pop()
    gen_centers(0, [])
    return np.array(centers_list), edges


def adaptive_ellipsoid_sample(data: np.ndarray, target_n_points: int = 500,
                               n_levels: int = 3) -> np.ndarray:
    """
    自适应椭球采样: 根据数据密度在不同区域采用不同采样密度
    数据密集区域使用更精细的网格
    """
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    # 特征分解确定椭球主轴
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 半轴长与特征值平方根成正比
    r = np.sqrt(np.maximum(eigvals, 1e-10))
    # 归一化
    r = r / np.max(r)
    samples = []
    for level in range(n_levels):
        n = 2 + level * 2
        # 在各级尺度下采样
        scale = 1.0 / (level + 1)
        pts = ellipsoid_grid(n, r * scale, mean)
        # 变换到数据空间
        pts = (eigvecs @ pts.T).T + mean
        samples.append(pts)
    all_samples = np.vstack(samples)
    # 如果过多则随机下采样
    if len(all_samples) > target_n_points:
        idx = np.random.choice(len(all_samples), target_n_points, replace=False)
        all_samples = all_samples[idx]
    return all_samples


def local_tangent_space(data: np.ndarray, center: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算局部切空间
    返回: (basis, eigenvalues)
        basis: (D, d) 切空间标准正交基
        eigenvalues: (D,) 特征值（反映局部曲率）
    """
    diff = data - center
    dist_sq = np.sum(diff ** 2, axis=1)
    idx = np.argsort(dist_sq)[:k]
    local_data = diff[idx]
    cov = local_data.T @ local_data / k
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx], eigvals[idx]
