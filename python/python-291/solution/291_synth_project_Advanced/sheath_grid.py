"""
sheath_grid.py
==============
等离子体鞘层非均匀网格生成模块。

本模块实现适用于鞘层计算的非均匀网格生成方法。鞘层物理特征：
    - 壁面附近存在强烈的电势梯度（尺度约为数个德拜长度）
    - 鞘层边缘电势变化较缓（前鞘区尺度约为数个离子拉莫尔半径）
因此需要壁面附近加密的自适应网格。

核心方法（融合种子项目）：
    1. 几何递推网格（基础方法）
    2. 双曲正切聚类（Tanh clustering）
    3. Winslow 自适应网格生成（基于监控函数）
    4. 最小包围球域分解（源自 Welzl 算法）：
       用于高维参数空间的最优域分解
"""

import math
import numpy as np
from typing import Tuple, Optional, List


def geometric_grid(
    L: float,
    N: int,
    ratio: float = 1.02,
    wall_at_right: bool = True,
) -> np.ndarray:
    """
    几何递推网格
        x_{i+1} - x_i = r * (x_i - x_{i-1})
    当 r > 1 时，网格向右（壁面）逐渐加密。

    参数：
        L: 域长度
        N: 网格点数
        ratio: 相邻网格间距比 r
        wall_at_right: 若 True，壁面在 x=L 处，网格在右端加密

    返回：
        x: shape (N,) 网格坐标数组
    """
    if N < 2:
        raise ValueError("网格点数 N 必须 >= 2")
    if ratio <= 0.0:
        raise ValueError("几何比 ratio 必须 > 0")

    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, L, N)

    # 计算单位几何网格：dx_i = ratio^i * dx_0
    # 累积距离: x_i = dx_0 * sum(ratio^j, j=0..i-1)
    indices = np.arange(N)
    weights = ratio ** indices
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1] * L
    x = np.zeros(N)
    x[1:] = cumulative[:-1]
    x[-1] = L

    if not wall_at_right:
        x = L - x[::-1]

    return x


def tanh_clustering(
    L: float,
    N: int,
    beta: float = 3.0,
    x_center: Optional[float] = None,
) -> np.ndarray:
    """
    双曲正切聚类网格
        在 x_center 附近产生聚类。
        均匀映射：ξ ∈ [0, 1]
        物理映射：x(ξ) = x_c + L/2 * tanh(β*(2ξ-1)) / tanh(β)

    参数：
        L: 域长度
        N: 网格点数
        beta: 聚类强度参数 (beta > 0)
              beta → 0: 均匀网格
              beta → ∞: 强烈聚类
        x_center: 聚类中心位置，默认 L*0.75（靠近壁面）

    返回：
        x: shape (N,) 网格坐标数组
    """
    if N < 2:
        raise ValueError("N 必须 >= 2")
    if beta < 0.0:
        raise ValueError("聚类参数 beta 必须 >= 0")

    if x_center is None:
        x_center = 0.75 * L

    # 均匀参数空间
    xi = np.linspace(0.0, 1.0, N)

    if beta < 1e-10:
        return xi * L

    # 双曲正切映射
    # s ∈ [-1, 1]: s(ξ) = tanh(β*(2ξ-1)) / tanh(β)
    s = np.tanh(beta * (2.0 * xi - 1.0)) / math.tanh(beta)

    # 映射到物理空间：以 x_center 为中心，左右范围为 L/2
    x = x_center + 0.5 * L * s

    # 裁剪到 [0, L]
    x = np.clip(x, 0.0, L)
    x[0] = 0.0
    x[-1] = L

    return x


def winslow_adaptive_grid(
    L: float,
    N: int,
    monitor_func: callable,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Winslow 自适应网格生成 (变分方法)
        最小化泛函：
            I[x] = ∫₀¹ M(x(ξ)) * (∂x/∂ξ)² dξ
        对应的 Euler-Lagrange 方程：
            ∂/∂ξ (M ∂x/∂ξ) = 0

    采用迭代方法求解：
        M_i = M(x_i)  (监控函数)
        新网格：通过求解三对角系统更新

    参数：
        L: 域长度
        N: 网格点数
        monitor_func: 监控函数 M(x) > 0, 值大处网格加密
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回：
        x: shape (N,) 自适应网格坐标数组
    """
    if N < 3:
        raise ValueError("N 必须 >= 3")

    # 初始均匀网格
    x = np.linspace(0.0, L, N)

    for iteration in range(max_iter):
        # 计算监控函数值
        M = np.array([monitor_func(xi) for xi in x])
        M = np.maximum(M, 1e-10)  # 确保正值

        # 构造三对角系统
        # 方程：d/dξ (M dx/dξ) = 0
        # 离散：M_{i+1/2}(x_{i+1}-x_i)/dξ² - M_{i-1/2}(x_i-x_{i-1})/dξ² = 0
        # 其中 M_{i+1/2} = 0.5*(M_i + M_{i+1})

        x_new = x.copy()

        for i in range(1, N - 1):
            dxi = 1.0 / (N - 1)
            M_plus = 0.5 * (M[i] + M[i + 1])
            M_minus = 0.5 * (M[i] + M[i - 1])

            # 系数
            a = -M_minus  # x_{i-1}
            b = M_plus + M_minus  # x_i
            c = -M_plus  # x_{i+1}

            rhs = 0.0  # 齐次方程
            if abs(b) > 1e-30:
                x_new[i] = (-a * x[i - 1] - c * x[i + 1] + rhs) / b

        # 检查收敛
        dx_max = np.max(np.abs(x_new - x))
        x = x_new

        if dx_max < tol:
            break

    # 确保单调性
    for i in range(1, N):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-15

    x[0] = 0.0
    x[-1] = L

    return x


def welzl_domain_decomposition(
    points_2d: np.ndarray,
    n_subdomains: int = 4,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    基于 Welzl/Ritter 最小包围球算法的参数空间域分解
    （源自种子项目 1413_welzl: ApproxMinBoundSphereND）

    算法思想：
        Ritter 算法求近似最小包围球：
        1. 沿各维度找到极值点，选取距离最大的一对
        2. 以这对点的中心为初始球心，距离一半为半径
        3. 遍历所有点，若点在外则扩展球

    在鞘层问题中的应用：
        给定 (x, k) 参数空间的采样点集，
        使用最小包围球算法对参数空间进行最优域分解。

    参数：
        points_2d: shape (M, 2) 参数空间采样点 (x, k)
        n_subdomains: 分解的子域数量

    返回：
        domains: List of (center, radius, indices) 各子域信息
    """
    M, dim = points_2d.shape
    if M < 2:
        return [(np.mean(points_2d, axis=0), 0.0, list(range(M)))]

    # Ritter 近似最小包围球
    def ritter_bounding_sphere(pts):
        if len(pts) < 2:
            return pts[0] if len(pts) == 1 else np.zeros(dim), 0.0

        # 步骤1：找最远点对
        # 沿各维度找极值
        max_dist_sq = -1.0
        p1, p2 = pts[0], pts[1]

        for d in range(dim):
            idx_min = np.argmin(pts[:, d])
            idx_max = np.argmax(pts[:, d])
            dist_sq = np.sum((pts[idx_min] - pts[idx_max])**2)
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                p1, p2 = pts[idx_min], pts[idx_max]

        # 初始球
        center = 0.5 * (p1 + p2)
        radius = math.sqrt(max_dist_sq) / 2.0

        # 步骤2：扩展包含所有点
        for p in pts:
            dist = np.linalg.norm(p - center)
            if dist > radius:
                new_radius = (radius + dist) / 2.0
                shift = (dist - new_radius) / dist
                center = center + (center - p) * shift
                radius = new_radius

        return center, radius

    # 递归域分解：k-means 风格，但用 Ritter 球衡量
    def recursive_split(pts, indices, depth, max_depth):
        center, radius = ritter_bounding_sphere(pts)

        if depth >= max_depth or len(pts) <= 2:
            return [(center, radius, indices)]

        # 沿最大方差方向分割
        variances = np.var(pts, axis=0)
        split_dim = np.argmax(variances)
        median_val = np.median(pts[:, split_dim])

        left_mask = pts[:, split_dim] <= median_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return [(center, radius, indices)]

        left_pts = pts[left_mask]
        right_pts = pts[right_mask]
        left_idx = [indices[i] for i in range(len(indices)) if left_mask[i]]
        right_idx = [indices[i] for i in range(len(indices)) if right_mask[i]]

        max_depth_half = max_depth - 1
        result = []
        result.extend(recursive_split(left_pts, left_idx, depth + 1, max_depth_half))
        result.extend(recursive_split(right_pts, right_idx, depth + 1, max_depth_half))
        return result

    all_indices = list(range(M))
    max_depth = max(1, int(math.ceil(math.log2(n_subdomains))))
    domains = recursive_split(points_2d, all_indices, 0, max_depth)

    return domains


def compute_grid_metrics(x: np.ndarray) -> dict:
    """
    计算网格度量信息

    返回字典：
        dx: shape (N-1,) 网格间距
        dx_min: 最小间距
        dx_max: 最大间距
        stretch_ratio: 最大相邻间距比
        uniformity: 均匀性指标 (0=完全均匀, 1=完全非均匀)
    """
    N = len(x)
    if N < 2:
        raise ValueError("至少需要 2 个网格点")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("网格必须严格单调递增")

    dx_min = np.min(dx)
    dx_max = np.max(dx)

    # 拉伸比
    ratios = dx[1:] / dx[:-1]
    stretch_ratio = max(np.max(ratios), 1.0 / np.min(ratios)) if len(ratios) > 0 else 1.0

    # 均匀性指标
    mean_dx = np.mean(dx)
    uniformity = np.std(dx) / mean_dx if mean_dx > 1e-30 else 0.0

    return {
        'dx': dx,
        'dx_min': dx_min,
        'dx_max': dx_max,
        'stretch_ratio': stretch_ratio,
        'uniformity': uniformity,
        'N': N,
    }


def sheath_reference_grid(params) -> np.ndarray:
    """
    为鞘层问题生成参考网格

    使用基于监控函数的自适应网格：
        M(x) = 1 + α * |d²φ_ref/dx²|
    其中 φ_ref 为参考电势分布

    参数：
        params: PlasmaParams 对象

    返回：
        x: shape (N_grid,) 无量纲网格坐标
    """
    L = params.L_domain
    N = params.N_grid

    # 参考电势：指数衰减近似
    # φ_ref(x) ≈ φ_wall * exp(-(L-x)/λ)
    # 其中 λ ~ 几个德拜长度（无量纲：λ ~ 3-5）
    phi_wall = params.dimensionless_wall_potential()
    lambda_sheath = 4.0  # 典型鞘层厚度（无量纲）

    def monitor(x_val):
        # 二阶导数（近似）
        if x_val < 1e-10 or x_val > L - 1e-10:
            return 5.0
        phi = phi_wall * math.exp(-(L - x_val) / lambda_sheath)
        d2phi = phi / lambda_sheath**2
        return 1.0 + 10.0 * abs(d2phi)

    x = winslow_adaptive_grid(L, N, monitor, max_iter=50)
    return x
