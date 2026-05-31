"""
cvt_reactor_discretization.py
==============================
基于 Centroidal Voronoi Tessellation (CVT) 的反应器空间离散化

基于种子项目 247_cvt_2d_lumping 融合重构。

科学背景：
---------
聚合反应器的优化设计需要在计算域内选择一组代表性节点（生成元），
使得节点分布与反应速率密度函数相匹配。CVT 通过 Lloyd 算法
迭代调整生成元位置，使其成为对应 Voronoi 单元的质心，
从而在空间上实现最优量化 (optimal quantization)。

密度函数 ρ(x) 与反应速率相关：
    ρ(x,y) = [R_p(x,y)]^γ

其中 R_p 为局部链增长速率，γ 为幂指数（通常取 2 或 4，
对应于高维最优量化理论中的密度-网格关系）。

Lloyd 算法：
    1. 计算当前生成元的 Voronoi 剖分
    2. 对每个 Voronoi 单元，计算密度加权质心：
         g_i^{new} = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx
    3. 更新生成元：g_i ← g_i^{new}
    4. 重复直至收敛

本模块采用 lumping（聚集）策略处理高密度区域，避免边界奇异性。
"""

import numpy as np
from typing import Tuple, Callable, Optional


def chebyshev_zero_density_1d(n: int) -> np.ndarray:
    """
    一维 Chebyshev 零点分布密度（基于 mu_chebyzero.m）。
    用于反应器入口边界层的节点加密。
    """
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    return x


def chebyshev_zero_density_2d(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    二维 Chebyshev 零点网格（基于 mu_2d_chebyzero.m）。
    返回 (X, Y) 网格坐标。
    """
    x = chebyshev_zero_density_1d(nx)
    y = chebyshev_zero_density_1d(ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def cvt_2d_lloyd(n_generators: int,
                 n_iterations: int,
                 n_samples: int,
                 density_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 x_min: float = -1.0,
                 x_max: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    二维 Lloyd CVT 算法（基于 cvt_2d_lumping.m 的 lumping 版本）。

    参数：
        n_generators : 生成元数量（CVT 节点数）
        n_iterations : Lloyd 迭代次数
        n_samples    : 每个方向采样点数（总样本 n_samples²）
        density_func : 密度函数 ρ(x,y)
        x_min, x_max : 计算域边界

    返回：
        g          : 最终生成元坐标 (n_generators, 2)
        energy_history : 每步能量
        motion_history : 每步平均移动距离
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    if n_generators < 3:
        raise ValueError("n_generators must be at least 3")

    # 初始化生成元
    g = x_min + (x_max - x_min) * rng.random((n_generators, 2))
    g_new = np.zeros_like(g)

    energy_history = np.zeros(n_iterations)
    motion_history = np.zeros(n_iterations)

    # 构建均匀采样网格（避免边界）
    eps_margin = 1.0e-6 * (x_max - x_min)
    s_1d = np.linspace(x_min + eps_margin, x_max - eps_margin, n_samples)
    sx_mat, sy_mat = np.meshgrid(s_1d, s_1d)
    sx_vec = sx_mat.flatten()
    sy_vec = sy_mat.flatten()

    # 密度计算与截断（防止 blow up）
    rho_mat = density_func(sx_mat, sy_mat)
    rho_mat = np.minimum(rho_mat, 10.0)
    # Lumping：密度幂次（2D 理论中为四次方）
    r_mat = rho_mat ** 4
    r_vec = r_mat.flatten()

    points = np.column_stack((sx_vec, sy_vec))

    for it in range(n_iterations):
        # 对每个采样点，找到最近的生成元（简化版，不用 Delaunay）
        # 使用广播计算距离
        # distances shape: (n_samples^2, n_generators)
        diff = points[:, np.newaxis, :] - g[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dists_sq, axis=1)

        # 计算每个生成元的质量与质心
        mass = np.zeros(n_generators)
        centroid_x = np.zeros(n_generators)
        centroid_y = np.zeros(n_generators)

        for k in range(n_generators):
            mask = (nearest == k)
            mass[k] = np.sum(r_vec[mask])
            if mass[k] > 1.0e-15:
                centroid_x[k] = np.sum(r_vec[mask] * sx_vec[mask]) / mass[k]
                centroid_y[k] = np.sum(r_vec[mask] * sy_vec[mask]) / mass[k]
            else:
                # 无样本时保持原位
                centroid_x[k] = g[k, 0]
                centroid_y[k] = g[k, 1]

        g_new[:, 0] = centroid_x
        g_new[:, 1] = centroid_y

        # 能量 = Σ r_i * |point_i - g_nearest|^2
        energy = 0.0
        for idx_pt, k in enumerate(nearest):
            energy += r_vec[idx_pt] * dists_sq[idx_pt, k]
        energy_history[it] = energy / n_samples

        # 平均移动
        motion = np.mean(np.sum((g_new - g) ** 2, axis=1))
        motion_history[it] = motion

        g = g_new.copy()

        # 边界投影
        g = np.clip(g, x_min, x_max)

    return g, energy_history, motion_history


def reaction_rate_density(x: np.ndarray, y: np.ndarray,
                          peak_x: float = 0.0,
                          peak_y: float = 0.0,
                          sigma: float = 0.5,
                          amplitude: float = 2.0) -> np.ndarray:
    """
    模拟反应器中的局部反应速率密度分布（高斯型热点）。

        ρ(x,y) = amplitude * exp( -((x-peak_x)² + (y-peak_y)²)/(2σ²) ) + 0.1
    """
    X = np.asarray(x)
    Y = np.asarray(y)
    rho = amplitude * np.exp(-((X - peak_x) ** 2 + (Y - peak_y) ** 2) / (2.0 * sigma ** 2)) + 0.1
    return rho


def optimal_reactor_nodes(n_nodes: int = 20,
                          n_iter: int = 30,
                          n_samples: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为聚合反应器计算最优空间离散节点。
    """
    g, energy, motion = cvt_2d_lloyd(
        n_generators=n_nodes,
        n_iterations=n_iter,
        n_samples=n_samples,
        density_func=lambda x, y: reaction_rate_density(x, y),
        x_min=-1.0, x_max=1.0
    )
    return g, energy, motion
