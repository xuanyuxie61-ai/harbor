#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_response.py
探测器响应模拟与击中重建模块

融合原项目:
- 353_fd1d_advection_ftcs: 一维平流-扩散方程（能量沉积传播）
- 579_image_edge: NEWS 边缘检测算子（径迹边界识别）
- 012_aperiodic_tile: 非周期平铺结构（探测器像素几何）

在BSM信号分析中用于:
- 模拟带电粒子穿过探测器敏感体积的能量沉积剖面
- 在二维击中矩阵上识别粒子径迹的边界
- 构建非周期像素排列的探测器几何模型
"""

import numpy as np
from typing import Tuple, Optional


def advection_diffusion_energy_deposit(
    nx: int = 101,
    nt: int = 1000,
    c: float = 1.0,
    diff_coeff: float = 0.001,
    x_min: float = 0.0,
    x_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟带电粒子在探测器材料中的一维能量沉积剖面。

    将不稳定的 FTCS 格式改进为稳定的 Lax-Wendroff 格式：
        u_j^{n+1} = u_j^n
            - (c dt / 2dx) (u_{j+1}^n - u_{j-1}^n)
            + (c^2 dt^2 / 2dx^2) (u_{j+1}^n - 2u_j^n + u_{j-1}^n)
            + (D dt / dx^2) (u_{j+1}^n - 2u_j^n + u_{j-1}^n)

    该方程描述带电粒子（如来自 Z' → e^+ e^- 衰变的高能电子）
    在穿过硅微条探测器时的能量沉积演化：
        ∂E/∂t = -v ∂E/∂x + D ∂²E/∂x²

    初始条件为 Bragg 峰近似的高斯型能量包：
        E(0,x) = (10x - 4)^2 (6 - 10x)^2  对于 0.4 ≤ x ≤ 0.6

    Parameters
    ----------
    nx : int
        空间网格数
    nt : int
        时间步数
    c : float
        粒子漂移速度（归一化）
    diff_coeff : float
        扩散系数 D [m^2/s]
    x_min, x_max : float
        空间范围 [m]

    Returns
    -------
    x : np.ndarray
        空间网格点
    u_final : np.ndarray
        最终时刻的能量沉积剖面
    """
    if nx < 3:
        raise ValueError("nx 必须 >= 3")
    if nt < 1:
        raise ValueError("nt 必须 >= 1")
    if c < 0.0:
        raise ValueError("漂移速度 c 必须非负")

    dx = (x_max - x_min) / (nx - 1)
    dt = 1.0 / nt

    # CFL 条件检查: 对 Lax-Wendroff + 扩散，要求
    #   σ = c dt / dx ≤ 1  且  D dt / dx^2 ≤ 0.5
    sigma = c * dt / dx
    d_factor = diff_coeff * dt / (dx ** 2)

    if sigma > 1.0:
        # 自适应调整 dt 以满足稳定性
        dt = dx / c * 0.9
        sigma = c * dt / dx
        nt = max(int(1.0 / dt) + 1, nt)
    if d_factor > 0.5:
        dt = 0.5 * dx ** 2 / diff_coeff * 0.9
        d_factor = diff_coeff * dt / (dx ** 2)
        nt = max(int(1.0 / dt) + 1, nt)

    x = np.linspace(x_min, x_max, nx)
    u = np.zeros(nx)

    # Bragg 峰初始条件（模拟最小电离粒子的能量沉积）
    mask = (0.4 <= x) & (x <= 0.6)
    u[mask] = (10.0 * x[mask] - 4.0) ** 2 * (6.0 - 10.0 * x[mask]) ** 2

    # 周期性边界条件的索引映射
    im1 = np.array([nx - 1] + list(range(nx - 1)))
    ip1 = np.array(list(range(1, nx)) + [0])

    for _ in range(nt):
        # Lax-Wendroff + 扩散项
        u_new = u.copy()
        # 对流项（中心差分 + 二阶修正）
        u_new -= sigma * 0.5 * (u[ip1] - u[im1])
        u_new += (sigma ** 2) * 0.5 * (u[ip1] - 2.0 * u + u[im1])
        # 扩散项
        u_new += d_factor * (u[ip1] - 2.0 * u + u[im1])
        u = u_new

    return x, u


def news_edge_detector(
    data: np.ndarray,
    threshold: float = 0.1,
    normalize: bool = True
) -> np.ndarray:
    """
    NEWS (North-East-West-South) 边缘检测算子。

    对于探测器二维击中图 A，每个像素 A(i,j) 的 NEWS 响应为：
        E(i,j) = |A(i-1,j) - A(i+1,j)| + |A(i,j-1) - A(i,j+1)|

    物理意义: 在径迹跟踪中，高 E 值对应粒子径迹的边界，
    低 E 值对应径迹内部或纯噪声区域。

    Parameters
    ----------
    data : np.ndarray
        二维探测器击中图，形状 (M, N)
    threshold : float
        边缘阈值（相对于最大响应的比例）
    normalize : bool
        是否将输出归一化到 [0, 1]

    Returns
    -------
    np.ndarray
        边缘响应图，形状 (M, N)
    """
    if data.ndim != 2:
        raise ValueError("输入必须是二维数组")
    m, n = data.shape
    if m < 3 or n < 3:
        # 对于小矩阵，直接返回零
        return np.zeros_like(data)

    # 扩展边界（零填充 + 复制边缘值以减小边界效应）
    b = np.zeros((m + 2, n + 2), dtype=float)
    b[1:m+1, 1:n+1] = data
    # 复制边缘
    b[0, 1:n+1] = b[1, 1:n+1]
    b[m+1, 1:n+1] = b[m, 1:n+1]
    b[1:m+1, 0] = b[1:m+1, 1]
    b[1:m+1, n+1] = b[1:m+1, n]
    # 角点取平均
    b[0, 0] = (b[0, 1] + b[1, 0]) / 2.0
    b[m+1, 0] = (b[m+1, 1] + b[m, 0]) / 2.0
    b[0, n+1] = (b[0, n] + b[1, n+1]) / 2.0
    b[m+1, n+1] = (b[m+1, n] + b[m, n+1]) / 2.0

    # NEWS 算子: 垂直梯度 + 水平梯度
    e = np.zeros((m + 2, n + 2), dtype=float)
    e[1:m+1, 1:n+1] = np.abs(-b[0:m, 1:n+1] + b[2:m+2, 1:n+1]) \
                     + np.abs(-b[1:m+1, 0:n] + b[1:m+1, 2:n+2])

    # 提取内部区域
    e = e[1:m+1, 1:n+1]

    if normalize:
        e_min = np.min(e)
        e_max = np.max(e)
        if e_max > e_min:
            e = (e - e_min) / (e_max - e_min)
        else:
            e = np.zeros_like(e)

    # 阈值处理
    e = np.where(e > threshold, e, 0.0)

    return e


def detector_hit_map(
    n_pixels: int = 64,
    noise_level: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟的 LHC 硅径迹探测器二维击中图。

    模拟 Z' → ℓ^+ ℓ^- 衰变产生的一对背对背高能轻子，
    在二维探测器平面上留下两条弯曲径迹。

    物理模型:
        - 径迹能量沉积服从朗道分布（近似为高斯）
        - 动量曲率半径 R = p / (0.3 B) [m]，B = 3.8 T (CMS/ATLAS)
        - 对于 1 TeV 轻子，R ≈ 877 m，曲率极小，近似直线

    Parameters
    ----------
    n_pixels : int
        探测器像素数（正方形）
    noise_level : float
        噪声相对幅度
    seed : int or None
        随机种子

    Returns
    -------
    hit_map : np.ndarray
        击中能量图，形状 (n_pixels, n_pixels)
    edge_map : np.ndarray
        边缘检测图
    """
    if seed is not None:
        np.random.seed(seed)

    hit_map = np.zeros((n_pixels, n_pixels))

    # 生成两条背对背径迹（模拟 Z' → ℓ⁺ℓ⁻）
    # 径迹 1: 从左下到右上
    # 径迹 2: 从右下到左上
    for track_idx in range(2):
        if track_idx == 0:
            x0, y0 = n_pixels * 0.15, n_pixels * 0.15
            angle = np.pi / 4.0
        else:
            x0, y0 = n_pixels * 0.85, n_pixels * 0.15
            angle = 3.0 * np.pi / 4.0

        # 径迹长度
        length = n_pixels * 0.7
        n_steps = int(length * 2)

        for step in range(n_steps):
            t = step / n_steps
            x = x0 + t * length * np.cos(angle)
            y = y0 + t * length * np.sin(angle)

            # 加上小曲率（磁场效应）
            curvature = 0.05 * np.sin(t * np.pi)
            x += curvature * n_pixels * 0.1 * np.sin(angle)
            y -= curvature * n_pixels * 0.1 * np.cos(angle)

            ix = int(round(x))
            iy = int(round(y))

            # 高斯型能量沉积（模拟 MIP 的 Landau 分布核心）
            sigma = 1.5
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < n_pixels and 0 <= jy < n_pixels:
                        dist2 = dx ** 2 + dy ** 2
                        energy = np.exp(-dist2 / (2.0 * sigma ** 2))
                        # 增加 Bragg 峰末端效应
                        if t > 0.8:
                            energy *= (1.0 + 2.0 * (t - 0.8) / 0.2)
                        hit_map[jx, jy] += energy

    # 添加均匀噪声
    noise = np.random.normal(0.0, noise_level * np.max(hit_map), (n_pixels, n_pixels))
    hit_map += noise
    hit_map = np.maximum(hit_map, 0.0)

    # 边缘检测
    edge_map = news_edge_detector(hit_map, threshold=0.15)

    return hit_map, edge_map


def aperiodic_detector_geometry(
    nmax: int = 3,
    scale: float = 1.0
) -> np.ndarray:
    """
    构建基于非周期平铺（Aperiodic Tile）的探测器像素几何。

    使用 Spectre-S 和 Mystic-M 平铺规则构造探测器敏感单元的
    准晶排列，用于研究非周期像素布局对位置分辨率的影响。

    算法基于自相似的迭代替换规则：
        S_{n+1} = M_n + 7 S_n
        M_{n+1} = M_n + 6 S_n

    Parameters
    ----------
    nmax : int
        迭代次数（0 ≤ nmax ≤ 6，避免过大）
    scale : float
        像素尺度 [mm]

    Returns
    -------
    np.ndarray
        探测器单元中心坐标，形状 (N_cells, 2)
    """
    if not (0 <= nmax <= 6):
        raise ValueError("nmax 必须在 [0, 6] 范围内")

    # 基础 Spectre-S 单元（14边形近似，取质心）
    cos30 = np.cos(np.pi / 6.0)
    sin30 = np.sin(np.pi / 6.0)

    # S0 基础形状: 正六边形中心 + 偏移
    base_vertices = np.array([
        [0.0, 0.0],
        [cos30, sin30],
        [cos30 + cos30, sin30 + 0.5],
        [cos30, sin30 + 1.0],
        [0.0, 1.0],
        [-cos30, sin30 + 1.0],
        [-cos30, sin30],
    ]) * scale

    # 计算单元中心
    centers = [np.mean(base_vertices, axis=0)]

    # 迭代放置（简化版，使用 Spectre-S 规则的核心思想）
    # 实际非周期平铺较复杂，这里用旋转+平移构造准晶近似
    angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    for n in range(1, nmax + 1):
        new_centers = []
        factor = scale * (1.5 ** n)
        for center in centers:
            for angle_deg in angles:
                angle = np.radians(angle_deg + 15.0 * n)
                offset = np.array([factor * np.cos(angle), factor * np.sin(angle)])
                new_centers.append(center + offset)
        centers.extend(new_centers)

    coords = np.array(centers)
    # 去重（考虑数值精度）
    coords_rounded = np.round(coords, decimals=8)
    unique_coords = np.unique(coords_rounded, axis=0)

    return unique_coords * scale


def detector_energy_resolution(
    energy: np.ndarray,
    a_stoch: float = 0.1,
    b_const: float = 0.01,
    c_noise: float = 0.5
) -> np.ndarray:
    """
    计算探测器的能量分辨率响应。

    典型的电磁量能器能量分辨率参数化：
        \frac{\sigma_E}{E} = \frac{a}{\sqrt{E}} \oplus b \oplus \frac{c}{E}

    其中:
        a = 10%   (随机项，反映淋浴涨落)
        b = 1%    (常数项，反映非均匀性)
        c = 0.5 GeV (噪声项)

    总分辨率:
        \sigma_E = E \sqrt{ \frac{a^2}{E} + b^2 + \frac{c^2}{E^2} }
                 = \sqrt{ a^2 E + b^2 E^2 + c^2 }

    Parameters
    ----------
    energy : np.ndarray
        真实能量 [GeV]
    a_stoch, b_const, c_noise : float
        分辨率参数

    Returns
    -------
    np.ndarray
        模拟的测量能量（加入高斯展宽）
    """
    energy = np.atleast_1d(energy)
    energy = np.maximum(energy, 1e-6)

    sigma_e = np.sqrt(a_stoch ** 2 * energy + b_const ** 2 * energy ** 2 + c_noise ** 2)

    # 边界: 分辨率不能大于能量本身（物理约束）
    sigma_e = np.minimum(sigma_e, 0.99 * energy)

    measured = energy + np.random.normal(0.0, sigma_e)
    measured = np.maximum(measured, 0.0)

    return measured
