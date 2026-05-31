"""
六边形边界离散化与处理模块

融合自:
- 108_boundary_word_hexagon: 六边形网格的边界词表示

在自适应网格细化 (AMR) 中，边界表示的精确性对数值解的精度至关重要。
六边形密铺具有最优的面积-周界比，在边界离散中提供更高的几何保真度。

数学背景:
    正六边形的密铺是平面上最密集的圆 packing，
    其边界可用一个"边界词"（boundary word）描述:
        在六边形格点上，每一步有 6 个可能的方向
        （通常记为 a, b, c, a^{-1}, b^{-1}, c^{-1}）。
    
    边界词的形式化:
        设六边形中心位于复平面的 Eisenstein 整数环:
            Z[ω] = {m + nω : m, n ∈ Z},  ω = e^{2πi/3}
        
        六边形网格的邻接关系由 6 个方向向量定义:
            d_0 = 1
            d_1 = ω
            d_2 = ω^2 = -1 - ω
            d_3 = -1
            d_4 = -ω
            d_5 = -ω^2 = 1 + ω
"""

import numpy as np


# 六边形网格的6个方向向量 (在二维平面上)
HEX_DIRECTIONS = np.array([
    [1.0, 0.0],           # 0°
    [0.5, np.sqrt(3.0) / 2.0],   # 60°
    [-0.5, np.sqrt(3.0) / 2.0],  # 120°
    [-1.0, 0.0],          # 180°
    [-0.5, -np.sqrt(3.0) / 2.0], # 240°
    [0.5, -np.sqrt(3.0) / 2.0]   # 300°
])


def axial_to_cartesian(q, r, size=1.0):
    """
    将六边形轴向坐标 (q, r) 转换为笛卡尔坐标 (x, y)。
    
    六边形轴向坐标系:
        x = size * (3/2 * q)
        y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    
    Parameters
    ----------
    q, r : int or ndarray
        轴向坐标
    size : float
        六边形边长
    
    Returns
    -------
    x, y : float or ndarray
        笛卡尔坐标
    """
    x = size * (3.0 / 2.0 * q)
    y = size * (np.sqrt(3.0) / 2.0 * q + np.sqrt(3.0) * r)
    return x, y


def cartesian_to_axial(x, y, size=1.0):
    """
    将笛卡尔坐标转换为六边形轴向坐标。
    
    Parameters
    ----------
    x, y : float or ndarray
    size : float
    
    Returns
    -------
    q, r : float or ndarray
    """
    q = (2.0 / 3.0 * x) / size
    r = (-1.0 / 3.0 * x + np.sqrt(3.0) / 3.0 * y) / size
    return q, r


def hex_round(q, r):
    """
    将浮点轴向坐标舍入到最近的六边形中心。
    
    Parameters
    ----------
    q, r : float or ndarray
    
    Returns
    -------
    q_int, r_int : int or ndarray
    """
    if isinstance(q, np.ndarray):
        s = -q - r
        x = np.round(q)
        y = np.round(r)
        z = np.round(s)

        x_diff = np.abs(x - q)
        y_diff = np.abs(y - r)
        z_diff = np.abs(z - s)

        mask_x = (x_diff > y_diff) & (x_diff > z_diff)
        x[mask_x] = -y[mask_x] - z[mask_x]

        mask_y = (~mask_x) & (y_diff > z_diff)
        y[mask_y] = -x[mask_y] - z[mask_y]

        mask_z = (~mask_x) & (~mask_y)
        z[mask_z] = -x[mask_z] - y[mask_z]

        return x.astype(int), y.astype(int)
    else:
        s = -q - r
        x = round(q)
        y = round(r)
        z = round(s)

        x_diff = abs(x - q)
        y_diff = abs(y - r)
        z_diff = abs(z - s)

        if x_diff > y_diff and x_diff > z_diff:
            x = -y - z
        elif y_diff > z_diff:
            y = -x - z
        else:
            z = -x - y

        return int(x), int(y)


def generate_hexagonal_lattice(radius, size=1.0):
    """
    生成以原点为中心、给定半径的六边形格点。
    
    Parameters
    ----------
    radius : int
        六边形区域的半径（轴向距离）
    size : float
        六边形边长
    
    Returns
    -------
    points : ndarray, shape (n, 2)
        六边形格点的笛卡尔坐标
    axial_coords : ndarray, shape (n, 2)
        对应的轴向坐标
    """
    points = []
    axial = []

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x, y = axial_to_cartesian(q, r, size)
            points.append([x, y])
            axial.append([q, r])

    return np.array(points), np.array(axial, dtype=int)


def boundary_word_to_polygon(boundary_word, start_q=0, start_r=0, size=1.0):
    """
    将六边形边界词转换为多边形顶点序列。
    
    边界词是一个方向序列，例如 [0, 1, 0, 5, 4, 5] 表示
    沿着六边形边缘依次移动的路径。
    
    Parameters
    ----------
    boundary_word : list of int
        方向索引 (0-5)
    start_q, start_r : int
        起始轴向坐标
    size : float
    
    Returns
    -------
    vertices : ndarray, shape (n, 2)
        多边形的顶点坐标
    """
    vertices = []
    q, r = start_q, start_r

    x, y = axial_to_cartesian(q, r, size)
    vertices.append([x, y])

    for direction in boundary_word:
        dq, dr = HEX_DIRECTIONS[direction][:2]  # 简化：仅使用方向索引
        # 轴向移动
        if direction == 0:
            q += 1
        elif direction == 1:
            r += 1
        elif direction == 2:
            q -= 1
            r += 1
        elif direction == 3:
            q -= 1
        elif direction == 4:
            r -= 1
        elif direction == 5:
            q += 1
            r -= 1

        x, y = axial_to_cartesian(q, r, size)
        vertices.append([x, y])

    return np.array(vertices)


def approximate_boundary_with_hexagons(
    boundary_func,
    domain_bounds,
    hex_size,
    n_samples=1000
):
    """
    用六边形格点近似连续边界。
    
    在区域边界附近采样，将边界投影到最近的六边形格点上，
    生成一个多边形近似。
    
    Parameters
    ----------
    boundary_func : callable
        boundary_func(t) -> (x, y), t ∈ [0, 2π]
        参数化边界曲线
    domain_bounds : tuple
        ((xmin, xmax), (ymin, ymax))
    hex_size : float
        六边形边长
    n_samples : int
        边界采样点数
    
    Returns
    -------
    hex_points : ndarray
        边界上的六边形格点
    boundary_points : ndarray
        原始边界采样点
    """
    t_vals = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    boundary_points = np.array([boundary_func(t) for t in t_vals])

    # 转换为轴向坐标并舍入
    q_vals, r_vals = cartesian_to_axial(
        boundary_points[:, 0], boundary_points[:, 1], hex_size
    )
    q_int, r_int = hex_round(q_vals, r_vals)

    # 去重
    unique_coords = set()
    hex_points = []
    for q, r in zip(q_int, r_int):
        key = (q, r)
        if key not in unique_coords:
            unique_coords.add(key)
            x, y = axial_to_cartesian(q, r, hex_size)
            hex_points.append([x, y])

    return np.array(hex_points), boundary_points


def hex_boundary_refinement_indicator(hex_points, solution_values, hex_size):
    """
    基于六边形边界离散计算边界细化指示子。
    
    在边界处，解的梯度通常较大，需要更精细的网格。
    该函数计算每个边界六边形的"边界通量"，用于指导 AMR。
    
    Parameters
    ----------
    hex_points : ndarray, shape (n, 2)
        六边形边界点
    solution_values : ndarray, shape (n,)
        边界点上的解值
    hex_size : float
    
    Returns
    -------
    indicators : ndarray, shape (n,)
        每个边界点的细化指示子
    """
    n = len(hex_points)
    indicators = np.zeros(n)

    for i in range(n):
        # 找到最近的邻居
        dists = np.linalg.norm(hex_points - hex_points[i], axis=1)
        neighbor_mask = (dists > 1e-10) & (dists < 2.5 * hex_size)

        if np.sum(neighbor_mask) > 0:
            neighbor_vals = solution_values[neighbor_mask]
            # 细化指示子 = 局部梯度 × 单元尺寸
            local_grad = np.max(np.abs(neighbor_vals - solution_values[i])) / hex_size
            indicators[i] = local_grad * hex_size
        else:
            indicators[i] = 0.0

    return indicators
