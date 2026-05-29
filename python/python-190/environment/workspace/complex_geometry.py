"""
complex_geometry.py
===================
基于种子项目 547_human_data 与 713_maple_area 的复杂几何处理模块。
提供人类轮廓边界生成、多边形内部 Monte Carlo 面积估计以及
复杂边界内的物理场采样，用于物理信息 GAN 的非规则域训练。

核心数学：
  1. 射线法判断点是否在多边形内部：
       从点 P 向右发射水平射线，统计与多边形边界的交点个数。
       若交点数为奇数，则 P 在内部；否则在外部。

  2. Monte Carlo 面积估计（种子项目 713_maple_area）：
       在包围盒 [xmin, xmax] × [ymin, ymax] 内均匀撒 N 个点，
       统计落在多边形内部的点数 N_in。
       面积估计：A_est = (N_in / N) · A_box。
       误差：σ_A = A_box · √(p·(1-p)/N)，其中 p = N_in/N。

  3. 复杂边界内的速度场采样：
       在非规则域内生成均匀采样点，结合 Navier-Stokes 精确解
       构造带有复杂边界的训练数据集。
"""

import numpy as np


def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """
    射线法判断点是否在多边形内部。

    Parameters
    ----------
    pt : np.ndarray, shape (2,)
        待判断点。
    poly : np.ndarray, shape (m, 2)
        多边形顶点（逆时针或顺时针均可）。

    Returns
    -------
    inside : bool
    """
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # 检查边是否与从 pt 向右的水平射线相交
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if xinters > x:
                inside = not inside
    return inside


def polygon_bounding_box(poly: np.ndarray) -> tuple:
    """返回多边形包围盒。"""
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    return (xmin, xmax, ymin, ymax)


def polygon_area_mc(poly: np.ndarray, n_samples: int = 10000,
                    seed: int = None) -> tuple:
    """
    Monte Carlo 估计多边形面积。

    Parameters
    ----------
    poly : np.ndarray, shape (m, 2)
        多边形顶点。
    n_samples : int
        采样点数。
    seed : int, optional
        随机种子。

    Returns
    -------
    area_est : float
        面积估计值。
    std_err : float
        标准误差。
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)
    box_area = (xmax - xmin) * (ymax - ymin)
    if box_area < 1e-15:
        return 0.0, 0.0
    x = rng.random(n_samples) * (xmax - xmin) + xmin
    y = rng.random(n_samples) * (ymax - ymin) + ymin
    pts = np.column_stack([x, y])
    inside = np.array([point_in_polygon(p, poly) for p in pts])
    p_in = float(np.mean(inside))
    area_est = p_in * box_area
    std_err = box_area * np.sqrt(p_in * (1.0 - p_in) / n_samples)
    return float(area_est), float(std_err)


def human_outline_polygon(scale: float = 1.0, n_points: int = 60) -> np.ndarray:
    """
    生成简化的人体轮廓多边形（参数化构造，基于种子项目 547_human_data 的概念）。

    使用分段参数曲线：
      · 头部：圆弧参数方程
      · 躯干：椭圆弧
      · 腿部：抛物线弧

    Parameters
    ----------
    scale : float
        缩放因子。
    n_points : int
        轮廓点总数。

    Returns
    -------
    poly : np.ndarray, shape (n_points, 2)
        逆时针排列的轮廓点。
    """
    pts = []
    # 分段参数数量
    n_head = n_points // 5
    n_torso = n_points // 5
    n_legs = n_points // 5
    n_arms = n_points // 5
    n_neck = n_points - n_head - n_torso - n_legs - n_arms

    # 头部圆弧（上半部分逆时针）
    for theta in np.linspace(np.pi * 0.5, -np.pi * 0.5, n_head):
        x = 0.5 * np.cos(theta)
        y = 2.5 + 0.5 * np.sin(theta)
        pts.append([x, y])

    # 颈部与右肩
    for t in np.linspace(0.0, 1.0, n_neck):
        x = 0.5 + 0.1 * t
        y = 2.0 - 0.3 * t
        pts.append([x, y])

    # 右臂外侧（简化）
    for t in np.linspace(0.0, 1.0, n_arms):
        x = 0.6 + 0.1 * np.sin(np.pi * t)
        y = 1.7 - 1.5 * t
        pts.append([x, y])

    # 右腿外侧
    for t in np.linspace(0.0, 1.0, n_legs):
        x = 0.4 * (1.0 - t)
        y = 0.2 - 2.0 * t
        pts.append([x, y])

    # 脚底到左腿外侧（对称返回）
    for t in np.linspace(0.0, 1.0, n_legs):
        x = -0.4 * t
        y = -1.8 + 2.0 * t
        pts.append([x, y])

    # 左臂外侧
    for t in np.linspace(0.0, 1.0, n_arms):
        x = -0.6 - 0.1 * np.sin(np.pi * t)
        y = 0.2 + 1.5 * t
        pts.append([x, y])

    # 左肩到颈部
    for t in np.linspace(0.0, 1.0, n_neck):
        x = -0.6 + 0.1 * t
        y = 1.7 + 0.3 * t
        pts.append([x, y])

    poly = np.array(pts, dtype=float) * scale
    return poly


def sample_in_polygon(poly: np.ndarray, n: int, seed: int = None) -> np.ndarray:
    """
    在多边形内部均匀随机采样 n 个点（拒绝采样）。

    Parameters
    ----------
    poly : np.ndarray, shape (m, 2)
        多边形顶点。
    n : int
        采样点数。
    seed : int, optional
        随机种子。

    Returns
    -------
    samples : np.ndarray, shape (n, 2)
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)
    samples = []
    max_attempts = n * 100
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        x = rng.random() * (xmax - xmin) + xmin
        y = rng.random() * (ymax - ymin) + ymin
        pt = np.array([x, y])
        if point_in_polygon(pt, poly):
            samples.append(pt)
        attempts += 1
    if len(samples) < n:
        # 退化为网格采样
        return grid_sample_in_polygon(poly, n)
    return np.array(samples)


def grid_sample_in_polygon(poly: np.ndarray, n: int) -> np.ndarray:
    """在多边形内部进行规则网格采样。"""
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)
    # 估计网格密度
    area_est, _ = polygon_area_mc(poly, n_samples=1000)
    side = np.sqrt(area_est / max(n, 1))
    if side < 1e-10:
        side = 0.01
    nx = max(3, int(np.ceil((xmax - xmin) / side)))
    ny = max(3, int(np.ceil((ymax - ymin) / side)))
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    inside = np.array([point_in_polygon(p, poly) for p in pts])
    valid = pts[inside]
    if valid.shape[0] < n:
        # 复制补充
        repeat = (n + valid.shape[0] - 1) // valid.shape[0]
        valid = np.tile(valid, (repeat, 1))[:n]
    return valid[:n]


def velocity_field_in_complex_domain(poly: np.ndarray, a: float, d: float,
                                     t_val: float, n_samples: int = 500,
                                     seed: int = None) -> tuple:
    """
    在复杂多边形域内生成带有物理精确解的速度场样本。
    将二维多边形点映射到三维空间中的 z=0 平面，计算 Ethier 精确解。

    Returns
    -------
    coords : np.ndarray, shape (n, 3)
        空间坐标 (x, y, z=0)。
    velocity : np.ndarray, shape (n, 3)
        速度场 (u, v, w)。
    pressure : np.ndarray, shape (n,)
        压力场 p。
    """
    pts_2d = sample_in_polygon(poly, n_samples, seed)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    z = np.zeros_like(x)
    t = np.full_like(x, t_val)
    from navier_stokes_exact import uvwp_ethier
    u, v, w, p = uvwp_ethier(a, d, x, y, z, t)
    coords = np.column_stack([x, y, z])
    velocity = np.column_stack([u, v, w])
    return coords, velocity, p
