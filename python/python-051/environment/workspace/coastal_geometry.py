"""
coastal_geometry.py
===================
曲边海岸几何处理与区域积分模块。

源自 184_circle_segment 的核心算法被扩展应用于海洋盆地边界：
- 圆形/弧形海岸线边界（如海湾、半岛）的精确面积与形心计算
- 高斯求积规则在曲线边界区域上的数值积分
- 刚性边界条件在曲边上的投影

数学基础
--------
1. 圆缺（circle segment）面积：
   对半径 R、弓高 h 的圆缺：
       θ = 2·arcsin(√(R² - (R-h)²) / R)
       A = R²(θ - sin θ)/2

2. 圆缺形心（centroid）距弦的距离：
       d = 4R sin³(θ/2) / (3(θ - sin θ))

3. 三角高斯求积（trigauss）：
   在标准三角形上构造高斯型求积规则，通过 Duffy 变换映射到曲边区域。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 圆缺几何（源自 184_circle_segment）
# ---------------------------------------------------------------------------

def circle_segment_area_from_height(R, h):
    """
    计算圆缺面积。

    参数
    ----
    R : float
        圆半径，R > 0
    h : float
        弓高，0 ≤ h ≤ 2R

    返回
    ----
    area : float
    """
    if R <= 0:
        raise ValueError("R > 0")
    if h <= 0.0:
        return 0.0
    if h >= 2.0 * R:
        return np.pi * R * R

    # θ = 2·arcsin(√(R² - (R-h)²)/R)
    tmp = np.sqrt(max(0.0, R * R - (R - h) * (R - h))) / R
    tmp = min(1.0, max(-1.0, tmp))
    theta = 2.0 * np.arcsin(tmp)

    if h <= R:
        area = R * R * (theta - np.sin(theta)) / 2.0
    else:
        theta = 2.0 * np.pi - theta
        area = R * R * (theta - np.sin(theta)) / 2.0

    return area


def circle_segment_centroid_from_height(R, h):
    """
    计算圆缺形心到弦的距离（朝向弧的方向为正）。

    返回
    ----
    d : float
        形心距离
    """
    if R <= 0:
        raise ValueError("R > 0")
    if h <= 0.0:
        return 0.0
    if h >= 2.0 * R:
        return 0.0

    tmp = np.sqrt(max(0.0, R * R - (R - h) * (R - h))) / R
    tmp = min(1.0, max(-1.0, tmp))
    theta = 2.0 * np.arcsin(tmp)
    if h > R:
        theta = 2.0 * np.pi - theta

    area = R * R * (theta - np.sin(theta)) / 2.0
    if abs(area) < 1e-30:
        return 0.0
    d = 4.0 * R * (np.sin(theta / 2.0) ** 3) / (3.0 * (theta - np.sin(theta)))
    return d


def circle_segment_height_from_area(R, area):
    """
    由面积反求圆缺弓高（牛顿迭代）。
    """
    if area <= 0.0:
        return 0.0
    if area >= np.pi * R * R:
        return 2.0 * R

    h = R  # 初始猜测
    for _ in range(50):
        f = circle_segment_area_from_height(R, h) - area
        if abs(f) < 1e-12:
            break
        # 导数近似
        dh = max(1e-8, h * 1e-6)
        fp = (circle_segment_area_from_height(R, h + dh) - circle_segment_area_from_height(R, h - dh)) / (2 * dh)
        if abs(fp) < 1e-30:
            break
        h = h - f / fp
        h = max(0.0, min(2.0 * R, h))
    return h


# ---------------------------------------------------------------------------
# 曲边海岸区域积分
# ---------------------------------------------------------------------------

def quadrature_on_curved_domain(func, x_range, z_range, arc_centers, arc_radii,
                                n_x=20, n_z=20):
    """
    在由矩形与圆弧边界定义的海洋区域上计算 ∫∫ f(x,z) dx dz。

    参数
    ----
    func : callable
        f(x, z) -> scalar
    x_range : (xmin, xmax)
    z_range : (zmin, zmax)
    arc_centers : list of (cx, cz)
        圆弧中心（用于定义凹陷/凸出的海岸边界）
    arc_radii : list of float
        对应圆弧半径（正为凸出，负为凹陷）
    n_x, n_z : int
        每个子区域的均匀采样数

    返回
    ----
    integral : float
    """
    xmin, xmax = x_range
    zmin, zmax = z_range
    dx = (xmax - xmin) / n_x
    dz = (zmax - zmin) / n_z

    total = 0.0
    for i in range(n_x):
        x0 = xmin + (i + 0.5) * dx
        for j in range(n_z):
            z0 = zmin + (j + 0.5) * dz
            # 判断是否在有效域内（扣除凹陷圆弧区域）
            inside = True
            for (cx, cz), r in zip(arc_centers, arc_radii):
                dist = np.sqrt((x0 - cx) ** 2 + (z0 - cz) ** 2)
                if r < 0 and dist < abs(r):
                    inside = False
                    break
                if r > 0 and dist > r:
                    inside = False
                    break
            if inside:
                total += func(x0, z0) * dx * dz

    return total


def coastal_boundary_length(arc_centers, arc_radii, arc_angles):
    """
    计算曲边海岸边界总长度。

    参数
    ----
    arc_angles : list of float
        每段圆弧对应的圆心角 [rad]
    """
    length = 0.0
    for r, ang in zip(arc_radii, arc_angles):
        length += abs(r) * ang
    return length


# ---------------------------------------------------------------------------
# 局部坐标变换与投影
# ---------------------------------------------------------------------------

def local_coordinates_circle_segment(x, z, cx, cz, R, theta_start, theta_end):
    """
    将全局坐标 (x,z) 转换到以圆弧局部正交坐标系 (s, n)：
        s : 沿弧长参数
        n : 法向距离（向外为正）
    """
    dx = x - cx
    dz = z - cz
    r_local = np.sqrt(dx ** 2 + dz ** 2)
    theta = np.arctan2(dz, dx)

    # 归一化到起始角
    theta_norm = (theta - theta_start) % (2.0 * np.pi)
    arc_len = abs(R) * theta_norm
    normal_dist = r_local - abs(R)

    return arc_len, normal_dist


def project_velocity_to_boundary(u, w, boundary_normal):
    """
    将速度场投影到边界法向与切向：
        u_n = u·n_x + w·n_z
        u_t = -u·n_z + w·n_x
    """
    nx, nz = boundary_normal
    un = u * nx + w * nz
    ut = -u * nz + w * nx
    return un, ut
