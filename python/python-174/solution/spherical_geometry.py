"""
spherical_geometry.py
球面几何计算模块

融合种子项目:
- 185_circles (圆的几何参数化与边界判定)
- 1132_spherical_harmonic (归一化连带Legendre函数)
- 952_quadrilateral (四边形面积计算)

科学背景:
在快速多极子方法(FMM)中，空间区域常被近似为球体或球壳。我们需要精确计算:
1. 球坐标系下的角度距离
2. 球面上点的均匀采样
3. 球冠面积与立体角
4. 四边形/三角形在球面上的投影面积

核心公式:
- 球坐标转换: x = r*sin(theta)*cos(phi), y = r*sin(theta)*sin(phi), z = r*cos(theta)
- 球面上两点间的大圆距离: d = r * arccos( sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2) )
- 球冠面积: A = 2*pi*r^2*(1-cos(alpha)), 其中alpha为半顶角
- 立体角: Omega = A / r^2
"""

import numpy as np
from math import factorial


def cartesian_to_spherical(xyz):
    """
    将笛卡尔坐标转换为球坐标 (r, theta, phi)
    
    参数:
        xyz: ndarray (N, 3) 或 (3,), 笛卡尔坐标
    
    返回:
        (r, theta, phi): 半径, 极角(0~pi), 方位角(0~2pi)
    """
    xyz = np.atleast_2d(xyz)
    r = np.linalg.norm(xyz, axis=1)
    r_safe = np.where(r < 1e-15, 1.0, r)
    theta = np.arccos(np.clip(xyz[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    将球坐标转换为笛卡尔坐标
    
    公式:
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def great_circle_distance(r, theta1, phi1, theta2, phi2):
    """
    计算球面上两点间的大圆距离 (Haversine公式变体)
    
    公式:
        cos(d/r) = sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)
        d = r * arccos( ... )
    
    数值稳定性处理: 当两点非常接近时使用 arcsin 公式
    """
    delta_phi = phi1 - phi2
    cos_d = (np.sin(theta1) * np.sin(theta2) * np.cos(delta_phi)
             + np.cos(theta1) * np.cos(theta2))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    # 边界处理: 若cos_d接近1, 使用小角度近似 d ≈ r * sqrt(2*(1-cos_d))
    if np.isscalar(cos_d):
        if cos_d > 0.999999999:
            sin_half = np.sin(0.5 * np.arccos(cos_d))
            return 2.0 * r * sin_half
        return r * np.arccos(cos_d)
    else:
        close_mask = cos_d > 0.999999999
        d = np.zeros_like(cos_d)
        d[close_mask] = r * np.sqrt(2.0 * (1.0 - cos_d[close_mask]))
        d[~close_mask] = r * np.arccos(cos_d[~close_mask])
        return d


def spherical_cap_area(r, alpha):
    """
    计算球冠面积
    
    公式: A = 2 * pi * r^2 * (1 - cos(alpha))
    其中 alpha 为半顶角 (0 <= alpha <= pi)
    """
    alpha = np.clip(alpha, 0.0, np.pi)
    return 2.0 * np.pi * r * r * (1.0 - np.cos(alpha))


def solid_angle(r, cap_area):
    """
    计算立体角
    
    公式: Omega = A / r^2 (球面度, steradians)
    """
    if r < 1e-15:
        raise ValueError("半径必须为正数")
    return cap_area / (r * r)


def quadrilateral_area_2d(quad):
    """
    计算2D四边形面积 (融合952_quadrilateral)
    
    将四边形分解为两个三角形求和:
        A = A_tri(N1,N2,N3) + A_tri(N3,N4,N1)
    
    参数:
        quad: ndarray (4, 2), 按顺时针或逆时针排列的四边形顶点
    """
    quad = np.asarray(quad)
    if quad.shape != (4, 2):
        raise ValueError("quad必须是(4,2)数组")
    # 三角形1: (0,1,2)
    tri1 = quad[[0, 1, 2]]
    area1 = 0.5 * abs(
        tri1[0, 0] * (tri1[1, 1] - tri1[2, 1])
        + tri1[1, 0] * (tri1[2, 1] - tri1[0, 1])
        + tri1[2, 0] * (tri1[0, 1] - tri1[1, 1])
    )
    # 三角形2: (0,2,3)
    tri2 = quad[[0, 2, 3]]
    area2 = 0.5 * abs(
        tri2[0, 0] * (tri2[1, 1] - tri2[2, 1])
        + tri2[1, 0] * (tri2[2, 1] - tri2[0, 1])
        + tri2[2, 0] * (tri2[0, 1] - tri2[1, 1])
    )
    return area1 + area2


def legendre_associated_normalized(n, m, x):
    """
    归一化连带Legendre函数 (融合1132_spherical_harmonic)
    
    递推公式:
        P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
        P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
        P_l^m(x) = [ (2l-1)*x*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x) ] / (l-m)
    
    归一化因子 (球谐归一化):
        N_l^m = sqrt( (2l+1)*(l-m)! / (4*pi*(l+m)!) )
    
    参数:
        n: int, 最大阶数
        m: int, 次阶数 (0 <= m <= n)
        x: float, -1 <= x <= 1
    
    返回:
        cx: ndarray (n+1,), cx[l] = N_l^m * P_l^m(x)
    """
    if m < 0:
        raise ValueError("m必须非负")
    if n < m:
        raise ValueError("n必须大于等于m")
    x = float(x)
    if x < -1.0 or x > 1.0:
        raise ValueError("x必须在[-1,1]范围内")

    cx = np.zeros(n + 1)
    cx[:m] = 0.0
    cx[m] = 1.0
    somx2 = np.sqrt(max(0.0, 1.0 - x * x))

    fact = 1.0
    for i in range(1, m + 1):
        cx[m] = -cx[m] * fact * somx2
        fact = fact + 2.0

    if m < n:
        cx[m + 1] = x * (2 * m + 1) * cx[m]

    for i in range(m + 2, n + 1):
        cx[i] = ((2 * i - 1) * x * cx[i - 1] + (-i - m + 1) * cx[i - 2]) / (i - m)

    # 球谐归一化
    for mm in range(m, n + 1):
        factor = np.sqrt(
            ((2 * mm + 1) * factorial(mm - m))
            / (4.0 * np.pi * factorial(mm + m))
        )
        cx[mm] = cx[mm] * factor

    return cx


def spherical_harmonic_basis(l_max, theta, phi):
    """
    计算球谐函数基
    
    公式:
        Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * exp(i*m*phi)
    
    返回实部和虚部分量 (c, s) 数组
    """
    cos_theta = np.cos(theta)
    c_all = []
    s_all = []
    for m in range(l_max + 1):
        plm = legendre_associated_normalized(l_max, m, cos_theta)
        c = plm * np.cos(m * phi)
        s = plm * np.sin(m * phi)
        c_all.append(c)
        s_all.append(s)
    return c_all, s_all


def uniform_sphere_sample(n):
    """
    在单位球面上均匀采样n个点
    
    使用Marsaglia方法:
        1. 生成正态分布随机向量 v = (x, y, z)
        2. 归一化: p = v / ||v||
    """
    if n <= 0:
        raise ValueError("n必须为正整数")
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def circle_points_on_plane(center, radius, normal, num_points=32):
    """
    在三维空间中给定平面上生成圆周上的点 (融合185_circles)
    
    参数:
        center: (3,), 圆心
        radius: float, 半径
        normal: (3,), 平面法向量
        num_points: int, 圆周采样点数
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    # 构造平面内的两个正交基
    if abs(normal[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-15)

    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    pts = center[:, None] + radius * (u[:, None] * np.cos(t)[None, :] + v[:, None] * np.sin(t)[None, :])
    return pts.T


def compute_bounding_sphere(points):
    """
    计算一组点的最小包围球 (近似)
    
    算法:
        1. 计算质心
        2. 计算最远点距离
        3. 返回 (center, radius)
    """
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("点集不能为空")
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    # 边界处理: 若radius为0, 至少给一个极小值
    if radius < 1e-15:
        radius = 1e-10
    return center, radius
