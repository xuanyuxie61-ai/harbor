# -*- coding: utf-8 -*-
"""
target_geometry.py

基于 cities (spherical distance), sphere_stereograph (stereographic projection),
与 quadrilateral (3D quadrilateral area) 的靶丸几何与坐标变换模块。

原项目 186_cities 提供了球面大圆距离计算；
原项目 1127_sphere_stereograph 提供了球面与平面的保角映射；
原项目 952_quadrilateral 提供了 3D 四边形面积计算。
三者融合后用于:
    1. 惯性约束聚变靶丸的球面几何建模（半径、表面积、体积）。
    2. 激光入射角的球面坐标表示与立体角计算。
    3. 立体投影将靶丸表面映射到平面，用于参数化表面密度分布。
    4. 3D 四边形面片网格用于计算靶丸表面积和射线-表面交点。

核心公式:
    球面距离 (Haversine):
        d = R * arccos(sin φ1 sin φ2 + cos φ1 cos φ2 cos(Δλ))

    立体投影 (从南极 S=(0,0,-1) 到平面 z=0):
        q = (2/(1+p_z)) * p + (1 - 2/(1+p_z)) * S

    3D 四边形面积 (Varignon 平行四边形法):
        A = 2 * | (P1-P3) × (P2-P4) | / 2
"""

import numpy as np


def spherical_distance(lat1, lon1, lat2, lon2, radius):
    """
    计算球面上两点间的大圆距离。

    公式:
        θ = arccos(sin(φ1) sin(φ2) + cos(φ1) cos(φ2) cos(Δλ))
        d = R * θ

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        纬度和经度 [rad]。
    radius : float
        球半径 [m]。

    Returns
    -------
    dist : float
        大圆距离 [m]。
    """
    if radius <= 0:
        raise ValueError("半径必须为正。")
    # 数值保护: 使用 arccos 的参数可能因浮点误差略微超出 [-1,1]
    arg = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    arg = np.clip(arg, -1.0, 1.0)
    theta = np.arccos(arg)
    dist = radius * theta
    return dist


def spherical_to_cartesian(lat, lon, radius):
    """
    球坐标转笛卡尔坐标。

    Parameters
    ----------
    lat, lon : float
        纬度和经度 [rad]。
    radius : float
        半径 [m]。

    Returns
    -------
    x, y, z : float
        笛卡尔坐标。
    """
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    笛卡尔坐标转球坐标。

    Parameters
    ----------
    x, y, z : float
        笛卡尔坐标。

    Returns
    -------
    lat, lon, r : float
        纬度、经度 [rad] 和半径 [m]。
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-30:
        return 0.0, 0.0, 0.0
    lat = np.arcsin(np.clip(z / r, -1.0, 1.0))
    lon = np.arctan2(y, x)
    return lat, lon, r


def stereographic_projection_sphere_to_plane(p):
    """
    从单位球面到平面的立体投影（南极投影到 z=0 平面）。

    原 sphere_stereograph 核心算法:
        s = 2 / (1 + p_z)
        q = s * p + (1 - s) * S
        其中 S = (0, 0, -1) 为南极。

    Parameters
    ----------
    p : ndarray, shape (3,) or (3, N)
        球面上的点（单位球）。

    Returns
    -------
    q : ndarray
        投影平面上的点（z 坐标为 0）。
    """
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        if len(p) != 3:
            raise ValueError("p 必须是 3 维向量。")
        pz = p[2]
        if 1.0 + pz < 1e-15:
            # 北极附近映射到无穷远，截断处理
            s = 1e15
        else:
            s = 2.0 / (1.0 + pz)
        S = np.array([0.0, 0.0, -1.0])
        q = s * p + (1.0 - s) * S
        return q[:2]  # 返回 (x, y)
    else:
        if p.shape[0] != 3:
            raise ValueError("p 的第一维必须为 3。")
        pz = p[2, :]
        s = np.where(1.0 + pz < 1e-15, 1e15, 2.0 / (1.0 + pz))
        q = np.zeros_like(p)
        for i in range(3):
            q[i, :] = s * p[i, :] + (1.0 - s) * (-1.0 if i == 2 else 0.0)
        return q[:2, :]  # (2, N)


def stereographic_projection_plane_to_sphere(q):
    """
    从平面到单位球面的逆立体投影。

    公式:
        令 u = q_x, v = q_y
        r^2 = u^2 + v^2
        p_x = 4u / (4 + r^2)
        p_y = 4v / (4 + r^2)
        p_z = (4 - r^2) / (4 + r^2)

    Parameters
    ----------
    q : ndarray, shape (2,) or (2, N)
        平面上的点。

    Returns
    -------
    p : ndarray
        球面上的点。
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        u, v = q[0], q[1]
        r2 = u**2 + v**2
        denom = 4.0 + r2
        px = 4.0 * u / denom
        py = 4.0 * v / denom
        pz = (4.0 - r2) / denom
        return np.array([px, py, pz])
    else:
        u = q[0, :]
        v = q[1, :]
        r2 = u**2 + v**2
        denom = 4.0 + r2
        p = np.zeros((3, q.shape[1]))
        p[0, :] = 4.0 * u / denom
        p[1, :] = 4.0 * v / denom
        p[2, :] = (4.0 - r2) / denom
        return p


def quadrilateral_area_3d(q):
    """
    计算三维空间中四边形的面积。

    原 quadrilateral_area_3d 核心算法 (Varignon 平行四边形法):
        1. 取四边中点构成 Varignon 平行四边形:
           P1 = (Q1+Q2)/2, P2 = (Q2+Q3)/2, P3 = (Q3+Q4)/2, P4 = (Q4+Q1)/2
        2. 平行四边形面积 = | (P2-P1) × (P4-P1) |
        3. 原四边形面积 = 2 * 平行四边形面积

    Parameters
    ----------
    q : ndarray, shape (3, 4)
        四个顶点，按逆时针顺序排列（假设共面）。

    Returns
    -------
    area : float
        四边形面积 [m^2]。
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (3, 4):
        raise ValueError("q 必须是形状 (3, 4) 的数组。")

    # Varignon 平行四边形顶点
    p = np.zeros((3, 4))
    p[:, 0] = (q[:, 0] + q[:, 1]) / 2.0
    p[:, 1] = (q[:, 1] + q[:, 2]) / 2.0
    p[:, 2] = (q[:, 2] + q[:, 3]) / 2.0
    p[:, 3] = (q[:, 3] + q[:, 0]) / 2.0

    # 平行四边形面积 = |cross(P1P2, P1P4)|
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 3] - p[:, 0]
    cross = np.cross(v1, v2)
    para_area = np.linalg.norm(cross)
    area = 2.0 * para_area
    return area


def icf_target_surface_mesh(R0, n_theta, n_phi):
    """
    生成惯性约束聚变靶丸的球面四边形网格。

    Parameters
    ----------
    R0 : float
        靶丸半径 [m]。
    n_theta, n_phi : int
        极角和方位角方向的网格数。

    Returns
    -------
    vertices : ndarray, shape (n_theta, n_phi, 3)
        网格顶点。
    face_areas : ndarray, shape (n_theta-1, n_phi-1)
        每个四边形面片的面积。
    total_area : float
        靶丸总表面积。
    """
    if R0 <= 0 or n_theta < 2 or n_phi < 2:
        raise ValueError("R0 必须为正，n_theta 和 n_phi 必须 >= 2。")

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    vertices = np.zeros((n_theta, n_phi, 3))
    for i in range(n_theta):
        for j in range(n_phi):
            vertices[i, j, :] = spherical_to_cartesian(
                np.pi / 2.0 - theta[i], phi[j], R0
            )

    face_areas = np.zeros((n_theta - 1, n_phi - 1))
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            q = np.zeros((3, 4))
            q[:, 0] = vertices[i, j, :]
            q[:, 1] = vertices[i + 1, j, :]
            q[:, 2] = vertices[i + 1, j + 1, :]
            q[:, 3] = vertices[i, j + 1, :]
            face_areas[i, j] = quadrilateral_area_3d(q)

    total_area = np.sum(face_areas)
    return vertices, face_areas, total_area


def solid_angle_subtended_by_quad(q, observer):
    """
    计算从观察点看四边形面元所张的立体角。

    使用向量法:
        Ω = 2 * arctan( | (r1 × r2) · r3 | / (|r1||r2||r3| + (r1·r2)|r3| + (r1·r3)|r2| + (r2·r3)|r1|) )
    （此处简化为将四边形剖分为两个三角形后求和）

    Parameters
    ----------
    q : ndarray, shape (3, 4)
        四边形顶点。
    observer : ndarray, shape (3,)
        观察点。

    Returns
    -------
    omega : float
        立体角 [sr]。
    """
    q = np.asarray(q, dtype=float)
    observer = np.asarray(observer, dtype=float)

    # 剖分为两个三角形: (0,1,2) 和 (0,2,3)
    tri_indices = [(0, 1, 2), (0, 2, 3)]
    omega_total = 0.0

    for (i1, i2, i3) in tri_indices:
        r1 = q[:, i1] - observer
        r2 = q[:, i2] - observer
        r3 = q[:, i3] - observer

        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)
        n3 = np.linalg.norm(r3)
        if n1 < 1e-20 or n2 < 1e-20 or n3 < 1e-20:
            continue

        r1u = r1 / n1
        r2u = r2 / n2
        r3u = r3 / n3

        dot12 = np.dot(r1u, r2u)
        dot13 = np.dot(r1u, r3u)
        dot23 = np.dot(r2u, r3u)

        cross12 = np.cross(r1u, r2u)
        triple = abs(np.dot(cross12, r3u))

        denom = 1.0 + dot12 + dot13 + dot23
        if denom < 1e-15:
            continue
        omega_tri = 2.0 * np.arctan2(triple, denom)
        omega_total += omega_tri

    return omega_total


def laser_incidence_angle_on_sphere(laser_dir, surface_normal):
    """
    计算激光入射方向与靶丸表面法向的夹角。

    公式:
        cos θ = | n̂ · d̂ |
        θ = arccos(|n̂ · d̂|)

    Parameters
    ----------
    laser_dir : ndarray, shape (3,)
        激光方向（单位向量）。
    surface_normal : ndarray, shape (3,)
        表面法向（单位向量）。

    Returns
    -------
    theta : float
        入射角 [rad]。
    cos_theta : float
        cos(θ)。
    """
    d = np.asarray(laser_dir, dtype=float)
    n = np.asarray(surface_normal, dtype=float)
    dn = np.linalg.norm(d)
    nn = np.linalg.norm(n)
    if dn < 1e-20 or nn < 1e-20:
        return 0.0, 1.0
    cos_val = abs(np.dot(d / dn, n / nn))
    cos_val = np.clip(cos_val, 0.0, 1.0)
    theta = np.arccos(cos_val)
    return theta, cos_val
