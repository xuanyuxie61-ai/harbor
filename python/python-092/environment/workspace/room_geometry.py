"""
room_geometry.py
三维室内声场几何定义与有向距离函数 (SDF)
基于 distmesh_3d 与 gpl_display 核心思想重构

声学工程应用：定义封闭空间（音乐厅、录音棚）的几何边界，
用于后续射线追踪与有限元网格生成。
"""

import numpy as np


def dsphere(p, xc, yc, zc, r):
    """
    球体有向距离函数。
    d < 0 表示球内，d > 0 表示球外。
    来自 distmesh_3d 核心算法。
    """
    return np.sqrt((p[:, 0] - xc) ** 2 +
                   (p[:, 1] - yc) ** 2 +
                   (p[:, 2] - zc) ** 2) - r


def dbox(p, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    轴对齐长方体有向距离函数。
    返回点到长方体边界的有向距离（内部为负）。
    """
    dx = np.maximum(np.maximum(x_min - p[:, 0], p[:, 0] - x_max), 0.0)
    dy = np.maximum(np.maximum(y_min - p[:, 1], p[:, 1] - y_max), 0.0)
    dz = np.maximum(np.maximum(z_min - p[:, 2], p[:, 2] - z_max), 0.0)
    # 内部距离：取最近面距离的负值
    inside_dist = -np.minimum(np.minimum(
        np.minimum(p[:, 0] - x_min, x_max - p[:, 0]),
        np.minimum(p[:, 1] - y_min, y_max - p[:, 1])),
        np.minimum(p[:, 2] - z_min, z_max - p[:, 2]))
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dist[dist < 1e-14] = inside_dist[dist < 1e-14]
    return dist


def ddiff(d1, d2):
    """
    集合差的有向距离函数：区域1 ∖ 区域2。
    来自 distmesh_3d。
    """
    return np.maximum(d1, -d2)


def dintersect(d1, d2):
    """
    集合交的有向距离函数：区域1 ∩ 区域2。
    来自 distmesh_3d。
    """
    return np.maximum(d1, d2)


def huniform(p):
    """
    均匀网格尺寸函数（返回全1）。
    来自 distmesh_3d。
    """
    return np.ones(p.shape[0])


def dshoebox_with_pillars(p):
    """
    定义一个 shoebox 音乐厅并带有圆柱形立柱（吸声柱）。
    用于声学模态分析与射线追踪。

    房间尺寸：10m x 8m x 5m (长 x 宽 x 高)
    吸声柱：半径 0.3m，位于 (3,3)、(7,5)
    """
    room = dbox(p, 0.0, 10.0, 0.0, 8.0, 0.0, 5.0)
    pillar1 = dsphere(p, 3.0, 3.0, 2.5, 0.3)
    pillar2 = dsphere(p, 7.0, 5.0, 2.5, 0.3)
    return ddiff(room, dintersect(pillar1, pillar2))


def extract_room_surfaces():
    """
    提取房间表面三角形patch数据。
    基于 gpl_display 的 structured/unstructured grid 思想，
    返回房间各墙面、天花、地板的三角形顶点列表。
    """
    surfaces = {}
    # 地板 (z=0)
    surfaces['floor'] = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0],
        [0.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0]
    ], dtype=float)
    # 天花 (z=5)
    surfaces['ceiling'] = np.array([
        [0.0, 0.0, 5.0], [10.0, 8.0, 5.0], [10.0, 0.0, 5.0],
        [0.0, 0.0, 5.0], [0.0, 8.0, 5.0], [10.0, 8.0, 5.0]
    ], dtype=float)
    # 前墙 (y=0)
    surfaces['front_wall'] = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 5.0], [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]
    ], dtype=float)
    # 后墙 (y=8)
    surfaces['back_wall'] = np.array([
        [0.0, 8.0, 0.0], [10.0, 8.0, 0.0], [10.0, 8.0, 5.0],
        [0.0, 8.0, 0.0], [10.0, 8.0, 5.0], [0.0, 8.0, 5.0]
    ], dtype=float)
    # 左墙 (x=0)
    surfaces['left_wall'] = np.array([
        [0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 8.0, 5.0],
        [0.0, 0.0, 0.0], [0.0, 8.0, 5.0], [0.0, 0.0, 5.0]
    ], dtype=float)
    # 右墙 (x=10)
    surfaces['right_wall'] = np.array([
        [10.0, 0.0, 0.0], [10.0, 8.0, 5.0], [10.0, 8.0, 0.0],
        [10.0, 0.0, 0.0], [10.0, 0.0, 5.0], [10.0, 8.0, 5.0]
    ], dtype=float)
    return surfaces


def compute_surface_normals(surfaces):
    """
    计算各表面的单位外法向量（指向房间内部为吸收面）。
    对于封闭空间，外法向指向房间外部。
    """
    normals = {}
    for name, tris in surfaces.items():
        # 每个表面由两个三角形组成，取第一个三角形的法向
        v0, v1, v2 = tris[0], tris[1], tris[2]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-14:
            n = n / n_norm
        normals[name] = n
    return normals


def triangle_area(v0, v1, v2):
    """
    计算三角形面积：A = 0.5 * ||(v1-v0) x (v2-v0)||
    """
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def room_surface_areas(surfaces):
    """
    计算房间各表面的总面积。
    """
    areas = {}
    for name, tris in surfaces.items():
        total = 0.0
        for i in range(0, len(tris), 3):
            total += triangle_area(tris[i], tris[i + 1], tris[i + 2])
        areas[name] = total
    return areas


def room_total_volume():
    """
    计算房间体积（shoebox：10 x 8 x 5 = 400 m³，减去柱子体积）。
    """
    V_room = 10.0 * 8.0 * 5.0
    # 两个圆柱，半径0.3，高5.0
    V_pillars = 2.0 * (np.pi * 0.3 ** 2 * 5.0)
    return V_room - V_pillars


def compute_sabine_reverberation_time(absorption_coeffs, surfaces):
    """
    使用 Sabine 公式计算混响时间：
    T60 = 0.161 * V / Σ(A_i * α_i)
    其中 V 为体积，A_i 为面积，α_i 为吸声系数。
    """
    areas = room_surface_areas(surfaces)
    V = room_total_volume()
    total_absorption = 0.0
    for name, area in areas.items():
        alpha = absorption_coeffs.get(name, 0.05)
        total_absorption += area * alpha
    if total_absorption < 1e-14:
        total_absorption = 1e-14
    T60 = 0.161 * V / total_absorption
    return T60
