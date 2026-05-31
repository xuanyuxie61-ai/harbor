"""
atmospheric_mesh.py
大气球面网格与三角化质量分析模块

整合原项目:
  - 1335_triangulation_delaunay_discrepancy: Delaunay 三角化质量度量
  - 186_cities: 经纬度距离计算 (球面几何)
  - 1422_xyl_display: 点线几何结构处理

功能:
  1. 全球经纬度网格生成与球面距离计算
  2. Delaunay 三角化离散度分析 (用于评估网格质量)
  3. 大气柱分层边界定义 (xyl 几何)
  4. 网格角度质量检查

核心公式:
  - 球面距离 (Haversine):
      d = R * arccos( sin φ1 sin φ2 + cos φ1 cos φ2 cos(Δλ) )
  - 局部 Delaunay 离散度:
      对每对相邻三角形，考虑公共边替换为对角线后最小角的变化量 Δα_min。
      discrepancy = max(0, α_min_new - α_min_old)
      若 discrepancy = 0，则为严格 Delaunay 三角化。
"""

import numpy as np
from math import radians, sin, cos, acos, sqrt, pi, degrees


class MeshError(Exception):
    pass


def ll_degrees_to_distance_earth(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    计算地球表面两点间的球面距离 (km)。

    公式 (球面余弦定理):
      Δσ = arccos( sin φ1 sin φ2 + cos φ1 cos φ2 cos(Δλ) )
      d = R * Δσ

    参数:
      lat1, lon1: 第一点经纬度 (度)
      lat2, lon2: 第二点经纬度 (度)
      radius: 地球半径 (km)

    返回:
      距离 (km)
    """
    phi1 = radians(lat1)
    lambda1 = radians(lon1)
    phi2 = radians(lat2)
    lambda2 = radians(lon2)

    cos_theta = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(lambda1 - lambda2)
    # 截断到 [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = acos(cos_theta)
    return radius * theta


def generate_lat_lon_grid(n_lat, n_lon):
    """
    生成全球均匀经纬度网格节点。

    返回:
      nodes: (n_lat*n_lon, 2) 数组，每行为 (lat, lon)
    """
    if n_lat < 2 or n_lon < 2:
        raise MeshError("generate_lat_lon_grid: 网格维度至少为 2")

    lats = np.linspace(-90.0, 90.0, n_lat)
    lons = np.linspace(-180.0, 180.0, n_lon)
    nodes = []
    for lat in lats:
        for lon in lons:
            nodes.append([lat, lon])
    return np.array(nodes, dtype=np.float64)


def compute_distance_table(nodes):
    """
    计算节点间的球面距离矩阵。

    参数:
      nodes: (N, 2) 经纬度数组

    返回:
      dist: (N, N) 距离矩阵 (km)
    """
    N = nodes.shape[0]
    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = ll_degrees_to_distance_earth(
                nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1]
            )
            dist[i, j] = d
            dist[j, i] = d
    return dist


def triangle_angles_2d(p1, p2, p3):
    """
    计算平面三角形三个内角 (度)。

    使用余弦定理:
      a = ||p2 - p3||, b = ||p1 - p3||, c = ||p1 - p2||
      cos A = (b^2 + c^2 - a^2) / (2bc)
    """
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)

    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        return 0.0, 0.0, 0.0

    cos_a = max(-1.0, min(1.0, (b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c)))
    cos_b = max(-1.0, min(1.0, (a ** 2 + c ** 2 - b ** 2) / (2.0 * a * c)))
    cos_c = max(-1.0, min(1.0, (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)))

    A = degrees(acos(cos_a))
    B = degrees(acos(cos_b))
    C = degrees(acos(cos_c))
    return A, B, C


def delaunay_discrepancy_simple(nodes, triangles):
    """
    简化版 Delaunay 离散度计算。

    对给定的三角形网格，计算最小角的最大可能改进量。
    若返回 0，则为严格 Delaunay 三角化。

    参数:
      nodes: (N, 2) 节点坐标
      triangles: (T, 3) 三角形节点索引

    返回:
      discrepancy: 最大离散度 (度)
      min_angle: 当前最小角 (度)
    """
    num_tri = triangles.shape[0]
    if num_tri == 0:
        return 0.0, 0.0

    angles = []
    for t in range(num_tri):
        i, j, k = triangles[t]
        p1 = nodes[i]
        p2 = nodes[j]
        p3 = nodes[k]
        A, B, C = triangle_angles_2d(p1, p2, p3)
        angles.extend([A, B, C])

    if len(angles) == 0:
        return 0.0, 0.0

    min_angle = min(angles)
    # 简化: 离散度为理论最小角 (60° 等边) 与实际最小角之差的上界估计
    discrepancy = max(0.0, 60.0 - min_angle)
    return discrepancy, min_angle


def define_atmospheric_layers(z_bottom, z_top, num_layers):
    """
    定义大气分层边界 (xyl_display 的几何结构概念)。

    返回:
      boundaries: (num_layers+1,) 高度边界数组
      mid_points: (num_layers,) 层中点高度
    """
    if z_top <= z_bottom or num_layers < 1:
        raise MeshError("define_atmospheric_layers: 参数非法")

    boundaries = np.linspace(z_bottom, z_top, num_layers + 1)
    mid_points = 0.5 * (boundaries[:-1] + boundaries[1:])
    return boundaries, mid_points


def compute_mesh_quality_metrics(nodes, triangles):
    """
    计算网格质量指标:
      - 最小角
      - 最大角
      - 角度标准差
      - Delaunay 离散度
    """
    discrepancy, min_angle = delaunay_discrepancy_simple(nodes, triangles)
    num_tri = triangles.shape[0]
    all_angles = []
    max_angle = 0.0

    for t in range(num_tri):
        i, j, k = triangles[t]
        A, B, C = triangle_angles_2d(nodes[i], nodes[j], nodes[k])
        all_angles.extend([A, B, C])
        max_angle = max(max_angle, A, B, C)

    angle_std = float(np.std(all_angles)) if all_angles else 0.0
    return {
        "delaunay_discrepancy": float(discrepancy),
        "min_angle_deg": float(min_angle),
        "max_angle_deg": float(max_angle),
        "angle_std_deg": angle_std,
        "num_triangles": num_tri,
    }


def generate_simple_triangulation(nodes_2d):
    """
    对矩形排列的 2D 节点生成简单三角化 (用于测试)。
    假设节点按行优先排列。
    """
    n = int(round(np.sqrt(nodes_2d.shape[0])))
    if n * n != nodes_2d.shape[0]:
        # 非完全平方，返回空
        return np.zeros((0, 3), dtype=int)

    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            p00 = i * n + j
            p10 = (i + 1) * n + j
            p01 = i * n + (j + 1)
            p11 = (i + 1) * n + (j + 1)
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])

    return np.array(triangles, dtype=int)
