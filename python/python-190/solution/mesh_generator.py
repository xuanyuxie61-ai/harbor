"""
mesh_generator.py
=================
基于种子项目 548_human_mesh2d 的三角网格生成模块。
在复杂边界（人类轮廓/任意多边形）内部生成 Delaunay 三角剖分，
用于物理信息 GAN 中在二维截面上计算 PDE 残差积分。

核心数学：
  Delaunay 三角剖分：对于平面点集 P，Delaunay 三角剖分 DT(P) 满足
    空外接圆条件：DT(P) 中任意三角形的外接圆内部不含 P 中其他点。
  这等价于在所有三角剖分中最大化最小角，从而保证网格质量。

  Bowyer-Watson 增量算法：
    1. 构建一个包含所有点的超级三角形。
    2. 逐点插入：找出所有外接圆包含该点的三角形（bad triangles）。
    3. 移除 bad triangles，留下一个多边形空洞。
    4. 将插入点与空洞边界所有顶点连接，形成新三角形。
    5. 最后移除与超级三角形共享顶点的所有三角形。
"""

import numpy as np


def circumcircle_center(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple:
    """
    计算三角形外接圆圆心与半径。

    Parameters
    ----------
    p1, p2, p3 : np.ndarray, shape (2,)
        三角形顶点。

    Returns
    -------
    center : np.ndarray, shape (2,)
        外接圆圆心。
    radius : float
        外接圆半径。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    d = 2.0 * ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))
    if abs(d) < 1e-14:
        # 三点共线，返回中点与极大半径
        center = (p1 + p2 + p3) / 3.0
        return center, 1e12

    ux = ((p1[0] ** 2 + p1[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p2[1] - p3[1])
          - (p2[0] ** 2 + p2[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p1[1] - p3[1]))
    uy = ((p2[0] ** 2 + p2[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p1[0] - p3[0])
          - (p1[0] ** 2 + p1[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p2[0] - p3[0]))
    center = np.array([ux / d, uy / d])
    radius = np.linalg.norm(center - p1)
    return center, radius


def point_in_circumcircle(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray) -> bool:
    """判断点 pt 是否在三角形 (p1,p2,p3) 的外接圆内（含边界）。"""
    center, radius = circumcircle_center(p1, p2, p3)
    return np.linalg.norm(pt - center) <= radius + 1e-12


def bowyer_watson(points: np.ndarray) -> list:
    """
    Bowyer-Watson 算法实现平面 Delaunay 三角剖分。

    Parameters
    ----------
    points : np.ndarray, shape (n, 2)
        平面点集。

    Returns
    -------
    triangles : list of tuple
        每个元素为顶点索引三元组 (i, j, k)。
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("至少需要 3 个点才能进行三角剖分。")

    # 计算超级三角形
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    dmax = max(dx, dy)
    xmid = (xmin + xmax) * 0.5
    ymid = (ymin + ymax) * 0.5

    # 超级三角形的三个顶点
    p_super = np.array([
        [xmid - 20.0 * dmax, ymid - 10.0 * dmax],
        [xmid, ymid + 20.0 * dmax],
        [xmid + 20.0 * dmax, ymid - 10.0 * dmax],
    ])
    pts_all = np.vstack([pts, p_super])
    super_idx = [n, n + 1, n + 2]

    triangles = [[super_idx[0], super_idx[1], super_idx[2]]]

    for i in range(n):
        pt = pts_all[i]
        bad_triangles = []
        for tri in triangles:
            if point_in_circumcircle(pt, pts_all[tri[0]], pts_all[tri[1]], pts_all[tri[2]]):
                bad_triangles.append(tri)

        # 收集空洞边界边（只出现一次的边）
        polygon = []
        for tri in bad_triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                shared = False
                for other in bad_triangles:
                    if other is tri:
                        continue
                    oe = [(other[0], other[1]), (other[1], other[2]), (other[2], other[0])]
                    # 检查是否为同一条边（方向无关）
                    if (e[0], e[1]) in oe or (e[1], e[0]) in oe:
                        shared = True
                        break
                if not shared:
                    polygon.append(e)

        # 移除 bad triangles
        for tri in bad_triangles:
            triangles.remove(tri)

        # 将新点与空洞边界连接
        for e in polygon:
            triangles.append([e[0], e[1], i])

    # 移除与超级三角形共享顶点的三角形
    final_triangles = []
    for tri in triangles:
        if super_idx[0] not in tri and super_idx[1] not in tri and super_idx[2] not in tri:
            final_triangles.append(tuple(tri))

    return final_triangles


def generate_mesh_from_boundary(boundary: np.ndarray, hmax: float = 0.25) -> tuple:
    """
    在封闭多边形边界内部生成带内部点的 Delaunay 三角剖分。

    Parameters
    ----------
    boundary : np.ndarray, shape (m, 2)
        按逆时针排列的边界顶点。
    hmax : float
        期望的最大边长（近似控制网格密度）。

    Returns
    -------
    nodes : np.ndarray, shape (N, 2)
        网格节点坐标。
    triangles : list of tuple
        三角形索引三元组。
    """
    boundary = np.asarray(boundary, dtype=float)
    # 计算边界包围盒，在内部均匀撒点
    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)
    # 在内部生成规则网格点，密度由 hmax 控制
    nx = max(3, int(np.ceil((xmax - xmin) / hmax)) + 1)
    ny = max(3, int(np.ceil((ymax - ymin) / hmax)) + 1)
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    candidates = np.column_stack([Xg.ravel(), Yg.ravel()])

    # 判断点是否在多边形内部（射线法）
    def point_in_polygon(pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # 检查边是否与从 pt 向右的水平射线相交
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if xinters > x:
                    inside = not inside
        return inside

    interior_points = []
    for pt in candidates:
        if point_in_polygon(pt, boundary):
            interior_points.append(pt)

    # 合并边界点与内部点（边界点在前，保证边界被保留）
    all_points = np.vstack([boundary, np.array(interior_points)])
    # 去重
    all_points = np.unique(np.round(all_points, 12), axis=0)
    triangles = bowyer_watson(all_points)
    return all_points, triangles


def human_outline_boundary(scale: float = 1.0) -> np.ndarray:
    """
    基于种子项目 547_human_data 的人类轮廓概念，生成一个简化的人体轮廓多边形。
    使用参数化曲线构造：头部圆弧 + 躯干 + 腿部轮廓。

    Parameters
    ----------
    scale : float
        轮廓缩放因子。

    Returns
    -------
    boundary : np.ndarray, shape (m, 2)
        人体轮廓边界点（逆时针）。
    """
    pts = []
    # 头部：上半圆弧，参数 θ ∈ [π/2, -π/2]
    for theta in np.linspace(np.pi * 0.5, -np.pi * 0.5, 15):
        x = 0.5 * np.cos(theta)
        y = 2.5 + 0.5 * np.sin(theta)
        pts.append([x, y])
    # 右肩到躯干右侧
    pts.append([0.6, 2.0])
    pts.append([0.55, 1.2])
    pts.append([0.5, 0.5])
    # 右腿外侧
    pts.append([0.45, 0.0])
    pts.append([0.3, -0.8])
    pts.append([0.2, -1.5])
    pts.append([0.15, -2.0])
    # 脚底
    pts.append([0.0, -2.2])
    # 左腿外侧（对称返回）
    pts.append([-0.15, -2.0])
    pts.append([-0.2, -1.5])
    pts.append([-0.3, -0.8])
    pts.append([-0.45, 0.0])
    pts.append([-0.5, 0.5])
    pts.append([-0.55, 1.2])
    pts.append([-0.6, 2.0])
    boundary = np.array(pts, dtype=float) * scale
    return boundary


def get_triangle_vertices(nodes: np.ndarray, triangles: list) -> list:
    """
    将三角形索引列表转换为顶点坐标列表。

    Returns
    -------
    verts : list of tuple
        每个元素为 (v1, v2, v3)，vi 为 shape (2,) 的 np.ndarray。
    """
    verts = []
    for tri in triangles:
        verts.append((nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]))
    return verts


def mesh_quality_stats(nodes: np.ndarray, triangles: list) -> dict:
    """
    计算三角网格质量统计量（最小角、面积比等）。

    Returns
    -------
    stats : dict
        包含 'min_angle_deg', 'max_area', 'min_area', 'area_ratio'。
    """
    angles = []
    areas = []
    for tri in triangles:
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        # 使用余弦定理计算角
        def angle_from_sides(x, y, z):
            num = x * x + y * y - z * z
            den = 2.0 * x * y
            if den < 1e-14:
                return 0.0
            return np.arccos(np.clip(num / den, -1.0, 1.0))
        angles.append(angle_from_sides(b, c, a))
        angles.append(angle_from_sides(a, c, b))
        angles.append(angle_from_sides(a, b, c))
        # 面积
        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        area_sq = max(area_sq, 0.0)
        areas.append(np.sqrt(area_sq))

    angles_deg = np.degrees(np.array(angles))
    areas_arr = np.array(areas)
    return {
        "min_angle_deg": float(np.min(angles_deg)),
        "max_area": float(np.max(areas_arr)),
        "min_area": float(np.min(areas_arr)),
        "area_ratio": float(np.max(areas_arr) / (np.min(areas_arr) + 1e-15)),
        "num_triangles": len(triangles),
        "num_nodes": nodes.shape[0],
    }
