"""
pbpk_geometry.py
基于种子项目 823_obj_to_tri_surface + 1378_usa_cvt_geo

实现器官几何建模：
1. 椭球/参数化曲面的三角剖分（fan triangulation）
2. Centroidal Voronoi Tessellation（CVT）在器官内生成最优采样点
3. 器官体积、表面积计算
4. 多器官空间配置（肝、肾、心、肺、脑、肌肉、脂肪、肿瘤等）

在 PBPK 模型中用于：
- 构建各器官的 3D 几何近似
- CVT 采样用于 Monte Carlo 积分和空间离散
- 器官间距离计算与血流连接拓扑
"""

import numpy as np
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# 三角剖分与网格处理（基于 OBJ fan triangulation）
# ---------------------------------------------------------------------------

def ellipsoid_tri_surface(a: float, b: float, c: float,
                           n_theta: int = 20, n_phi: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 fan triangulation 生成椭球表面的三角网格。
    返回：
        vertices : (N_v, 3) 顶点数组
        triangles : (N_t, 3) 三角形索引数组
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Ellipsoid axes must be positive")
    if n_theta < 3 or n_phi < 2:
        raise ValueError("Grid resolution too low")

    vertices = []
    # 顶点生成（球坐标参数化）
    for j in range(n_phi + 1):
        phi = np.pi * j / n_phi
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / n_theta
            x = a * np.sin(phi) * np.cos(theta)
            y = b * np.sin(phi) * np.sin(theta)
            z = c * np.cos(phi)
            vertices.append([x, y, z])
    vertices = np.array(vertices)

    triangles = []
    # Fan triangulation
    for j in range(n_phi):
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            v0 = j * n_theta + i
            v1 = j * n_theta + i_next
            v2 = (j + 1) * n_theta + i_next
            v3 = (j + 1) * n_theta + i
            triangles.append([v0, v1, v3])
            triangles.append([v1, v2, v3])
    triangles = np.array(triangles)
    return vertices, triangles


def mesh_surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """
    计算三角网格的表面积。
    """
    if triangles.size == 0:
        return 0.0
    area = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        area += 0.5 * np.linalg.norm(cross)
    return area


def mesh_volume_tetrahedral(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """
    使用原点-三角形四面体分解估算封闭曲面的体积。
    """
    if triangles.size == 0:
        return 0.0
    vol = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        vol += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(vol)


# ---------------------------------------------------------------------------
# Centroidal Voronoi Tessellation（CVT）
# ---------------------------------------------------------------------------

def inside_ellipsoid_points(points: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """判断点是否在椭球内。"""
    return (points[:, 0] / a) ** 2 + (points[:, 1] / b) ** 2 + (points[:, 2] / c) ** 2 <= 1.0


def cvt_ellipsoid_lloyd(a: float, b: float, c: float,
                         n_generators: int, n_samples: int = 10000,
                         n_iterations: int = 10, seed: int = 42) -> np.ndarray:
    """
    使用 Lloyd 算法在椭球内生成 Centroidal Voronoi Tessellation 采样点。
    算法：
        1. 在包围盒内拒绝采样 n_samples 个均匀点
        2. 将每个点分配到最近的 generator
        3. 用每个 cell 内样本的质心更新 generator 位置
        4. 迭代 n_iterations 次

    在 PBPK 中用于在器官内生成空间代表性的药物浓度采样点。
    """
    if a <= 0 or b <= 0 or c <= 0 or n_generators < 1:
        raise ValueError("Invalid CVT parameters")
    rng = np.random.default_rng(seed)
    # 初始化 generators（均匀随机在椭球内）
    generators = np.empty((n_generators, 3))
    count = 0
    while count < n_generators:
        pts = rng.uniform([-a, -b, -c], [a, b, c], size=(n_generators * 2, 3))
        mask = inside_ellipsoid_points(pts, a, b, c)
        valid = pts[mask]
        need = n_generators - count
        take = min(need, len(valid))
        generators[count:count + take] = valid[:take]
        count += take

    # Lloyd 迭代
    for it in range(n_iterations):
        # 拒绝采样
        samples = rng.uniform([-a, -b, -c], [a, b, c], size=(n_samples, 3))
        mask = inside_ellipsoid_points(samples, a, b, c)
        samples = samples[mask]
        if len(samples) == 0:
            continue
        # 最近邻分配
        dists = np.linalg.norm(samples[:, None, :] - generators[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        # 更新质心
        new_generators = np.empty_like(generators)
        for k in range(n_generators):
            cell_pts = samples[nearest == k]
            if len(cell_pts) > 0:
                new_generators[k] = cell_pts.mean(axis=0)
            else:
                new_generators[k] = generators[k]
        generators = new_generators
    return generators


# ---------------------------------------------------------------------------
# 多器官 PBPK 几何配置
# ---------------------------------------------------------------------------

ORGAN_DEFS: Dict[str, dict] = {
    "liver":   {"axes": (0.18, 0.12, 0.10), "blood_flow_frac": 0.25, "Kp": 2.5},
    "kidney":  {"axes": (0.06, 0.04, 0.04), "blood_flow_frac": 0.20, "Kp": 1.2},
    "heart":   {"axes": (0.05, 0.04, 0.04), "blood_flow_frac": 0.05, "Kp": 1.8},
    "lung":    {"axes": (0.12, 0.10, 0.08), "blood_flow_frac": 1.00, "Kp": 1.0},
    "brain":   {"axes": (0.07, 0.06, 0.06), "blood_flow_frac": 0.12, "Kp": 0.5},
    "muscle":  {"axes": (0.25, 0.15, 0.10), "blood_flow_frac": 0.15, "Kp": 0.7},
    "adipose": {"axes": (0.20, 0.15, 0.10), "blood_flow_frac": 0.05, "Kp": 5.0},
    "tumor":   {"axes": (0.03, 0.02, 0.02), "blood_flow_frac": 0.02, "Kp": 1.5},
}


def build_organ_geometries(n_cvt_points: int = 200) -> Dict[str, dict]:
    """
    为所有器官构建几何模型（椭球近似）和 CVT 采样点。
    返回字典，每个器官包含：vertices, triangles, surface_area, volume, cvt_points。
    """
    geometries = {}
    for name, defs in ORGAN_DEFS.items():
        a, b, c = defs["axes"]
        vertices, triangles = ellipsoid_tri_surface(a, b, c, n_theta=16, n_phi=12)
        sa = mesh_surface_area(vertices, triangles)
        vol = mesh_volume_tetrahedral(vertices, triangles)
        cvt_pts = cvt_ellipsoid_lloyd(a, b, c, n_cvt_points, n_samples=20000, n_iterations=5)
        geometries[name] = {
            "axes": (a, b, c),
            "vertices": vertices,
            "triangles": triangles,
            "surface_area": sa,
            "volume": vol,
            "cvt_points": cvt_pts,
            "blood_flow_frac": defs["blood_flow_frac"],
            "Kp": defs["Kp"],
        }
    return geometries


def compute_organ_distances(geometries: Dict[str, dict]) -> np.ndarray:
    """
    计算各器官质心之间的欧氏距离矩阵（m）。
    用于评估药物在器官间扩散的时间尺度。
    """
    names = list(geometries.keys())
    n = len(names)
    D = np.zeros((n, n))
    centers = []
    for name in names:
        # 质心近似为原点（椭球中心）
        centers.append(np.array([0.0, 0.0, 0.0]))
    # 这里简化为固定空间位置
    fixed_positions = {
        "liver":   np.array([0.05, 0.10, 0.05]),
        "kidney":  np.array([0.15, 0.08, 0.04]),
        "heart":   np.array([0.08, 0.12, 0.08]),
        "lung":    np.array([0.08, 0.18, 0.10]),
        "brain":   np.array([0.08, 0.25, 0.10]),
        "muscle":  np.array([0.30, 0.05, 0.05]),
        "adipose": np.array([0.25, 0.15, 0.03]),
        "tumor":   np.array([0.06, 0.09, 0.04]),
    }
    for i in range(n):
        for j in range(i + 1, n):
            pi = fixed_positions[names[i]]
            pj = fixed_positions[names[j]]
            D[i, j] = D[j, i] = np.linalg.norm(pi - pj)
    return D


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    v, t = ellipsoid_tri_surface(0.1, 0.08, 0.06)
    print(f"Ellipsoid mesh: {len(v)} vertices, {len(t)} triangles")
    sa = mesh_surface_area(v, t)
    vol = mesh_volume_tetrahedral(v, t)
    print(f"Surface area: {sa:.6f} m^2, Volume: {vol:.6f} m^3")
    # 理论值对比
    a, b, c = 0.1, 0.08, 0.06
    sa_exact = 4 * np.pi * ((a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6)**(1/1.6) / 3  # 近似
    vol_exact = 4/3 * np.pi * a * b * c
    print(f"Approx exact volume: {vol_exact:.6f}")
    cvt = cvt_ellipsoid_lloyd(0.1, 0.08, 0.06, 50, n_samples=5000, n_iterations=5)
    print(f"CVT generators: {len(cvt)}")
    geoms = build_organ_geometries(n_cvt_points=100)
    print(f"Organs built: {list(geoms.keys())}")
    D = compute_organ_distances(geoms)
    print(f"Distance matrix shape: {D.shape}")
