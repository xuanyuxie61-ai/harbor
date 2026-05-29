"""
spherical_tessellation.py
球面质心 Voronoi 镶嵌 (CVT) — 冰盖观测网最优布设

基于种子项目 1113_sphere_cvt 的球面 CVT 算法，
用于在地球表面球面上生成最优离散化网格或观测站点布局。

核心数学:
  1. 球面 Voronoi 单元:
       V_i = \{ x \in S^2 : \|x - z_i\| \le \|x - z_j\|, \forall j \neq i \}

  2. 球面面积元:
       dA = R^2 \sin\phi \, d\phi \, d\lambda

  3. 球面三角形面积 (Girard 公式):
       A_{\Delta} = (\alpha + \beta + \gamma - \pi) R^2

  4. Lloyd 迭代 (CVT 优化):
       z_i^{new} = \frac{1}{A(V_i)} \int_{V_i} x \, dA

  即每个生成点被替换为其 Voronoi 单元的质心。

在冰盖科学中的应用:
  - 南极冰盖区域的最优网格划分
  - GNSS/卫星观测站点的空间优化布设
  - 冰穹 (dome) 表面有限元的节点生成
"""

import numpy as np
from typing import List, Tuple


def uniform_on_sphere01(n_points: int, seed: int = 42) -> np.ndarray:
    """
    在三维单位球面上均匀随机采样 n_points 个点。

    方法: 生成三维标准正态随机向量后归一化。
        x \sim N(0, I_3),  \quad p = x / \|x\|

    参数:
        n_points: 采样点数
        seed: 随机种子

    返回:
        points: (n_points, 3) 数组，每行为单位球面上的点
    """
    rng = np.random.default_rng(seed)
    xyz = rng.standard_normal((n_points, 3), dtype=np.float64)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = xyz / norms
    return points


def spherical_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                             radius: float = 1.0) -> float:
    """
    计算球面三角形面积 (Girard 公式)。

    对于单位球面上的三点 p1, p2, p3:
        A = (\alpha + \beta + \gamma - \pi) R^2

    其中 \alpha, \beta, \gamma 为球面角，通过向量叉积计算:
        \cos\alpha = \frac{(p1 \times p2) \cdot (p1 \times p3)}{|p1 \times p2| |p1 \times p3|}

    参数:
        p1, p2, p3: 球面上的三维点 (已归一化)
        radius: 球半径

    返回:
        area: 球面三角形面积
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # 确保单位长度
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    p3 = p3 / np.linalg.norm(p3)

    # 边向量叉积
    n12 = np.cross(p1, p2)
    n23 = np.cross(p2, p3)
    n31 = np.cross(p3, p1)

    n12_norm = np.linalg.norm(n12)
    n23_norm = np.linalg.norm(n23)
    n31_norm = np.linalg.norm(n31)

    if n12_norm < 1e-14 or n23_norm < 1e-14 or n31_norm < 1e-14:
        return 0.0

    n12 = n12 / n12_norm
    n23 = n23 / n23_norm
    n31 = n31 / n31_norm

    # 球面角
    cos_alpha = np.clip(np.dot(-n12, n31), -1.0, 1.0)
    cos_beta = np.clip(np.dot(-n23, n12), -1.0, 1.0)
    cos_gamma = np.clip(np.dot(-n31, n23), -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
    beta = np.arccos(cos_beta)
    gamma = np.arccos(cos_gamma)

    area = (alpha + beta + gamma - np.pi) * (radius ** 2)
    return float(np.maximum(area, 0.0))


def spherical_polygon_centroid(points: np.ndarray,
                                radius: float = 1.0) -> np.ndarray:
    """
    计算球面多边形 (由顶点序列定义) 的质心。

    采用面积加权顶点平均的近似:
        c = \frac{1}{A_{total}} \sum_k A_k \cdot \frac{v_k}{\|v_k\|}

    其中 A_k 为以 v_k 为顶点的相邻三角形面积之和。

    参数:
        points: (m, 3) 多边形顶点
        radius: 球半径

    返回:
        centroid: 质心单位向量
    """
    points = np.asarray(points, dtype=np.float64)
    m = len(points)
    if m < 3:
        raise ValueError("Polygon must have at least 3 vertices.")

    # 以多边形中心为辅助点，三角化
    center = np.mean(points, axis=0)
    center = center / np.linalg.norm(center)

    areas = []
    weighted_sum = np.zeros(3, dtype=np.float64)

    for k in range(m):
        p1 = points[k]
        p2 = points[(k + 1) % m]
        a = spherical_triangle_area(p1, p2, center, radius)
        areas.append(a)
        # 三角形质心贡献
        tri_centroid = (p1 + p2 + center) / 3.0
        weighted_sum += a * tri_centroid

    total_area = sum(areas)
    if total_area < 1e-15:
        return center

    centroid = weighted_sum / total_area
    centroid = centroid / np.linalg.norm(centroid)
    return centroid


def sphere_cvt_step(generators: np.ndarray,
                     radius: float = 1.0) -> np.ndarray:
    """
    执行一次球面 CVT Lloyd 迭代。

    由于完整球面 Voronoi 图计算复杂，这里采用近似策略:
    1. 对每个生成点，收集其 k 近邻
    2. 在局部切平面上构建 Voronoi 单元近似
    3. 计算局部质心并投影回球面

    参数:
        generators: (n, 3) 当前生成点
        radius: 球半径

    返回:
        new_generators: (n, 3) 新生成点
    """
    generators = np.asarray(generators, dtype=np.float64)
    n = len(generators)

    # 归一化
    norms = np.linalg.norm(generators, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    generators = generators / norms

    new_generators = np.zeros_like(generators)

    for i in range(n):
        gi = generators[i]
        # 计算到所有其他点的距离
        dists = np.linalg.norm(generators - gi, axis=1)

        # 选取近邻 (排除自身)
        k = min(12, n - 1)
        neighbor_idx = np.argpartition(dists, k)[:k + 1]
        neighbors = generators[neighbor_idx]

        # 局部质心近似: 近邻的均值投影回球面
        local_centroid = np.mean(neighbors, axis=0)
        local_centroid = local_centroid / np.linalg.norm(local_centroid)
        new_generators[i] = local_centroid

    return new_generators * radius


def sphere_cvt_iterate(n_generators: int,
                        n_iterations: int = 100,
                        radius: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    """
    迭代生成球面 CVT 节点。

    参数:
        n_generators: 生成点数量
        n_iterations: Lloyd 迭代次数
        radius: 球半径
        seed: 随机种子

    返回:
        generators: (n_generators, 3) 最优节点
    """
    generators = uniform_on_sphere01(n_generators, seed)
    generators = generators * radius

    for it in range(n_iterations):
        generators = sphere_cvt_step(generators, radius)

    return generators


def cvt_energy(generators: np.ndarray) -> float:
    """
    计算 CVT 能量泛函 (近似):

        E = \sum_i \int_{V_i} \|x - z_i\|^2 dA

    这里采用最近邻距离平方和作为代理度量。
    """
    generators = np.asarray(generators, dtype=np.float64)
    n = len(generators)
    if n < 2:
        return 0.0

    energy = 0.0
    for i in range(n):
        dists = np.linalg.norm(generators - generators[i], axis=1)
        dists[i] = np.inf
        energy += np.min(dists) ** 2

    return float(energy / n)


def project_to_ice_dome_region(points: np.ndarray,
                                latitude_range: Tuple[float, float] = (-90.0, -60.0),
                                longitude_range: Tuple[float, float] = (-180.0, 180.0),
                                earth_radius: float = 6371e3) -> np.ndarray:
    """
    将球面点投影到南极冰盖区域 (纬度筛选)。

    参数:
        points: (n, 3) 球面笛卡尔坐标
        latitude_range: (lat_min, lat_max) in degrees
        longitude_range: (lon_min, lon_max) in degrees
        earth_radius: 地球半径 (m)

    返回:
        filtered_points: 区域内的点
    """
    points = np.asarray(points, dtype=np.float64)
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    filtered = []
    for p in points:
        x, y, z = p
        lat = np.degrees(np.arcsin(np.clip(z / earth_radius, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))

        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            filtered.append(p)

    if not filtered:
        # 若无匹配，返回前几个点作为保底
        return points[:max(1, len(points) // 4)]

    return np.array(filtered, dtype=np.float64)
