"""
asteroid_geometry.py

基于 triangulate (耳切法多边形三角剖分) 与 sierpinski_carpet_chaos (IFS 分形)
核心算法，构建不规则小行星的三维形状模型与表面网格。

科学背景：
小行星（尤其是 C 型、S 型碎石堆）形状高度不规则。
本项目使用 IFS (Iterated Function System) 迭代函数系统生成表面粗糙度，
再通过耳切三角剖分将二维截面映射为三维多面体，用于多面体引力场计算。

数学模型：
- IFS 映射: x_{k+1} = A x_k + b_j,  j ∈ {0,...,7}
- 多面体体积由三角剖分后累加计算:
    V = Σ_{triangles} (1/6) | (v1 × v2) · v3 |
- 质心:
    r_c = (1/V) Σ (1/24) |det| (v1+v2+v3)
"""

import numpy as np
from typing import List, Tuple, Optional


class AsteroidGeometryError(Exception):
    pass


def sierpinski_ifs_transforms(scale: float = 1.0 / 3.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成 Sierpinski Carpet 的 8 个 IFS 仿射变换 (A, b)。
    在小行星建模中，这些变换用于生成分形表面纹理/粗糙度。
    """
    A = np.array([[scale, 0.0], [0.0, scale]])
    translations = [
        np.array([0.0, 2.0 * scale]),
        np.array([scale, 2.0 * scale]),
        np.array([2.0 * scale, 2.0 * scale]),
        np.array([0.0, scale]),
        np.array([2.0 * scale, scale]),
        np.array([0.0, 0.0]),
        np.array([scale, 0.0]),
        np.array([2.0 * scale, 0.0]),
    ]
    return [(A.copy(), b) for b in translations]


def generate_fractal_profile(
    n_iterations: int = 5000,
    scale: float = 1.0 / 3.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    使用 IFS 迭代生成分形表面轮廓点集（二维）。
    这些点经过缩放后作为小行星表面高度图的基础。

    参数:
        n_iterations: 迭代次数，决定采样密度。
        scale: IFS 收缩比例。
        seed: 随机种子，保证可复现。

    返回:
        points: shape (n_iterations, 2) 的数组。
    """
    if seed is not None:
        np.random.seed(seed)

    transforms = sierpinski_ifs_transforms(scale)
    x = np.random.rand(2)
    points = np.zeros((n_iterations, 2))

    # 前 100 次作为 burn-in
    for _ in range(100):
        idx = np.random.randint(0, 8)
        A, b = transforms[idx]
        x = A @ x + b

    for i in range(n_iterations):
        idx = np.random.randint(0, 8)
        A, b = transforms[idx]
        x = A @ x + b
        points[i] = x.copy()

    return points


def polygon_area_2d(vertices: np.ndarray) -> float:
    """
    计算二维多边形的有向面积（耳切法中的 area_poly2）。
    使用鞋带公式:
        A = 0.5 Σ_{i} (x_i y_{i+1} − x_{i+1} y_i)
    """
    n = vertices.shape[0]
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
    return 0.5 * area


def is_collinear(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, eps: float = 1e-10) -> bool:
    """
    判断三点是否近似共线（triangulate 中的 collinear 判定）。
    计算三角形面积与最大边长平方的比值。
    """
    area = 0.5 * abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    )
    d12 = np.sum((v1 - v2) ** 2)
    d23 = np.sum((v2 - v3) ** 2)
    d31 = np.sum((v3 - v1) ** 2)
    area_max = max(d12, d23, d31)
    if area_max <= eps:
        return True
    return 2.0 * area <= eps * area_max


def is_convex_vertex(poly: np.ndarray, i: int) -> bool:
    """
    判断多边形第 i 个顶点是否为凸顶点（用于耳切法）。
    对于逆时针排列的简单多边形，凸顶点的叉积 > 0。
    """
    n = poly.shape[0]
    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    cross = (poly[i, 0] - poly[im1, 0]) * (poly[ip1, 1] - poly[i, 1]) - \
            (poly[i, 1] - poly[im1, 1]) * (poly[ip1, 0] - poly[i, 0])
    return cross > 0  # 逆时针排列的凸顶点


def point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    判断点 p 是否严格在三角形 abc 内部（用于耳切法的对角线检查）。
    使用重心坐标法。
    """
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-14:
        return False
    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    gamma = 1.0 - alpha - beta
    return (alpha > 1e-10) and (beta > 1e-10) and (gamma > 1e-10)


def ear_clip_triangulation(poly: np.ndarray) -> np.ndarray:
    """
    耳切法 (Ear Clipping) 实现简单多边形的三角剖分。
    基于 triangulate.m 的核心算法思想。

    输入 poly: (n, 2) 逆时针排列的二维顶点。
    返回 triangles: (n-2, 3) 的顶点索引数组。
    """
    n = poly.shape[0]
    if n < 3:
        raise AsteroidGeometryError("多边形至少需要 3 个顶点")
    if n == 3:
        return np.array([[0, 1, 2]], dtype=int)

    # 确保逆时针
    if polygon_area_2d(poly) < 0:
        poly = poly[::-1]

    # 检查是否有重复顶点
    for i in range(n):
        j = (i + 1) % n
        if np.allclose(poly[i], poly[j]):
            raise AsteroidGeometryError(f"顶点 {i} 与 {j} 重合")

    indices = list(range(n))
    triangles = []

    while len(indices) > 3:
        n_cur = len(indices)
        ear_found = False
        for i in range(n_cur):
            im1 = (i - 1) % n_cur
            ip1 = (i + 1) % n_cur
            idx_i = indices[i]
            idx_im1 = indices[im1]
            idx_ip1 = indices[ip1]

            v_im1 = poly[idx_im1]
            v_i = poly[idx_i]
            v_ip1 = poly[idx_ip1]

            if is_collinear(v_im1, v_i, v_ip1):
                continue
            if not is_convex_vertex(poly[np.array(indices)], i):
                continue

            # 检查是否有其他顶点在三角形内
            valid = True
            for j in range(n_cur):
                if j in (im1, i, ip1):
                    continue
                if point_in_triangle_2d(poly[indices[j]], v_im1, v_i, v_ip1):
                    valid = False
                    break
            if valid:
                triangles.append([idx_im1, idx_i, idx_ip1])
                del indices[i]
                ear_found = True
                break

        if not ear_found:
            # 退化情况：移除一个近似共线的顶点
            for i in range(n_cur):
                im1 = (i - 1) % n_cur
                ip1 = (i + 1) % n_cur
                if is_collinear(poly[indices[im1]], poly[indices[i]], poly[indices[ip1]]):
                    del indices[i]
                    break
            else:
                raise AsteroidGeometryError("无法完成三角剖分，可能为自相交多边形")

    triangles.append([indices[0], indices[1], indices[2]])
    return np.array(triangles, dtype=int)


def generate_asteroid_cross_section(
    a: float = 2.0,
    b: float = 1.5,
    c: float = 1.0,
    n_points: int = 64,
    roughness_amplitude: float = 0.05,
    roughness_scale: float = 1.0 / 3.0,
    seed: int = 42
) -> np.ndarray:
    """
    生成小行星的二维截面轮廓（椭圆+分形扰动）。
    使用 IFS 分形扰动模拟表面粗糙度。

    参数:
        a, b: 椭圆半轴
        c: 第三轴（用于后续拉伸为三维）
        n_points: 圆周采样点数
        roughness_amplitude: 粗糙度振幅
        roughness_scale: IFS 尺度
        seed: 随机种子

    返回:
        vertices: (n_points, 2) 的轮廓点，按角度排序确保为星形简单多边形
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    # 基础椭圆
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    # 添加分形粗糙度：先生成分形序列，再平滑（移动平均）以避免自相交
    fractal = generate_fractal_profile(n_iterations=n_points, scale=roughness_scale, seed=seed)
    np.random.seed(seed)
    raw_noise = 2.0 * fractal[:, 0] - 1.0
    # 3点移动平均平滑，保证星形性质
    window = 3
    pad = window // 2
    padded = np.concatenate([raw_noise[-pad:], raw_noise, raw_noise[:pad]])
    smooth_noise = np.convolve(padded, np.ones(window) / window, mode='valid')
    radial_noise = roughness_amplitude * smooth_noise

    r_base = np.sqrt(x ** 2 + y ** 2)
    r_perturbed = r_base * (1.0 + radial_noise)
    # 严格保持原始角度，确保星形多边形（从原点出发的射线与边界唯一相交）
    x_pert = r_perturbed * np.cos(theta)
    y_pert = r_perturbed * np.sin(theta)

    vertices = np.column_stack((x_pert, y_pert))
    return vertices


def revolve_to_3d(poly2d: np.ndarray, z_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将二维截面绕 z 轴旋转生成三维多面体顶点与三角面片。
    使用经度分割。

    返回:
        vertices: (n_vertices, 3)
        faces: (n_faces, 3) 顶点索引
    """
    n_lat = poly2d.shape[0]
    n_lon = max(24, n_lat // 2)

    vertices = []
    for j in range(n_lon):
        phi = 2.0 * np.pi * j / n_lon
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        for i in range(n_lat):
            r = np.sqrt(poly2d[i, 0] ** 2 + poly2d[i, 1] ** 2)
            # 取与 x 轴的夹角作为纬度角
            lat_angle = np.arctan2(poly2d[i, 1], poly2d[i, 0])
            x = r * np.cos(lat_angle) * cos_phi
            y = r * np.cos(lat_angle) * sin_phi
            z = r * np.sin(lat_angle) * z_scale
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        for i in range(n_lat - 1):
            v0 = j * n_lat + i
            v1 = j * n_lat + i + 1
            v2 = j_next * n_lat + i + 1
            v3 = j_next * n_lat + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces, dtype=int)


def polyhedron_volume_and_com(vertices: np.ndarray, faces: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算多面体的体积与质心。
    基于四面体累加法：
        V = Σ (1/6) | (v1 − v0) · ((v2 − v0) × (v3 − v0)) |
        r_com = (1/V) Σ (1/24) |det| (v0+v1+v2+v3)/4
    为简化，假设多面体为封闭曲面，原点在外部亦可（有向体积累加）。
    """
    vol = 0.0
    com = np.zeros(3)
    for f in faces:
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        # 以原点为参考的四面体有向体积
        tet_vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        vol += tet_vol
        com += tet_vol * (v0 + v1 + v2) / 4.0

    if abs(vol) < 1e-14:
        return 0.0, np.zeros(3)
    com /= vol
    return abs(vol), com


def triangle_area_3d(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    计算三维空间三角形面积。
        A = 0.5 | (v2 − v1) × (v3 − v1) |
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    计算多面体总表面积。
    """
    area = 0.0
    for f in faces:
        area += triangle_area_3d(vertices[f[0]], vertices[f[1]], vertices[f[2]])
    return area
