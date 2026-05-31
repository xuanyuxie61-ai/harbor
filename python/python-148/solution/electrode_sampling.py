"""
electrode_sampling.py — 最优电极空间采样与皮层曲面几何
=========================================================
融合 test_triangulation（三角剖分/CVT/点定位）与 xyz_display
（3D 坐标几何）两个项目的核心算法。

科学背景：
脑机接口微电极阵列（MEA）的空间布局直接影响信号解码性能。
最优电极位置应最大化空间覆盖同时最小化冗余，
这对应于 Voronoi 镶嵌的质心（Centroidal Voronoi Tessellation, CVT）。

核心数学：
---
**CVT 最优性条件：**
对区域 Ω 上的密度函数 ρ(r)，生成点集 {z_i} 的 Voronoi 单元 V_i 满足：

    z_i = ∫_{V_i} r ρ(r) dr / ∫_{V_i} ρ(r) dr

即生成点位于其 Voronoi 单元的加权质心。
CVT 最小化能量泛函：

    F(Z) = sum_i ∫_{V_i} ρ(r) |r - z_i|^2 dr

---
**点-三角形测试：**
对三角形 ABC 和点 P，使用重心坐标：

    P = u*A + v*B + w*C,  u+v+w = 1

P 在三角形内当且仅当 u,v,w ≥ 0。
由叉积符号判断：
    cross1 = (B-A) × (P-A)
    cross2 = (C-B) × (P-B)
    cross3 = (A-C) × (P-C)
    inside = sign(cross1) == sign(cross2) == sign(cross3)

---
**Brent 一维最小化：**
用于 CVT 迭代中的步长选择（这里采用简化 Lloyd 迭代）。

---
**皮层电极 3D 几何：**
电极坐标 (x,y,z) 位于皮层曲面近似上：
    z = f_cortex(x,y) = R - sqrt(R^2 - x^2 - y^2)  （球冠近似）
其中 R 为皮层曲率半径（约 80 mm）。
"""

import numpy as np


def point_in_triangle_2d(A, B, C, P):
    """
    2D 点-三角形包含测试（叉积法）。
    A,B,C 为三角形顶点，P 为测试点。返回布尔值。
    """
    A, B, C, P = map(np.asarray, (A, B, C, P))
    cross1 = np.cross(B - A, P - A)
    cross2 = np.cross(C - B, P - B)
    cross3 = np.cross(A - C, P - C)
    # 对退化情况使用容差
    eps = 1e-12
    s1 = np.sign(cross1)
    s2 = np.sign(cross2)
    s3 = np.sign(cross3)
    # 允许在边界上
    if abs(cross1) < eps:
        s1 = 0
    if abs(cross2) < eps:
        s2 = 0
    if abs(cross3) < eps:
        s3 = 0
    # 所有非零符号一致（或为零）
    signs = [s for s in [s1, s2, s3] if s != 0]
    if len(signs) == 0:
        return True
    return all(s == signs[0] for s in signs)


def point_in_polygon_2d(vertices, P):
    """
    2D 点-多边形包含测试（射线法/奇偶规则）。
    vertices : Nx2 多边形顶点（顺序可为顺时针或逆时针）
    P        : 测试点
    """
    vertices = np.asarray(vertices, dtype=float)
    P = np.asarray(P, dtype=float)
    n = len(vertices)
    inside = False
    x, y = P
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # 检查边是否与水平射线相交
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if xinters > x:
                inside = not inside
    return inside


def hexagonal_grid_points(center, radius, n_layers):
    """
    生成以 center 为中心的六边形格点（同心六层）。
    参考 hex_grid_angle 算法。
    """
    center = np.asarray(center, dtype=float)
    points = [center.copy()]
    # 六边形六个方向
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    for layer in range(1, n_layers + 1):
        # 第 layer 层的点数 = 6 * layer
        for side in range(6):
            for step in range(layer):
                # 从该层的一个角出发，沿各边移动
                start = center + radius * layer * directions[side]
                end = center + radius * layer * directions[(side + 1) % 6]
                # 线性插值
                t = step / layer
                p = start + t * (end - start)
                points.append(p)
    return np.array(points, dtype=float)


def lloyd_cvt_iteration(points, domain, n_samples=1000, density_func=None):
    """
    单步 Lloyd CVT 迭代。
    通过蒙特卡洛采样近似 Voronoi 质心。
    domain = (xmin, xmax, ymin, ymax)
    """
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    xmin, xmax, ymin, ymax = domain
    # 在域内随机采样
    samples = np.column_stack([
        np.random.uniform(xmin, xmax, n_samples),
        np.random.uniform(ymin, ymax, n_samples)
    ])
    if density_func is not None:
        weights = np.array([density_func(s[0], s[1]) for s in samples])
        weights = np.maximum(weights, 1e-10)
    else:
        weights = np.ones(n_samples)
    # 分配到最近的生成点
    new_points = np.zeros_like(points)
    point_weights = np.zeros(n_points)
    for s, w in zip(samples, weights):
        dists = np.sum((points - s) ** 2, axis=1)
        idx = np.argmin(dists)
        new_points[idx] += w * s
        point_weights[idx] += w
    # 避免除零
    point_weights = np.maximum(point_weights, 1e-15)
    new_points = new_points / point_weights[:, None]
    return new_points


class CorticalSurfaceGeometry:
    """
    皮层曲面几何模型，基于 xyz_display 的 3D 坐标思想。
    用球冠近似局部皮层表面。
    """

    def __init__(self, curvature_radius=80.0, patch_radius=5.0):
        """
        curvature_radius : 皮层曲率半径（mm）
        patch_radius     : 局部 patches 半径（mm）
        """
        self.R = curvature_radius
        self.patch_r = patch_radius

    def surface_height(self, x, y):
        """
        球冠表面：z = R - sqrt(R^2 - x^2 - y^2)
        当 x^2 + y^2 > R^2 时退化为平面。
        """
        r2 = x ** 2 + y ** 2
        if r2 >= self.R ** 2:
            return 0.0
        return self.R - np.sqrt(self.R ** 2 - r2)

    def surface_normal(self, x, y):
        """
        计算表面法向量 n = (-∂z/∂x, -∂z/∂y, 1) / ||...||
        ∂z/∂x = x / sqrt(R^2 - x^2 - y^2)
        """
        r2 = x ** 2 + y ** 2
        if r2 >= self.R ** 2 - 1e-6:
            return np.array([0.0, 0.0, 1.0])
        dz_dx = x / np.sqrt(self.R ** 2 - r2)
        dz_dy = y / np.sqrt(self.R ** 2 - r2)
        n = np.array([-dz_dx, -dz_dy, 1.0])
        n = n / np.linalg.norm(n)
        return n

    def generate_electrode_positions(self, layout='hex', n_electrodes=64,
                                     n_layers=None):
        """
        生成电极在皮层表面的 3D 位置。
        layout : 'hex'（六边形）或 'cvt'（CVT 优化）
        """
        if n_layers is None:
            # 六边形格点公式：N = 1 + 3*n*(n+1)
            # 解方程求 n：n = ceil( (-3 + sqrt(9 + 12*(n_electrodes-1))) / 6 )
            n_layers = max(1, int(np.ceil((-3 + np.sqrt(9 + 12 * (n_electrodes - 1))) / 6)))
            # 确保足够点数
            while True:
                test_points = hexagonal_grid_points(
                    center=[0.0, 0.0],
                    radius=self.patch_r / n_layers,
                    n_layers=n_layers)
                if len(test_points) >= n_electrodes:
                    break
                n_layers += 1
        if layout == 'hex':
            points_2d = hexagonal_grid_points(
                center=[0.0, 0.0],
                radius=self.patch_r / n_layers,
                n_layers=n_layers)
            # 截断到所需数量
            points_2d = points_2d[:n_electrodes]
        elif layout == 'cvt':
            # 初始六边形格点，然后 Lloyd 迭代优化
            points_2d = hexagonal_grid_points(
                center=[0.0, 0.0],
                radius=self.patch_r / n_layers,
                n_layers=n_layers)
            points_2d = points_2d[:n_electrodes]
            domain = (-self.patch_r, self.patch_r, -self.patch_r, self.patch_r)
            for _ in range(20):
                points_2d = lloyd_cvt_iteration(points_2d, domain, n_samples=2000)
                # 限制在圆域内
                dists = np.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)
                mask = dists > self.patch_r
                if np.any(mask):
                    angle = np.arctan2(points_2d[mask, 1], points_2d[mask, 0])
                    points_2d[mask, 0] = self.patch_r * np.cos(angle) * 0.95
                    points_2d[mask, 1] = self.patch_r * np.sin(angle) * 0.95
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # 映射到 3D 曲面
        positions_3d = []
        for p in points_2d:
            x, y = p
            z = self.surface_height(x, y)
            positions_3d.append([x, y, z])
        return np.array(positions_3d, dtype=float)

    def compute_inter_electrode_distances(self, positions):
        """
        计算所有电极对之间的欧氏距离矩阵。
        """
        n = len(positions)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def triangulate_electrode_patch(self, positions, max_edge_length=None):
        """
        对电极位置进行 Delaunay 三角剖分（投影到切平面后）。
        返回三角形列表，每个三角形为三个电极索引。
        """
        from scipy.spatial import Delaunay
        # 投影到局部切平面（近似为 z=0 平面）
        pts_2d = positions[:, :2]
        if len(pts_2d) < 3:
            return np.zeros((0, 3), dtype=int)
        tri = Delaunay(pts_2d)
        triangles = tri.simplices
        if max_edge_length is not None:
            # 过滤掉边长过长的三角形
            filtered = []
            for tri_idx in triangles:
                p0, p1, p2 = positions[tri_idx]
                e0 = np.linalg.norm(p0 - p1)
                e1 = np.linalg.norm(p1 - p2)
                e2 = np.linalg.norm(p2 - p0)
                if max(e0, e1, e2) <= max_edge_length:
                    filtered.append(tri_idx)
            triangles = np.array(filtered, dtype=int)
        return triangles


class ElectrodeArray:
    """
    电极阵列管理器：生成布局、计算采样矩阵、处理空间信号。
    """

    def __init__(self, n_electrodes=64, geometry=None):
        if geometry is None:
            geometry = CorticalSurfaceGeometry()
        self.geometry = geometry
        self.n_electrodes = n_electrodes
        self.positions = None
        self.triangles = None

    def generate_layout(self, layout='cvt'):
        self.positions = self.geometry.generate_electrode_positions(
            layout=layout, n_electrodes=self.n_electrodes)
        self.triangles = self.geometry.triangulate_electrode_patch(
            self.positions, max_edge_length=2.0)
        return self.positions

    def sample_neural_field(self, field_func):
        """
        在每个电极位置 (x,y,z) 处采样神经场。
        field_func(x,y,z) -> float
        """
        if self.positions is None:
            self.generate_layout()
        vals = np.array([field_func(p[0], p[1], p[2]) for p in self.positions], dtype=float)
        return vals

    def compute_spatial_coverage(self):
        """
        计算电极阵列的空间覆盖度：
        覆盖面积 ≈ 所有三角形面积之和。
        """
        if self.triangles is None or len(self.triangles) == 0:
            return 0.0
        total_area = 0.0
        for tri in self.triangles:
            p0, p1, p2 = self.positions[tri]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p0 - p2)
            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
            total_area += area
        return total_area
