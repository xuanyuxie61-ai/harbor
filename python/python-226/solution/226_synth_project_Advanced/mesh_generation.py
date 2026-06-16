"""
网格生成与Voronoi分割模块
========================
实现中微子传播域的三角形网格生成，
以及参数空间的Voronoi剖分用于反演问题的分区搜索。

核心算法：
1. 2D三角形网格生成（Delaunay三角化）：
   - 输入：边界顶点集合
   - 输出：三角形网格 (节点坐标 + 单元连接)
   - 约束：最大单元尺寸 h_max

2. Voronoi剖分：
   - 给定点集 {p_i}，Voronoi单元 V_i = {x: ||x-p_i|| ≤ ||x-p_j|| ∀j}
   - 对偶关系：Voronoi图 ⊥ Delaunay三角化
   - 应用：参数空间的不确定性分区

3. 点包含测试（Point-in-Polygon）：
   - 射线法（Ray Casting）
   -  winding number方法

4. 边界辅助点技术：
   - 在Voronoi图外围添加矩形辅助点
   - 避免无限远边的处理

数据来源：
- 725_matlab-map: 地图投影、点包含测试、Voronoi图
- 580_image_mesh2d: 2D三角网格生成
"""

import numpy as np
from typing import Tuple, List, Optional


class TriangularMesh2D:
    """
    2D三角形网格生成器。

    实现简单的Delaunay三角化：
    1. 在计算域内生成候选点
    2. 使用Bowyer-Watson算法进行Delaunay三角化
    3. 删除外部三角形
    """

    def __init__(self, boundary_vertices: np.ndarray, max_element_size: float = 0.1):
        """
        初始化网格生成器。

        参数：
            boundary_vertices: 边界顶点 (N_boundary, 2)
            max_element_size: 最大单元尺寸
        """
        self.boundary = boundary_vertices
        self.h_max = max_element_size
        self.nodes = None
        self.elements = None

    def generate_interior_points(self) -> np.ndarray:
        """
        在边界内生成内部点。

        方法：
        1. 计算边界包围盒
        2. 生成均匀候选点
        3. 筛选边界内的点

        返回：
            interior_pts: 内部点坐标 (N_interior, 2)
        """
        # 包围盒
        x_min, y_min = np.min(self.boundary, axis=0)
        x_max, y_max = np.max(self.boundary, axis=0)

        # 生成均匀网格点
        nx = int((x_max - x_min) / self.h_max) + 1
        ny = int((y_max - y_min) / self.h_max) + 1

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])

        # 筛选边界内的点
        interior = []
        for pt in candidates:
            if self.point_in_polygon(pt, self.boundary):
                # 确保不在边界上
                dist_to_boundary = np.min(np.linalg.norm(self.boundary - pt, axis=1))
                if dist_to_boundary > self.h_max * 0.1:
                    interior.append(pt)

        return np.array(interior) if interior else np.empty((0, 2))

    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        射线法判断点是否在多边形内。

        算法：
        从点向右发射水平射线，计算与多边形边的交点数。
        奇数交点 → 在内部，偶数 → 在外部。

        参数：
            point: 测试点 (2,)
            polygon: 多边形顶点 (N, 2)

        返回：
            is_inside: 是否在内部
        """
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside

            j = i

        return inside

    def delaunay_triangulation(self, points: np.ndarray) -> np.ndarray:
        """
        简化的Delaunay三角化。

        使用scipy的Delaunay实现。

        参数：
            points: 点集 (N, 2)

        返回：
            triangles: 三角形连接 (M, 3)
        """
        from scipy.spatial import Delaunay

        if len(points) < 3:
            return np.empty((0, 3), dtype=int)

        try:
            tri = Delaunay(points)
            return tri.simplices
        except Exception:
            return np.empty((0, 3), dtype=int)

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整的三角形网格。

        步骤：
        1. 合并边界点和内部点
        2. Delaunay三角化
        3. 删除中心在边界外的三角形

        返回：
            nodes: 节点坐标 (N_nodes, 2)
            elements: 三角形连接 (N_elements, 3)
        """
        # 内部点
        interior = self.generate_interior_points()

        # 合并
        if len(interior) > 0:
            all_points = np.vstack([self.boundary, interior])
        else:
            all_points = self.boundary.copy()

        # 去重
        all_points = np.unique(all_points, axis=0)

        # Delaunay三角化
        triangles = self.delaunay_triangulation(all_points)

        # 过滤：删除中心在边界外的三角形
        valid_triangles = []
        for tri in triangles:
            center = np.mean(all_points[tri], axis=0)
            if self.point_in_polygon(center, self.boundary):
                valid_triangles.append(tri)

        self.nodes = all_points
        self.elements = np.array(valid_triangles) if valid_triangles else np.empty((0, 3), dtype=int)

        return self.nodes, self.elements

    def compute_mesh_quality(self) -> dict:
        """
        计算网格质量指标。

        指标：
        1. 最小角
        2. 最大面积比
        3. 单元数量

        返回：
            quality: 质量指标字典
        """
        if self.elements is None or len(self.elements) == 0:
            return {'n_elements': 0, 'min_angle': 0, 'max_area_ratio': 0}

        angles = []
        areas = []

        for tri in self.elements:
            pts = self.nodes[tri]
            # 三条边
            edges = [pts[1] - pts[0], pts[2] - pts[1], pts[0] - pts[2]]
            lengths = [np.linalg.norm(e) for e in edges]

            # 面积
            area = 0.5 * abs(np.cross(edges[0], -edges[2]))
            areas.append(area)

            # 角度（余弦定理）
            for i in range(3):
                a, b, c = lengths[(i+1)%3], lengths[(i+2)%3], lengths[i]
                if a * b > 1e-15:
                    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angles.append(np.arccos(cos_angle))

        angles = np.array(angles)
        areas = np.array(areas)

        return {
            'n_elements': len(self.elements),
            'n_nodes': len(self.nodes),
            'min_angle_deg': np.degrees(np.min(angles)) if len(angles) > 0 else 0,
            'max_angle_deg': np.degrees(np.max(angles)) if len(angles) > 0 else 0,
            'mean_area': np.mean(areas) if len(areas) > 0 else 0,
            'total_area': np.sum(areas),
        }


class VoronoiPartitioner:
    """
    Voronoi参数空间分割器。

    用于中微子振荡参数反演中的搜索空间划分。

    数学定义：
    给定点集 {c_i}（中心），Voronoi单元为：
    V_i = {x ∈ ℝ² : ||x - c_i|| ≤ ||x - c_j|| ∀j ≠ i}

    对偶关系：
    Voronoi图 ↔ Delaunay三角化
    """

    def __init__(self, centers: np.ndarray):
        """
        初始化Voronoi分割器。

        参数：
            centers: 中心点坐标 (N_centers, 2)
        """
        self.centers = centers
        self.n_centers = len(centers)

    def compute_voronoi_regions(self, bounds: Tuple[float, float, float, float] = None,
                                 resolution: int = 100) -> List[np.ndarray]:
        """
        计算Voronoi区域。

        方法：在网格上对每个点计算最近中心

        参数：
            bounds: 区域边界 (x_min, x_max, y_min, y_max)
            resolution: 网格分辨率

        返回：
            regions: 每个中心的区域点集列表
        """
        if bounds is None:
            x_min, y_min = np.min(self.centers, axis=0) - 1
            x_max, y_max = np.max(self.centers, axis=0) + 1
        else:
            x_min, x_max, y_min, y_max = bounds

        # 生成网格
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # 对每个网格点找最近中心
        regions = [[] for _ in range(self.n_centers)]

        for pt in grid_points:
            dists = np.linalg.norm(self.centers - pt, axis=1)
            nearest = np.argmin(dists)
            regions[nearest].append(pt)

        return [np.array(r) if r else np.empty((0, 2)) for r in regions]

    def add_boundary_points(self, margin: float = 5.0) -> np.ndarray:
        """
        添加边界辅助点以避免无限Voronoi边。

        在中心点集的包围盒外添加一圈辅助点。

        参数：
            margin: 边界外延距离

        返回：
            augmented_centers: 增广后的中心点集
        """
        x_min, y_min = np.min(self.centers, axis=0) - margin
        x_max, y_max = np.max(self.centers, axis=0) + margin

        # 在边界上均匀分布辅助点
        n_per_side = 10
        boundary_pts = []

        # 下边
        for x in np.linspace(x_min, x_max, n_per_side):
            boundary_pts.append([x, y_min])
        # 上边
        for x in np.linspace(x_min, x_max, n_per_side):
            boundary_pts.append([x, y_max])
        # 左边
        for y in np.linspace(y_min, y_max, n_per_side):
            boundary_pts.append([x_min, y])
        # 右边
        for y in np.linspace(y_min, y_max, n_per_side):
            boundary_pts.append([x_max, y])

        boundary_pts = np.array(boundary_pts)
        augmented = np.vstack([self.centers, boundary_pts])

        return augmented

    def nearest_center(self, points: np.ndarray) -> np.ndarray:
        """
        对每个点找最近中心。

        参数：
            points: 查询点 (N, 2)

        返回：
            indices: 最近中心索引 (N,)
        """
        indices = np.zeros(len(points), dtype=int)
        for i, pt in enumerate(points):
            dists = np.linalg.norm(self.centers - pt, axis=1)
            indices[i] = np.argmin(dists)
        return indices


def create_rectangular_boundary(width: float = 1.0, height: float = 1.0,
                                 n_points_per_side: int = 10) -> np.ndarray:
    """
    创建矩形边界顶点。

    参数：
        width, height: 矩形尺寸
        n_points_per_side: 每边的点数

    返回：
        boundary: 边界顶点 (N, 2)
    """
    pts = []

    # 下边
    for x in np.linspace(0, width, n_points_per_side, endpoint=False):
        pts.append([x, 0])
    # 右边
    for y in np.linspace(0, height, n_points_per_side, endpoint=False):
        pts.append([width, y])
    # 上边
    for x in np.linspace(width, 0, n_points_per_side, endpoint=False):
        pts.append([x, height])
    # 左边
    for y in np.linspace(height, 0, n_points_per_side, endpoint=False):
        pts.append([0, y])

    return np.array(pts)


def create_circular_boundary(radius: float = 1.0, center: np.ndarray = None,
                              n_points: int = 40) -> np.ndarray:
    """
    创建圆形边界顶点。

    参数：
        radius: 半径
        center: 圆心
        n_points: 边界点数

    返回：
        boundary: 边界顶点 (n_points, 2)
    """
    if center is None:
        center = np.array([0.0, 0.0])

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return np.column_stack([x, y])
