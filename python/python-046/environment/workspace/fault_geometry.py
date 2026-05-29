"""
fault_geometry.py
断层几何建模与自适应网格生成模块。

融合种子项目:
  - 255_cvt_corn: CVT (Centroidal Voronoi Tessellation) 区域划分与自适应采样思想
  - 954_quadrilateral_mesh_order1_display: 四边形网格生成与节点-单元拓扑关系

在 InSAR 形变反演中的应用:
  1. 将断层面离散化为四边形或三角形单元；
  2. 在断层滑移梯度较大区域（如闭锁-蠕滑过渡带）使用 CVT 自适应加密；
  3. 建立节点编号、单元拓扑、边界标识等有限元前处理数据结构。
"""

import numpy as np
from utils import check_finite, compute_triangle_area, clip_to_range


class FaultMesh:
    """
    断层面网格类。
    断层参数化：沿走向 (strike) x ∈ [0, L], 沿倾向 (dip) z ∈ [0, W]。
    """

    def __init__(self, length, width, strike_deg, dip_deg, num_strike, num_dip,
                 adaptivity=False):
        """
        参数:
            length: 断层走向长度 (km)
            width:  断层倾向宽度 (km)
            strike_deg: 走向角 (度，从北顺时针)
            dip_deg: 倾角 (度，从水平面向下)
            num_strike: 走向方向单元数
            num_dip: 倾向方向单元数
            adaptivity: 是否启用 CVT 自适应加密
        """
        self.length = length
        self.width = width
        self.strike_deg = strike_deg
        self.dip_deg = dip_deg
        self.num_strike = num_strike
        self.num_dip = num_dip
        self.adaptivity = adaptivity

        # 构造网格
        if adaptivity:
            self.nodes, self.elements = self._build_cvt_adaptive_mesh()
        else:
            self.nodes, self.elements = self._build_regular_quadrilateral_mesh()

        self.num_nodes = self.nodes.shape[0]
        self.num_elements = self.elements.shape[0]

        # 标记边界节点
        self.boundary_flags = self._mark_boundary_nodes()

    def _build_regular_quadrilateral_mesh(self):
        """
        生成规则四边形网格。
        节点按行优先编号：先沿走向，再沿倾向。
        每个四边形单元拆分为两个三角形单元（T3）。
        """
        nx = self.num_strike + 1
        ny = self.num_dip + 1
        dx = self.length / self.num_strike
        dy = self.width / self.num_dip

        nodes = np.zeros((nx * ny, 2))
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                nodes[idx, 0] = i * dx  # 走向坐标
                nodes[idx, 1] = j * dy  # 倾向坐标

        # 四边形拆分为三角形
        elements = []
        for j in range(self.num_dip):
            for i in range(self.num_strike):
                n1 = j * nx + i
                n2 = j * nx + (i + 1)
                n3 = (j + 1) * nx + i
                n4 = (j + 1) * nx + (i + 1)
                # 三角形1: n1-n2-n4
                elements.append([n1, n2, n4])
                # 三角形2: n1-n4-n3
                elements.append([n1, n4, n3])

        return nodes, np.array(elements, dtype=int)

    def _build_cvt_adaptive_mesh(self):
        """
        基于 CVT (Centroidal Voronoi Tessellation) 的自适应网格生成。
        在滑动梯度预期较大的区域（断层中部深度，即闭锁区）加密采样点。

        CVT 能量泛函:
            F(P) = Σ_i ∫_{V_i} ρ(x) ||x - p_i||^2 dx
        其中 ρ(x) 为密度函数，在闭锁区取较大值。
        这里采用 Lloyd 松弛的简化版本。
        """
        nx = self.num_strike + 1
        ny = self.num_dip + 1
        n_total = nx * ny

        # 初始均匀采样
        np.random.seed(42)
        pts = np.zeros((n_total, 2))
        idx = 0
        for j in range(ny):
            for i in range(nx):
                pts[idx, 0] = i * self.length / self.num_strike
                pts[idx, 1] = j * self.width / self.num_dip
                idx += 1

        # 密度函数：在断层中部深度（~0.5W）处密度最大
        def density(x, y):
            # 高斯型密度，中心在 (L/2, 0.5W)
            cx, cy = 0.5 * self.length, 0.5 * self.width
            sx, sy = 0.3 * self.length, 0.2 * self.width
            d = np.exp(-((x - cx) ** 2) / (2 * sx ** 2) -
                       ((y - cy) ** 2) / (2 * sy ** 2))
            return 0.5 + 2.0 * d

        # Lloyd 松弛（简化版）
        n_lloyd = 5
        for _ in range(n_lloyd):
            # 计算每个采样点的 Voronoi 重心（使用蒙特卡洛近似）
            # 为简化，直接在局部区域重新分配点位置
            # 按密度加权的重心
            new_pts = np.zeros_like(pts)
            for k in range(n_total):
                xk, yk = pts[k]
                # 找邻居（距离最近的若干点）
                dists = np.sum((pts - pts[k]) ** 2, axis=1)
                neigh = np.argsort(dists)[1:min(9, n_total)]
                # 在邻居包围的局部区域内采样
                xmin = max(0.0, np.min(pts[neigh, 0]))
                xmax = min(self.length, np.max(pts[neigh, 0]))
                ymin = max(0.0, np.min(pts[neigh, 1]))
                ymax = min(self.width, np.max(pts[neigh, 1]))
                # 拒绝采样，按密度加权
                best_pt = pts[k]
                best_w = density(*best_pt)
                for _trial in range(20):
                    rx = xmin + np.random.rand() * (xmax - xmin)
                    ry = ymin + np.random.rand() * (ymax - ymin)
                    w = density(rx, ry)
                    if w > best_w:
                        best_pt = np.array([rx, ry])
                        best_w = w
                new_pts[k] = best_pt
            pts = new_pts.copy()

        # 使用 Delaunay 三角剖分
        from scipy.spatial import Delaunay
        # 确保边界点
        boundary_pts = []
        for i in range(nx):
            boundary_pts.append([i * self.length / self.num_strike, 0.0])
            boundary_pts.append([i * self.length / self.num_strike, self.width])
        for j in range(1, ny - 1):
            boundary_pts.append([0.0, j * self.width / self.num_dip])
            boundary_pts.append([self.length, j * self.width / self.num_dip])
        boundary_pts = np.array(boundary_pts)
        all_pts = np.vstack([pts, boundary_pts])
        tri = Delaunay(all_pts)
        nodes = all_pts
        elements = tri.simplices

        # 只保留在矩形域内的三角形（过滤 Delaunay 外接三角形）
        valid = []
        for tri_idx in range(elements.shape[0]):
            c = np.mean(nodes[elements[tri_idx]], axis=0)
            if 0.0 <= c[0] <= self.length and 0.0 <= c[1] <= self.width:
                valid.append(elements[tri_idx])
        elements = np.array(valid, dtype=int)

        return nodes, elements

    def _mark_boundary_nodes(self):
        """
        标记边界节点: 走向两端和倾向两端。
        返回布尔数组，True 表示边界节点。
        """
        flags = np.zeros(self.num_nodes, dtype=bool)
        tol = 1e-6
        for i in range(self.num_nodes):
            x, y = self.nodes[i]
            if (abs(x) < tol or abs(x - self.length) < tol or
                    abs(y) < tol or abs(y - self.width) < tol):
                flags[i] = True
        return flags

    def get_element_centroids(self):
        """
        计算每个三角形单元的形心。
        """
        centroids = np.zeros((self.num_elements, 2))
        for e in range(self.num_elements):
            nids = self.elements[e]
            centroids[e] = np.mean(self.nodes[nids], axis=0)
        return centroids

    def map_to_3d(self, origin=np.array([0.0, 0.0, 0.0])):
        """
        将断层面 2D 坐标映射到 3D 空间。
        走向沿 x 轴，倾向沿 -z 方向倾斜。

        3D 坐标:
            X = x * cos(strike) + origin[0]
            Y = x * sin(strike) + origin[1]
            Z = -y * sin(dip) + origin[2]
        """
        strike_rad = np.deg2rad(self.strike_deg)
        dip_rad = np.deg2rad(self.dip_deg)
        nodes_3d = np.zeros((self.num_nodes, 3))
        for i in range(self.num_nodes):
            x2d, y2d = self.nodes[i]
            nodes_3d[i, 0] = origin[0] + x2d * np.cos(strike_rad)
            nodes_3d[i, 1] = origin[1] + x2d * np.sin(strike_rad)
            nodes_3d[i, 2] = origin[2] - y2d * np.sin(dip_rad)
        return nodes_3d

    def element_areas(self):
        """
        计算每个三角形单元的面积。
        """
        areas = np.zeros(self.num_elements)
        for e in range(self.num_elements):
            n1, n2, n3 = self.elements[e]
            areas[e] = compute_triangle_area(
                self.nodes[n1], self.nodes[n2], self.nodes[n3])
        return areas


class SurfaceGrid:
    """
    地表观测网格（InSAR 像素位置）。
    """

    def __init__(self, x_range, y_range, nx, ny):
        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        self.points = np.column_stack([X.ravel(), Y.ravel()])
        self.num_points = self.points.shape[0]
