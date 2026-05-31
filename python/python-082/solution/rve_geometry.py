"""
rve_geometry.py
代表性体积单元（RVE）几何建模与多边形纤维截面拓扑管理。
原项目映射：
  - 891_polygonal_surface_display 的节点/面片读取与拓扑数据结构
  - 1385_vandermonde_interp_2d 的二维多项式插值用于纤维界面应力场重构
科学背景：
  在纤维增强复合材料中，纤维通常呈多边形或椭圆形截面分布。
  RVE是介观尺度均匀化的基本单元，其几何精度直接影响损伤预测。
"""

import numpy as np
from utils import validate_positive, r8mat_print_some


class PolygonalFiber:
    """多边形纤维截面，用于RVE中的增强相几何描述。"""

    def __init__(self, center, radius, n_sides=6, orientation=0.0):
        """
        初始化正n边形纤维截面。
        center: (2,) 纤维中心坐标
        radius: 外接圆半径
        n_sides: 边数（默认六边形模拟碳纤维截面）
        orientation: 旋转角度（弧度）
        """
        validate_positive(radius, "fiber radius")
        if n_sides < 3:
            raise ValueError("Polygon must have at least 3 sides.")
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.n_sides = int(n_sides)
        self.orientation = float(orientation)
        self.nodes = self._generate_nodes()
        self.faces = self._generate_faces()

    def _generate_nodes(self):
        """生成多边形顶点。"""
        angles = np.linspace(0, 2 * np.pi, self.n_sides, endpoint=False) + self.orientation
        nodes = np.zeros((self.n_sides, 2))
        nodes[:, 0] = self.center[0] + self.radius * np.cos(angles)
        nodes[:, 1] = self.center[1] + self.radius * np.sin(angles)
        return nodes

    def _generate_faces(self):
        """生成面片连接关系（每行一个面片，存储顶点索引）。"""
        faces = np.zeros((self.n_sides, 2), dtype=int)
        for i in range(self.n_sides):
            faces[i, 0] = i
            faces[i, 1] = (i + 1) % self.n_sides
        return faces

    def area(self):
        """正n边形面积公式：A = (n/2) * R^2 * sin(2π/n)。"""
        return 0.5 * self.n_sides * self.radius ** 2 * np.sin(2.0 * np.pi / self.n_sides)

    def moment_of_inertia(self):
        """截面惯性矩（用于纤维屈曲分析）。"""
        return self.n_sides * self.radius ** 4 / 24.0 * (
            np.sin(2.0 * np.pi / self.n_sides) +
            2.0 * np.sin(4.0 * np.pi / self.n_sides)
        )


class RVEGeometry:
    """代表性体积单元（Representative Volume Element）几何管理。"""

    def __init__(self, width, height, fiber_list=None, nx=40, ny=40):
        """
        width, height: RVE平面尺寸（微米）
        fiber_list: PolygonalFiber列表
        nx, ny: 背景网格划分数
        """
        validate_positive(width, "RVE width")
        validate_positive(height, "RVE height")
        self.width = float(width)
        self.height = float(height)
        self.fibers = fiber_list if fiber_list is not None else []
        self.nx = nx
        self.ny = ny
        self._build_mesh()

    def _build_mesh(self):
        """构建背景笛卡尔网格。"""
        self.x_grid = np.linspace(0.0, self.width, self.nx)
        self.y_grid = np.linspace(0.0, self.height, self.ny)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

    def fiber_volume_fraction(self):
        """计算纤维体积分数 V_f = Σ A_fiber / A_RVE。"""
        total_fiber_area = sum(f.area() for f in self.fibers)
        rve_area = self.width * self.height
        return total_fiber_area / rve_area

    def point_in_fiber(self, x, y):
        """判断点(x,y)是否位于任一纤维内部（射线法）。"""
        for f in self.fibers:
            if self._point_in_polygon(x, y, f.nodes):
                return True
        return False

    @staticmethod
    def _point_in_polygon(x, y, poly_nodes):
        """射线法判断点是否在多边形内。"""
        n = len(poly_nodes)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly_nodes[i]
            xj, yj = poly_nodes[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-16) + xi):
                inside = not inside
            j = i
        return inside

    def compute_interface_nodes(self):
        """
        提取纤维-基体界面节点坐标，用于界面脱粘损伤计算。
        返回界面节点数组 (N_interface, 2)。
        """
        interface_nodes = []
        for f in self.fibers:
            # 在纤维边界上均匀采样
            n_sample = max(20, f.n_sides * 3)
            t = np.linspace(0, 1, n_sample, endpoint=False)
            for ti in t:
                # 线性插值边界
                idx = int(ti * f.n_sides) % f.n_sides
                next_idx = (idx + 1) % f.n_sides
                frac = ti * f.n_sides - idx
                pt = (1 - frac) * f.nodes[idx] + frac * f.nodes[next_idx]
                interface_nodes.append(pt)
        if len(interface_nodes) == 0:
            return np.zeros((0, 2))
        return np.array(interface_nodes)

    def vandermonde_interp_2d_field(self, field_values, eval_points, degree=3):
        """
        基于二维Vandermonde矩阵对场量进行多项式插值。
        原项目映射：1385_vandermonde_interp_2d_matrix, r8poly_value_2d。
        科学公式：
          p(x,y) = Σ_{s=0}^{m} Σ_{ex+ey=s} c_{ex,ey} * x^{ex} * y^{ey}
          其中 T(M+1) = (M+1)(M+2)/2 为系数总数。
        """
        m = degree
        # 构建Vandermonde矩阵
        n_data = len(field_values)
        tmp1 = (m + 1) * (m + 2) // 2
        if n_data < tmp1:
            # 数据不足时降阶
            while m > 0 and n_data < (m + 1) * (m + 2) // 2:
                m -= 1
            tmp1 = (m + 1) * (m + 2) // 2

        A = np.zeros((n_data, tmp1))
        x_data = self.X.flatten()[:n_data]
        y_data = self.Y.flatten()[:n_data]
        j = 0
        for s in range(m + 1):
            for ex in range(s, -1, -1):
                ey = s - ex
                A[:, j] = (x_data ** ex) * (y_data ** ey)
                j += 1

        # 最小二乘求解系数
        c, _, _, _ = np.linalg.lstsq(A, field_values, rcond=None)

        # 在eval_points处求值
        eval_points = np.asarray(eval_points)
        n_eval = eval_points.shape[0]
        p = np.zeros(n_eval)
        j = 0
        for s in range(m + 1):
            for ex in range(s, -1, -1):
                ey = s - ex
                p += c[j] * (eval_points[:, 0] ** ex) * (eval_points[:, 1] ** ey)
                j += 1
        return p

    def print_geometry_summary(self):
        """打印RVE几何摘要。"""
        print("=" * 60)
        print("RVE Geometry Summary")
        print("=" * 60)
        print(f"  Dimensions: {self.width:.4f} x {self.height:.4f} um")
        print(f"  Grid: {self.nx} x {self.ny}")
        print(f"  Number of fibers: {len(self.fibers)}")
        print(f"  Fiber volume fraction: {self.fiber_volume_fraction():.4f}")
        if len(self.fibers) > 0:
            print("  Fiber nodes (first fiber):")
            r8mat_print_some(self.fibers[0].n_sides, 2, self.fibers[0].nodes, 1, 1,
                             min(5, self.fibers[0].n_sides), 2, "  Node coordinates:")
        print("=" * 60)


def generate_hexagonal_fiber_rve(width=100.0, height=100.0, fiber_radius=8.0,
                                  n_fibers_x=3, n_fibers_y=3):
    """生成六边形纤维呈六方排布的RVE。"""
    fibers = []
    dx_f = width / (n_fibers_x + 1)
    dy_f = height / (n_fibers_y + 1)
    for iy in range(n_fibers_y):
        offset = 0.5 * dx_f if iy % 2 == 1 else 0.0
        for ix in range(n_fibers_x):
            cx = dx_f * (ix + 1) + offset
            cy = dy_f * (iy + 1)
            if cx + fiber_radius < width and cy + fiber_radius < height:
                fibers.append(PolygonalFiber([cx, cy], fiber_radius, n_sides=6))
    return RVEGeometry(width, height, fibers, nx=50, ny=50)
