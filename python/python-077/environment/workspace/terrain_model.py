"""
terrain_model.py
海底/陆上地形模型与有限元网格

融合源项目：
- 502_hand_data: 轮廓数据采集（地形轮廓点的采集与处理）
- 406_fem2d_mesh_display: FEM二维网格生成（节点-单元拓扑关系）
- 468_geometry: 多边形面积、三角形面积（地形区域面积计算）
"""

import numpy as np
from typing import List, Tuple, Optional


class TerrainProfile:
    """
    地形轮廓模型。

    物理模型：
    -----------
    地形高程 z(x, y) 的离散表示。对于风电场微观选址，地形起伏会影响：
    
    1. 风速随高度的对数律修正：
        u(z) = u_ref · ln(z/z_0) / ln(z_ref/z_0)
    
    2. 地形加速因子（speed-up）：
        S = 1 + Δh / L · f(θ)
    
    其中 Δh 为地形高度差，L 为特征长度，f(θ) 为方向因子。
    """

    def __init__(self):
        self.points = np.zeros((0, 2))  # (x, z) 或 (x, y, z) 的简化 2D 轮廓
        self.closed = False

    def add_point(self, x: float, z: float):
        """添加地形轮廓点 (x, z)。"""
        self.points = np.vstack([self.points, [x, z]])

    def close_profile(self):
        """闭合轮廓（如果需要计算面积）。"""
        if len(self.points) > 0 and not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])
        self.closed = True

    def polygon_area(self) -> float:
        """
        计算地形轮廓多边形的有向面积。

        采用 shoelace 公式（源自 468_geometry 的 polygon_area）：

            A = 0.5 · Σ_{i=1}^{n} (x_i · (z_{i+1} - z_{i-1}))

        其中下标循环：z_0 = z_n, z_{n+1} = z_1。

        Returns
        -------
        float
            多边形面积 [m²]。逆时针为正。
        """
        if len(self.points) < 3:
            return 0.0
        pts = self.points
        n = len(pts)
        area = 0.0
        for i in range(n):
            im1 = (i - 1) % n
            ip1 = (i + 1) % n
            area += pts[i, 0] * (pts[ip1, 1] - pts[im1, 1])
        return 0.5 * area

    def triangle_area(self, i: int, j: int, k: int) -> float:
        """
        计算由三个轮廓点构成的三角形面积。

        源自 468_geometry 的 triangle_area：

            A = 0.5 · |x1(z2 - z3) + x2(z3 - z1) + x3(z1 - z2)|
        """
        p1, p2, p3 = self.points[i], self.points[j], self.points[k]
        area = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        return area

    def elevation_at(self, x: float) -> float:
        """
        线性插值求 x 处的高程。

        Parameters
        ----------
        x : float
            水平位置 [m]。

        Returns
        -------
        float
            高程 [m]。
        """
        if len(self.points) == 0:
            return 0.0
        pts = self.points
        if x <= pts[0, 0]:
            return pts[0, 1]
        if x >= pts[-1, 0]:
            return pts[-1, 1]

        # 二分查找
        idx = np.searchsorted(pts[:, 0], x, side='right') - 1
        idx = max(0, min(idx, len(pts) - 2))
        x0, z0 = pts[idx]
        x1, z1 = pts[idx + 1]
        dx = x1 - x0
        if abs(dx) < 1e-14:
            return z0
        t = (x - x0) / dx
        return z0 + t * (z1 - z0)

    def slope_at(self, x: float) -> float:
        """
        计算 x 处的地形坡度 [m/m]。

        采用中心差分：
            dz/dx ≈ (z(x+δ) - z(x-δ)) / (2·δ)
        """
        delta = 1.0
        z_plus = self.elevation_at(x + delta)
        z_minus = self.elevation_at(x - delta)
        return (z_plus - z_minus) / (2.0 * delta)


class FEM2DMesh:
    """
    二维有限元网格生成器。

    融合 406_fem2d_mesh_display 的节点-单元拓扑思想。

    网格参数：
        - 矩形计算域 [xmin, xmax] × [ymin, ymax]
        - 划分为 nx × ny 个四边形单元
        - 每个四边形可进一步剖分为 2 个三角形
    """

    def __init__(self, xmin: float = 0.0, xmax: float = 5000.0,
                 ymin: float = 0.0, ymax: float = 5000.0,
                 nx: int = 20, ny: int = 20):
        """
        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            计算域边界 [m]。
        nx, ny : int
            x 和 y 方向的单元数。
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("单元数必须为正")
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.nodes = None
        self.elements = None
        self._generate_mesh()

    def _generate_mesh(self):
        """生成结构化四边形网格。"""
        nx, ny = self.nx, self.ny
        x = np.linspace(self.xmin, self.xmax, nx + 1)
        y = np.linspace(self.ymin, self.ymax, ny + 1)
        X, Y = np.meshgrid(x, y)

        n_nodes = (nx + 1) * (ny + 1)
        self.nodes = np.zeros((n_nodes, 2))
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = j * (nx + 1) + i
                self.nodes[idx] = [X[j, i], Y[j, i]]

        # 四边形单元：每个单元由 4 个节点组成
        n_elements = nx * ny
        self.elements = np.zeros((n_elements, 4), dtype=int)
        for j in range(ny):
            for i in range(nx):
                eidx = j * nx + i
                n1 = j * (nx + 1) + i
                n2 = n1 + 1
                n3 = n1 + (nx + 1) + 1
                n4 = n1 + (nx + 1)
                self.elements[eidx] = [n1, n2, n3, n4]

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_elements(self) -> int:
        return len(self.elements)

    def element_area(self, eidx: int) -> float:
        """
        计算第 eidx 个单元的面积（将四边形剖分为两个三角形求和）。

        源自 468_geometry 的三角形面积公式。
        """
        elem = self.elements[eidx]
        n1, n2, n3, n4 = elem
        p1 = self.nodes[n1]
        p2 = self.nodes[n2]
        p3 = self.nodes[n3]
        p4 = self.nodes[n4]

        # 三角形 1-2-3
        area1 = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        # 三角形 1-3-4
        area2 = 0.5 * abs(
            p1[0] * (p3[1] - p4[1]) +
            p3[0] * (p4[1] - p1[1]) +
            p4[0] * (p1[1] - p3[1])
        )
        return area1 + area2

    def total_domain_area(self) -> float:
        """计算整个计算域面积。"""
        return sum(self.element_area(i) for i in range(self.n_elements()))

    def find_element_containing(self, x: float, y: float) -> int:
        """
        找到包含点 (x, y) 的单元索引。

        采用暴力搜索，对于结构化网格可直接计算索引。
        """
        nx = self.nx
        dx = (self.xmax - self.xmin) / nx
        dy = (self.ymax - self.ymin) / self.ny

        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return -1

        i = int((x - self.xmin) / dx)
        j = int((y - self.ymin) / dy)
        i = min(i, nx - 1)
        j = min(j, self.ny - 1)
        return j * nx + i

    def node_neighborhood(self, node_idx: int, radius: float) -> List[int]:
        """
        找到给定节点半径范围内的所有节点索引。

        Parameters
        ----------
        node_idx : int
            中心节点索引。
        radius : float
            搜索半径 [m]。

        Returns
        -------
        List[int]
            邻域节点索引列表。
        """
        p = self.nodes[node_idx]
        neighbors = []
        for i, q in enumerate(self.nodes):
            if i != node_idx and np.linalg.norm(p - q) <= radius:
                neighbors.append(i)
        return neighbors
