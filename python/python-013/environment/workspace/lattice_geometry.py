"""
lattice_geometry.py

基于 hex_grid (1146_square_hex_grid) 与 trinity (1358_trinity) 的
晶格几何构造模块。

为二维三角/六角Hubbard模型提供晶格坐标、近邻表、布里渊区边界。
三角格子可产生几何阻挫，是强关联物理中研究量子自旋液体的重要平台。
"""

import numpy as np
from typing import Tuple, List


class TriangularLattice:
    """
    二维三角晶格 (Triangular Lattice) 几何结构。
    
    三角晶格的基矢:
        a1 = a * (1, 0)
        a2 = a * (1/2, sqrt(3)/2)
    
    倒格矢:
        b1 = (2π/a) * (1, -1/√3)
        b2 = (2π/a) * (0, 2/√3)
    """

    def __init__(self, nx: int, ny: int, a: float = 1.0):
        if nx < 1 or ny < 1:
            raise ValueError("晶格尺寸 nx, ny 必须 >= 1")
        if a <= 0:
            raise ValueError("晶格常数 a 必须 > 0")
        self.nx = nx
        self.ny = ny
        self.a = a
        self.nsites = nx * ny
        self._build_lattice()
        self._build_neighbors()
        self._build_brillouin_zone()

    def _build_lattice(self):
        """构造实空间格点坐标，采用周期性边界条件。"""
        a = self.a
        a1 = np.array([a, 0.0])
        a2 = np.array([a * 0.5, a * np.sqrt(3.0) * 0.5])
        self.sites = np.zeros((self.nsites, 2))
        idx = 0
        for iy in range(self.ny):
            for ix in range(self.nx):
                self.sites[idx] = ix * a1 + iy * a2
                idx += 1
        self.a1 = a1
        self.a2 = a2

    def _build_neighbors(self):
        """
        构造近邻连接表。三角晶格每个格点有6个最近邻。
        采用周期性边界条件 (PBC)。
        """
        self.neighbors = [[] for _ in range(self.nsites)]
        # 六个近邻位移 (以晶格单位)
        deltas = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for idx in range(self.nsites):
            ix = idx % self.nx
            iy = idx // self.nx
            nbrs = []
            for dx, dy in deltas:
                jx = (ix + dx) % self.nx
                jy = (iy + dy) % self.ny
                jdx = jx + jy * self.nx
                nbrs.append(jdx)
            self.neighbors[idx] = nbrs

    def _build_brillouin_zone(self):
        """构造第一布里渊区 (六边形)。"""
        a = self.a
        b1 = np.array([2.0 * np.pi / a, -2.0 * np.pi / (a * np.sqrt(3.0))])
        b2 = np.array([0.0, 4.0 * np.pi / (a * np.sqrt(3.0))])
        self.b1 = b1
        self.b2 = b2
        # BZ 顶点 (六边形)
        self.bz_vertices = np.array([
            (2.0/3.0) * b1 + (1.0/3.0) * b2,
            (1.0/3.0) * b1 + (2.0/3.0) * b2,
            (-1.0/3.0) * b1 + (1.0/3.0) * b2,
            (-2.0/3.0) * b1 + (-1.0/3.0) * b2,
            (-1.0/3.0) * b1 + (-2.0/3.0) * b2,
            (1.0/3.0) * b1 + (-1.0/3.0) * b2,
        ])

    def reciprocal_lattice_points(self) -> np.ndarray:
        """返回离散的倒空间格点。"""
        # [HOLE 1] TODO: 修复倒空间格点生成
        # 需要利用倒格矢 b1, b2 构造均匀离散的 k 点网格。
        # 提示: 考虑 nx, ny 方向的均匀采样与倒格矢的线性组合。
        raise NotImplementedError("HOLE 1: 请实现 reciprocal_lattice_points")

    def site_index(self, ix: int, iy: int) -> int:
        """边界安全的格点索引。"""
        ix = ix % self.nx
        iy = iy % self.ny
        return ix + iy * self.nx


def hex_grid_in_brillouin_zone(n_layers: int, bz_vertices: np.ndarray) -> np.ndarray:
    """
    基于 hex_grid 思想，在六边形布里渊区内生成六角网格采样点。
    
    参数:
        n_layers: 层数，控制网格密度
        bz_vertices: 六边形BZ顶点
    
    返回:
        网格点坐标数组，形状 (N, 2)
    """
    if n_layers < 1:
        return bz_vertices[:1]
    # 六边形中心
    center = np.mean(bz_vertices, axis=0)
    # 六边形"半径"(中心到顶点距离)
    R = np.linalg.norm(bz_vertices[0] - center)
    # 六角网格生成
    points = [center]
    hx = R / n_layers
    hy = hx * np.sqrt(3.0) / 2.0
    for layer in range(1, n_layers + 1):
        # 每层6个方向
        for dir_idx in range(6):
            angle = dir_idx * np.pi / 3.0
            base = np.array([np.cos(angle), np.sin(angle)]) * hy * layer * (2.0 / np.sqrt(3.0))
            # 沿该方向layer步
            for step in range(layer):
                offset_angle = (dir_idx + 2) * np.pi / 3.0
                offset = np.array([np.cos(offset_angle), np.sin(offset_angle)]) * hx * step
                pt = center + base + offset
                # 检查是否在六边形内
                if _point_in_hexagon(pt, bz_vertices):
                    points.append(pt)
    return np.array(points)


def _point_in_hexagon(pt: np.ndarray, vertices: np.ndarray) -> bool:
    """射线法判断点是否在凸六边形内。"""
    x, y = pt
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def trinity_triangle_tiling_brillouin_zone(k_points: np.ndarray, bz_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 trinity 三角铺砖思想，将布里渊区剖分为三角形网格。
    使用Delaunay三角化实现，返回三角形顶点索引与面积权重。
    
    参数:
        k_points: BZ内的采样点
        bz_vertices: BZ边界顶点
    
    返回:
        triangles: 三角形顶点索引，形状 (M, 3)
        weights: 每个三角形的面积权重
    """
    from scipy.spatial import Delaunay
    if len(k_points) < 3:
        raise ValueError("k_points 数量必须 >= 3")
    tri = Delaunay(k_points)
    triangles = tri.simplices
    # 计算每个三角形面积
    weights = np.zeros(len(triangles))
    for i, tri_idx in enumerate(triangles):
        p0, p1, p2 = k_points[tri_idx]
        area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        weights[i] = area
    total = np.sum(weights)
    if total > 0:
        weights /= total
    return triangles, weights


if __name__ == "__main__":
    lat = TriangularLattice(4, 4, a=1.0)
    print(f"Sites: {lat.nsites}, Neighbors per site: {len(lat.neighbors[0])}")
    kpts = lat.reciprocal_lattice_points()
    print(f"Reciprocal points: {len(kpts)}")
