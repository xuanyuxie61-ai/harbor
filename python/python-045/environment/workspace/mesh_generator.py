#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_generator.py
非结构化与结构化网格生成器

融合种子项目：
  - 820_numgrid: 二维区域网格点编号与区域定义
  - 425_ffmatlib (ffreadmesh, prepare_mesh): 网格数据结构准备
  - 1320_triangle_to_fem: Triangle 网格到 FEM 格式转换
  - 1322_triangle_to_xml: 网格格式转换与 I/O

核心功能：
  1. 多种二维区域定义（矩形、L形、圆盘、环形、心形等）
  2. 规则矩形网格生成（用于有限差分）
  3. 非结构化三角形网格生成（基于 Delaunay 剖分）
  4. 网格编号、边界标记、元素连接性
  5. 网格质量评估
"""

import numpy as np


class Region2D:
    """
    二维区域定义类（基于 numgrid 的思想扩展）

    支持的区域类型：
      'S' : 正方形 (-1,1) x (-1,1)
      'L' : L形区域
      'D' : 单位圆盘
      'A' : 环形 (annulus)
      'R' : 矩形 [xmin,xmax] x [zmin,zmax]
    """

    def __init__(self, region_type, params=None):
        self.region_type = region_type.upper()
        self.params = params or {}

    def contains(self, x, y):
        """
        判断点 (x, y) 是否在区域内

        Parameters
        ----------
        x, y : float or ndarray
            坐标

        Returns
        -------
        bool or ndarray of bool
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if self.region_type == 'S':
            return (x > -1.0) & (x < 1.0) & (y > -1.0) & (y < 1.0)
        elif self.region_type == 'L':
            return (x > -1.0) & (x < 1.0) & (y > -1.0) & (y < 1.0) & ((x > 0.0) | (y > 0.0))
        elif self.region_type == 'D':
            return x ** 2 + y ** 2 < 1.0
        elif self.region_type == 'A':
            r2 = x ** 2 + y ** 2
            r_in = self.params.get('r_in', 1.0 / np.sqrt(3.0))
            r_out = self.params.get('r_out', 1.0)
            return (r2 < r_out ** 2) & (r2 > r_in ** 2)
        elif self.region_type == 'R':
            xmin = self.params.get('xmin', 0.0)
            xmax = self.params.get('xmax', 1.0)
            zmin = self.params.get('zmin', 0.0)
            zmax = self.params.get('zmax', 1.0)
            return (x >= xmin) & (x <= xmax) & (y >= zmin) & (y <= zmax)
        elif self.region_type == 'H':
            # 心形 (cardioid) 区域
            rho = 0.75
            sigma = 0.75
            r2 = x ** 2 + y ** 2
            return r2 * (r2 - sigma * y) < rho * x ** 2
        else:
            raise ValueError(f"不支持的区域类型: {self.region_type}")

    def bounding_box(self):
        """返回区域的边界框 (xmin, xmax, ymin, ymax)"""
        if self.region_type in ('S', 'L'):
            return (-1.0, 1.0, -1.0, 1.0)
        elif self.region_type == 'D':
            return (-1.0, 1.0, -1.0, 1.0)
        elif self.region_type == 'A':
            r_out = self.params.get('r_out', 1.0)
            return (-r_out, r_out, -r_out, r_out)
        elif self.region_type == 'R':
            return (self.params.get('xmin', 0.0), self.params.get('xmax', 1.0),
                    self.params.get('zmin', 0.0), self.params.get('zmax', 1.0))
        elif self.region_type == 'H':
            return (-1.0, 1.0, -1.0, 1.0)
        else:
            raise ValueError(f"不支持的区域类型: {self.region_type}")


class StructuredMesh2D:
    """
    二维结构化矩形网格（用于有限差分正演）

    基于 numgrid 的编号思想，对活跃节点进行连续编号。
    """

    def __init__(self, region, nx, ny):
        self.region = region
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("网格分辨率至少为 2")

        xmin, xmax, ymin, ymax = region.bounding_box()
        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # 创建二维坐标网格
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 标记活跃节点
        self.active = region.contains(self.X, self.Y)

        # 连续编号活跃节点
        self.node_number = np.zeros((nx, ny), dtype=np.int32)
        self.node_coords = []
        self.inv_map = {}  # node_index -> (i, j)
        idx = 0
        for j in range(ny):
            for i in range(nx):
                if self.active[i, j]:
                    self.node_number[i, j] = idx
                    self.node_coords.append((self.X[i, j], self.Y[i, j]))
                    self.inv_map[idx] = (i, j)
                    idx += 1
                else:
                    self.node_number[i, j] = -1
        self.n_nodes = idx
        self.node_coords = np.array(self.node_coords, dtype=np.float64)

        # 识别边界节点
        self._mark_boundary()

    def _mark_boundary(self):
        """标记边界节点：活跃但邻居有不活跃的"""
        self.boundary = np.zeros((self.nx, self.ny), dtype=bool)
        for j in range(self.ny):
            for i in range(self.nx):
                if not self.active[i, j]:
                    continue
                is_boundary = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= self.nx or nj < 0 or nj >= self.ny:
                        is_boundary = True
                        break
                    if not self.active[ni, nj]:
                        is_boundary = True
                        break
                self.boundary[i, j] = is_boundary

        self.boundary_nodes = []
        self.interior_nodes = []
        for idx in range(self.n_nodes):
            i, j = self.inv_map[idx]
            if self.boundary[i, j]:
                self.boundary_nodes.append(idx)
            else:
                self.interior_nodes.append(idx)
        self.boundary_nodes = np.array(self.boundary_nodes, dtype=np.int32)
        self.interior_nodes = np.array(self.interior_nodes, dtype=np.int32)

    def get_node_index(self, i, j):
        """获取网格点 (i,j) 的节点编号"""
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return -1
        return self.node_number[i, j]

    def get_neighbors(self, node_idx):
        """
        获取节点的邻居索引（东、西、南、北）

        Returns
        -------
        list of (neighbor_idx, direction)
        """
        i, j = self.inv_map[node_idx]
        neighbors = []
        for di, dj, name in [(1, 0, 'E'), (-1, 0, 'W'), (0, 1, 'N'), (0, -1, 'S')]:
            ni, nj = i + di, j + dj
            nidx = self.get_node_index(ni, nj)
            if nidx >= 0:
                neighbors.append((nidx, name))
        return neighbors

    def build_laplacian_fd(self, conductivity_at_nodes):
        """
        构建有限差分 Laplacian 算子矩阵 A，用于 MT 正演

        二维扩散方程的离散：
            ∂²u/∂x² + ∂²u/∂y² + k² u = 0
        其中 k² = iωμ₀σ

        使用五点差分格式：
            (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / dx²
          + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / dy²
          + k²_{i,j} * u_{i,j} = 0

        Returns
        -------
        A : ndarray, shape (n_nodes, n_nodes)
            复数系数矩阵
        """
        sigma = np.asarray(conductivity_at_nodes, dtype=np.complex128)
        if len(sigma) != self.n_nodes:
            raise ValueError("电导率数组长度必须与节点数一致")

        A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.complex128)
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        for idx in range(self.n_nodes):
            i, j = self.inv_map[idx]
            neighbors = self.get_neighbors(idx)

            # 对角项
            coeff = 0.0
            for nidx, direction in neighbors:
                if direction in ('E', 'W'):
                    coeff -= 1.0 / dx2
                    A[idx, nidx] += 1.0 / dx2
                else:
                    coeff -= 1.0 / dy2
                    A[idx, nidx] += 1.0 / dy2

            # 自相互作用项包含 k²
            A[idx, idx] = coeff + sigma[idx]

        return A


class UnstructuredMesh2D:
    """
    二维非结构化三角形网格

    基于 Triangle 剖分思想，手动实现简单的 Delaunay 风格网格生成。
    """

    def __init__(self, points, triangles, boundary_edges=None):
        """
        Parameters
        ----------
        points : ndarray, shape (n_points, 2)
            节点坐标
        triangles : ndarray, shape (n_triangles, 3)
            三角形单元，每行三个节点索引
        boundary_edges : ndarray or None, shape (n_edges, 2)
            边界边，每行两个节点索引
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.n_points = len(self.points)
        self.n_triangles = len(self.triangles)

        if boundary_edges is None:
            self.boundary_edges = self._extract_boundary_edges()
        else:
            self.boundary_edges = np.asarray(boundary_edges, dtype=np.int32)

        self._compute_element_properties()

    def _extract_boundary_edges(self):
        """从三角形连接性中提取边界边"""
        edge_count = {}
        for tri in self.triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                key = tuple(sorted(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = [list(e) for e, count in edge_count.items() if count == 1]
        return np.array(boundary, dtype=np.int32)

    def _compute_element_properties(self):
        """计算每个三角形的面积和外心"""
        self.areas = np.zeros(self.n_triangles, dtype=np.float64)
        self.centroids = np.zeros((self.n_triangles, 2), dtype=np.float64)

        for e, tri in enumerate(self.triangles):
            p0 = self.points[tri[0]]
            p1 = self.points[tri[1]]
            p2 = self.points[tri[2]]
            # 面积 = 0.5 * |cross(p1-p0, p2-p0)|
            area = 0.5 * np.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                                (p2[0] - p0[0]) * (p1[1] - p0[1]))
            self.areas[e] = area
            self.centroids[e] = (p0 + p1 + p2) / 3.0

    def mesh_quality(self):
        """
        计算网格质量指标

        最小角、最大角、面积比等。
        """
        min_angles = []
        max_angles = []
        for tri in self.triangles:
            p0, p1, p2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)
            # 用余弦定理计算角度
            angles = []
            for sides in [(a, b, c), (b, c, a), (c, a, b)]:
                s0, s1, s2 = sides
                cos_angle = (s1 ** 2 + s2 ** 2 - s0 ** 2) / (2.0 * s1 * s2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(np.arccos(cos_angle) * 180.0 / np.pi)
            min_angles.append(min(angles))
            max_angles.append(max(angles))
        return {
            'min_angle': np.min(min_angles),
            'max_angle': np.max(max_angles),
            'mean_area': np.mean(self.areas),
            'min_area': np.min(self.areas),
            'max_area': np.max(self.areas),
        }


def generate_rectangular_mesh(xmin, xmax, zmin, zmax, nx, nz):
    """
    生成规则矩形结构化网格

    这是 MT 二维正演中最常用的网格类型。
    """
    region = Region2D('R', {'xmin': xmin, 'xmax': xmax, 'zmin': zmin, 'zmax': zmax})
    mesh = StructuredMesh2D(region, nx, nz)
    return mesh


def generate_annulus_mesh(r_inner, r_outer, n_radial, n_angular):
    """
    生成环形区域上的三角形网格

    融合 820_numgrid 的环形区域定义与三角形网格生成。
    """
    # 在极坐标下生成点
    r = np.linspace(r_inner, r_outer, n_radial)
    theta = np.linspace(0, 2.0 * np.pi, n_angular, endpoint=False)

    points = []
    for ri in r:
        for th in theta:
            points.append([ri * np.cos(th), ri * np.sin(th)])
    points = np.array(points, dtype=np.float64)

    # 生成三角形连接性（简单扇形剖分）
    triangles = []
    for i in range(n_radial - 1):
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            p0 = i * n_angular + j
            p1 = (i + 1) * n_angular + j
            p2 = (i + 1) * n_angular + j_next
            p3 = i * n_angular + j_next
            triangles.append([p0, p1, p2])
            triangles.append([p0, p2, p3])
    triangles = np.array(triangles, dtype=np.int32)

    # 提取边界边
    boundary_edges = []
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        boundary_edges.append([j, j_next])  # 内边界
        boundary_edges.append([(n_radial - 1) * n_angular + j,
                               (n_radial - 1) * n_angular + j_next])  # 外边界
    boundary_edges = np.array(boundary_edges, dtype=np.int32)

    return UnstructuredMesh2D(points, triangles, boundary_edges)


if __name__ == "__main__":
    mesh = generate_rectangular_mesh(0.0, 10000.0, 0.0, 5000.0, 21, 11)
    print(f"矩形网格: {mesh.n_nodes} 节点, dx={mesh.dx:.1f}, dy={mesh.dy:.1f}")
    print(f"边界节点: {len(mesh.boundary_nodes)}, 内部节点: {len(mesh.interior_nodes)}")

    ann_mesh = generate_annulus_mesh(1000.0, 5000.0, 6, 16)
    print(f"环形网格: {ann_mesh.n_points} 节点, {ann_mesh.n_triangles} 三角形")
    q = ann_mesh.mesh_quality()
    print(f"网格质量: 最小角={q['min_angle']:.2f}°, 最大角={q['max_angle']:.2f}°")
