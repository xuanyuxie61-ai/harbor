"""
mesh_generator.py
重离子碰撞重叠区三角网格生成器

基于种子项目:
- 1330_triangulation: Delaunay三角剖分、三角网格邻接关系、网格质量评估
- 415_fem2d_scalar_display_brief: 有限元标量场离散化思想

物理应用:
在QGP流体力学计算中，需要在横向平面生成高质量的三角网格，
用于离散化粘性流体力学方程。
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import Delaunay


class MeshGenerator:
    """
    为QGP横向演化生成自适应三角网格。
    """

    def __init__(self, max_area: float = 0.5, min_angle: float = 25.0):
        """
        初始化网格生成参数。

        Parameters
        ----------
        max_area : float
            最大三角形面积 [fm²]
        min_angle : float
            最小内角 [度]
        """
        self.max_area = max_area
        self.min_angle_deg = min_angle
        self.min_angle_rad = np.deg2rad(min_angle)

    def generate_uniform_grid(self, xlim: Tuple[float, float],
                              ylim: Tuple[float, float],
                              nx: int = 40, ny: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        在矩形区域生成均匀点阵，然后进行Delaunay三角剖分。

        Parameters
        ----------
        xlim, ylim : Tuple[float, float]
            区域边界 [fm]
        nx, ny : int
            x, y方向点数

        Returns
        -------
        points : np.ndarray
            节点坐标 (N, 2)
        triangles : np.ndarray
            三角形连接关系 (M, 3)
        """
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

        tri = Delaunay(points)
        triangles = tri.simplices
        return points, triangles

    def generate_adaptive_mesh(self, xlim: Tuple[float, float],
                               ylim: Tuple[float, float],
                               density_func,
                               n_base: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于密度函数的自适应网格生成。

        在高密度区域加密网格，低密度区域稀疏化。

        Parameters
        ----------
        xlim, ylim : Tuple[float, float]
            区域边界
        density_func : callable
            密度函数 f(x, y) -> density
        n_base : int
            基础网格分辨率

        Returns
        -------
        points : np.ndarray
            节点坐标
        triangles : np.ndarray
            三角形连接关系
        """
        # 先生成粗网格
        x = np.linspace(xlim[0], xlim[1], n_base)
        y = np.linspace(ylim[0], ylim[1], n_base)
        X, Y = np.meshgrid(x, y)
        base_points = np.column_stack((X.ravel(), Y.ravel()))

        # 根据密度自适应加细
        densities = np.array([density_func(p[0], p[1]) for p in base_points])
        max_rho = np.max(densities)
        if max_rho < 1e-15:
            return base_points, Delaunay(base_points).simplices

        # 在密度高的区域插入额外点
        extra_points = []
        threshold = 0.3 * max_rho
        for i, p in enumerate(base_points):
            if densities[i] > threshold:
                # 在邻域内插入4个额外点
                dx = (xlim[1] - xlim[0]) / n_base * 0.25
                dy = (ylim[1] - ylim[0]) / n_base * 0.25
                offsets = [(-dx, -dy), (dx, -dy), (-dx, dy), (dx, dy)]
                for ox, oy in offsets:
                    px, py = p[0] + ox, p[1] + oy
                    if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                        extra_points.append([px, py])

        if extra_points:
            all_points = np.vstack([base_points, np.array(extra_points)])
        else:
            all_points = base_points

        tri = Delaunay(all_points)
        return all_points, tri.simplices

    def triangle_area(self, points: np.ndarray,
                      triangles: np.ndarray) -> np.ndarray:
        """
        计算所有三角形的面积。

        Area = 0.5 |x₁(y₂ - y₃) + x₂(y₃ - y₁) + x₃(y₁ - y₂)|

        Parameters
        ----------
        points : np.ndarray
            节点坐标
        triangles : np.ndarray
            三角形索引

        Returns
        -------
        np.ndarray
            面积数组
        """
        p1 = points[triangles[:, 0]]
        p2 = points[triangles[:, 1]]
        p3 = points[triangles[:, 2]]
        area = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        return area

    def triangle_quality(self, points: np.ndarray,
                         triangles: np.ndarray) -> np.ndarray:
        """
        计算三角形质量度量 (0~1, 1为等边三角形)。

        q = 4√3 · Area / (a² + b² + c²)

        Parameters
        ----------
        points : np.ndarray
            节点坐标
        triangles : np.ndarray
            三角形索引

        Returns
        -------
        np.ndarray
            质量度量数组
        """
        p1 = points[triangles[:, 0]]
        p2 = points[triangles[:, 1]]
        p3 = points[triangles[:, 2]]

        a2 = np.sum((p2 - p3) ** 2, axis=1)
        b2 = np.sum((p3 - p1) ** 2, axis=1)
        c2 = np.sum((p1 - p2) ** 2, axis=1)

        area = self.triangle_area(points, triangles)
        denom = a2 + b2 + c2
        quality = np.zeros_like(area)
        mask = denom > 1e-15
        quality[mask] = 4.0 * np.sqrt(3.0) * area[mask] / denom[mask]
        return quality

    def adjacency_count(self, node_num: int,
                        triangles: np.ndarray) -> np.ndarray:
        """
        统计每个节点的邻接节点数。

        基于1330_triangulation的邻接计数思想。

        Parameters
        ----------
        node_num : int
            节点总数
        triangles : np.ndarray
            三角形索引

        Returns
        -------
        np.ndarray
            每个节点的邻接数
        """
        adjacency = [set() for _ in range(node_num)]
        for tri in triangles:
            i, j, k = tri
            adjacency[i].update([j, k])
            adjacency[j].update([i, k])
            adjacency[k].update([i, j])
        counts = np.array([len(s) for s in adjacency])
        return counts

    def boundary_nodes(self, triangles: np.ndarray) -> np.ndarray:
        """
        识别边界节点（基于边出现次数：内部边出现2次，边界边出现1次）。

        Parameters
        ----------
        triangles : np.ndarray
            三角形索引

        Returns
        -------
        np.ndarray
            边界节点索引
        """
        edge_count = {}
        for tri in triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                key = tuple(sorted(e))
                edge_count[key] = edge_count.get(key, 0) + 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        boundary_nodes = set()
        for e in boundary_edges:
            boundary_nodes.update(e)
        return np.array(sorted(boundary_nodes))

    def integrate_scalar(self, points: np.ndarray, triangles: np.ndarray,
                         scalar_values: np.ndarray) -> float:
        """
        在三角网格上积分标量场。

        ∫ f(x,y) dA ≈ Σ_k f_avg,k · Area_k

        Parameters
        ----------
        points : np.ndarray
            节点坐标
        triangles : np.ndarray
            三角形索引
        scalar_values : np.ndarray
            节点上的标量值

        Returns
        -------
        float
            积分值
        """
        areas = self.triangle_area(points, triangles)
        # 每个三角形上取三个节点值的平均
        tri_vals = scalar_values[triangles]
        avg_vals = np.mean(tri_vals, axis=1)
        integral = np.sum(avg_vals * areas)
        return float(integral)

    def gradient_scalar(self, points: np.ndarray, triangles: np.ndarray,
                        scalar_values: np.ndarray) -> np.ndarray:
        """
        计算标量场在三角网格上的分段常数梯度。

        在三角形内，线性插值给出常梯度:
        ∇f = [ (f₂ - f₁)(y₃ - y₁) - (f₃ - f₁)(y₂ - y₁) ] / (2A)
             [ (f₃ - f₁)(x₂ - x₁) - (f₂ - f₁)(x₃ - x₁) ] / (2A)

        Parameters
        ----------
        points : np.ndarray
            节点坐标
        triangles : np.ndarray
            三角形索引
        scalar_values : np.ndarray
            节点标量值

        Returns
        -------
        np.ndarray
            每个三角形的梯度 (M, 2)
        """
        p1 = points[triangles[:, 0]]
        p2 = points[triangles[:, 1]]
        p3 = points[triangles[:, 2]]
        f1 = scalar_values[triangles[:, 0]]
        f2 = scalar_values[triangles[:, 1]]
        f3 = scalar_values[triangles[:, 2]]

        area = self.triangle_area(points, triangles)
        area = np.where(area < 1e-15, 1e-15, area)

        df21 = f2 - f1
        df31 = f3 - f1

        dx21 = p2[:, 0] - p1[:, 0]
        dx31 = p3[:, 0] - p1[:, 0]
        dy21 = p2[:, 1] - p1[:, 1]
        dy31 = p3[:, 1] - p1[:, 1]

        dfdx = (df21 * dy31 - df31 * dy21) / (2.0 * area)
        dfdy = (df31 * dx21 - df21 * dx31) / (2.0 * area)

        grad = np.column_stack((dfdx, dfdy))
        return grad
