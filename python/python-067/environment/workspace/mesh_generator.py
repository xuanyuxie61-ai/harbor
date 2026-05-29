# -*- coding: utf-8 -*-
"""
mesh_generator.py
裂隙介质计算网格生成与细化模块

融合种子项目：
    - 1378_usa_cvt_geo：质心 Voronoi 剖分 (CVT) 采样
    - 1338_triangulation_l2q：线性到二次三角元的网格升级
    - 578_image_double：网格上采样/插值

在水文地质数值模拟中，高质量的计算网格对渗流和溶质运移的精度至关重要。
本模块提供：
    1. CVT 优化采样点生成（用于裂隙面内采样）
    2. Delaunay 三角剖分
    3. 线性到二次三角元的升级（中点插值）
    4. 网格上采样与细化

核心公式：
    CVT 能量泛函：
        F(z_1, ..., z_n) = Σ ∫_{V_i} ρ(x) ||x - z_i||² dA
    
    Lloyd 松弛迭代：
        z_i^{(k+1)} = centroid(V_i^{(k)})
    
    二次三角元形函数（6节点）：
        N_1 = (2ξ - 1)ξ
        N_2 = (2η - 1)η
        N_3 = (2ζ - 1)ζ, ζ = 1 - ξ - η
        N_4 = 4ξη
        N_5 = 4ηζ
        N_6 = 4ζξ
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Delaunay


class MeshGenerator:
    """
    裂隙介质计算网格生成器

    基于 CVT (Centroidal Voronoi Tessellation) 和 Delaunay 三角剖分，
    为裂隙渗流模拟生成高质量计算网格。
    """

    def __init__(self, domain: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)):
        """
        Parameters
        ----------
        domain : tuple
            模拟区域 (xmin, xmax, ymin, ymax)
        """
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.points = np.zeros((0, 2))
        self.triangles = np.zeros((0, 3), dtype=int)
        self.quadratic_nodes = np.zeros((0, 2))
        self.quadratic_triangles = np.zeros((0, 6), dtype=int)

    def generate_uniform_grid(self, nx: int, ny: int) -> np.ndarray:
        """
        生成均匀网格节点

        Parameters
        ----------
        nx, ny : int
            x 和 y 方向节点数

        Returns
        -------
        np.ndarray
            节点坐标 (n_points, 2)
        """
        if nx <= 1 or ny <= 1:
            raise ValueError("nx 和 ny 必须大于 1")
        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        xv, yv = np.meshgrid(x, y)
        points = np.column_stack([xv.ravel(), yv.ravel()])
        self.points = points
        return points

    def generate_random_points(self, n_points: int, seed: int = 42) -> np.ndarray:
        """
        在区域内随机生成节点

        Parameters
        ----------
        n_points : int
            节点数量
        seed : int
            随机种子

        Returns
        -------
        np.ndarray
            节点坐标 (n_points, 2)
        """
        if n_points <= 0:
            raise ValueError("n_points 必须为正")
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xmin, self.xmax, n_points)
        y = rng.uniform(self.ymin, self.ymax, n_points)
        points = np.column_stack([x, y])
        self.points = points
        return points

    def cvt_relaxation(self, n_points: int, n_iterations: int = 10,
                       n_samples: int = 5000, seed: int = 42) -> np.ndarray:
        """
        质心 Voronoi 剖分 (CVT) 松弛优化

        基于 usa_cvt_geo_estimate 的 Lloyd 松弛算法：
            1. 随机采样区域内点
            2. 将采样点分配到最近的生成点
            3. 更新生成点为其 Voronoi 单元的质心
            4. 重复直到收敛

        Parameters
        ----------
        n_points : int
            生成点数量
        n_iterations : int
            Lloyd 松弛迭代次数
        n_samples : int
            每次迭代的采样点数
        seed : int
            随机种子

        Returns
        -------
        np.ndarray
            CVT 优化后的节点坐标 (n_points, 2)
        """
        if n_points <= 0:
            raise ValueError("n_points 必须为正")

        rng = np.random.default_rng(seed)
        # 初始化生成点
        generators = rng.uniform(
            [self.xmin, self.ymin], [self.xmax, self.ymax], size=(n_points, 2)
        )

        for _ in range(n_iterations):
            # 随机采样
            samples = rng.uniform(
                [self.xmin, self.ymin], [self.xmax, self.ymax], size=(n_samples, 2)
            )

            # 找到每个采样点最近的生成点
            dists = np.linalg.norm(samples[:, None, :] - generators[None, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)

            # 更新生成点为 Voronoi 单元质心
            new_generators = np.zeros_like(generators)
            counts = np.zeros(n_points)
            for i in range(n_samples):
                gi = nearest[i]
                new_generators[gi] += samples[i]
                counts[gi] += 1

            for j in range(n_points):
                if counts[j] > 0:
                    generators[j] = new_generators[j] / counts[j]
                else:
                    # 空单元重新随机化
                    generators[j] = rng.uniform(
                        [self.xmin, self.ymin], [self.xmax, self.ymax]
                    )

        self.points = generators
        return generators

    def delaunay_triangulation(self) -> np.ndarray:
        """
        对当前节点进行 Delaunay 三角剖分

        Returns
        -------
        np.ndarray
            三角形索引 (n_triangles, 3)
        """
        if len(self.points) < 3:
            raise ValueError("至少需要 3 个节点才能进行三角剖分")
        tri = Delaunay(self.points)
        self.triangles = tri.simplices
        return tri.simplices

    def upgrade_to_quadratic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将线性三角元 (3节点) 升级为二次三角元 (6节点)

        基于 triangulation_l2q 的中点插值算法：
            - 保留原 3 个顶点
            - 在每条边的中点处插入新节点
            - 二次元的 6 个节点顺序：3顶点 + 3边中点

        Returns
        -------
        tuple
            (quadratic_nodes, quadratic_triangles)
            quadratic_nodes: (n_nodes, 2)
            quadratic_triangles: (n_triangles, 6)
        """
        if len(self.triangles) == 0:
            raise ValueError("先进行三角剖分")

        n_tri = len(self.triangles)
        n_orig = len(self.points)

        # 保留原始节点
        nodes = [p for p in self.points]
        tri_quad = np.zeros((n_tri, 6), dtype=int)
        tri_quad[:, :3] = self.triangles

        # 边到全局中点节点的映射
        edge_to_mid = {}
        next_node_idx = n_orig

        for t in range(n_tri):
            tri = self.triangles[t]
            for e in range(3):
                i1 = tri[e]
                i2 = tri[(e + 1) % 3]
                edge = tuple(sorted([i1, i2]))

                if edge not in edge_to_mid:
                    # 创建中点节点
                    mid = 0.5 * (self.points[i1] + self.points[i2])
                    nodes.append(mid)
                    edge_to_mid[edge] = next_node_idx
                    next_node_idx += 1

                tri_quad[t, 3 + e] = edge_to_mid[edge]

        self.quadratic_nodes = np.array(nodes)
        self.quadratic_triangles = tri_quad
        return self.quadratic_nodes, self.quadratic_triangles

    def refine_mesh(self, n_refinements: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 image_double 思想进行网格细化（中点细分）

        每次细化将每个三角形分成 4 个小三角形：
            - 在每条边中点处插入新节点
            - 连接中点形成 4 个子三角形

        Parameters
        ----------
        n_refinements : int
            细化次数

        Returns
        -------
        tuple
            (refined_points, refined_triangles)
        """
        if n_refinements < 0:
            raise ValueError("n_refinements 必须为非负")

        points = self.points.copy()
        triangles = self.triangles.copy()

        for _ in range(n_refinements):
            n_points = len(points)
            n_tri = len(triangles)

            # 边到全局索引映射
            edge_map = {}
            new_points = list(points)
            next_idx = n_points

            new_triangles = []

            for t in range(n_tri):
                tri = triangles[t]
                mid_indices = []
                for e in range(3):
                    edge = tuple(sorted([tri[e], tri[(e + 1) % 3]]))
                    if edge not in edge_map:
                        mid = 0.5 * (points[edge[0]] + points[edge[1]])
                        new_points.append(mid)
                        edge_map[edge] = next_idx
                        next_idx += 1
                    mid_indices.append(edge_map[edge])

                m0, m1, m2 = mid_indices
                v0, v1, v2 = tri

                # 4 个子三角形
                new_triangles.append([v0, m0, m2])
                new_triangles.append([v1, m1, m0])
                new_triangles.append([v2, m2, m1])
                new_triangles.append([m0, m1, m2])

            points = np.array(new_points)
            triangles = np.array(new_triangles, dtype=int)

        self.points = points
        self.triangles = triangles
        return points, triangles

    def compute_triangle_quality(self) -> np.ndarray:
        """
        计算三角形质量指标（内切圆/外接圆半径比）

        质量指标 q = 2 * r_in / R_circ，q ∈ (0, 1]
        q = 1 为等边三角形（最优）

        Returns
        -------
        np.ndarray
            每个三角形的质量指标
        """
        if len(self.triangles) == 0:
            return np.array([])

        qualities = []
        for tri in self.triangles:
            p0, p1, p2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p0 - p2)

            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-20))

            # 外接圆半径
            if area < 1e-12:
                qualities.append(0.0)
                continue
            R = (a * b * c) / (4.0 * area)

            # 内切圆半径
            r = area / s

            q = 2.0 * r / R if R > 1e-12 else 0.0
            qualities.append(min(max(q, 0.0), 1.0))

        return np.array(qualities)

    def mesh_statistics(self) -> dict:
        """返回网格统计信息"""
        stats = {
            "n_points": len(self.points),
            "n_triangles": len(self.triangles),
            "n_quadratic_nodes": len(self.quadratic_nodes),
            "n_quadratic_triangles": len(self.quadratic_triangles)
        }

        if len(self.triangles) > 0:
            q = self.compute_triangle_quality()
            stats["min_quality"] = float(np.min(q))
            stats["mean_quality"] = float(np.mean(q))
            stats["max_quality"] = float(np.max(q))

        return stats
