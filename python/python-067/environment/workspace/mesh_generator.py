# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Delaunay


class MeshGenerator:

    def __init__(self, domain: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)):
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.points = np.zeros((0, 2))
        self.triangles = np.zeros((0, 3), dtype=int)
        self.quadratic_nodes = np.zeros((0, 2))
        self.quadratic_triangles = np.zeros((0, 6), dtype=int)

    def generate_uniform_grid(self, nx: int, ny: int) -> np.ndarray:
        if nx <= 1 or ny <= 1:
            raise ValueError("nx 和 ny 必须大于 1")
        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        xv, yv = np.meshgrid(x, y)
        points = np.column_stack([xv.ravel(), yv.ravel()])
        self.points = points
        return points

    def generate_random_points(self, n_points: int, seed: int = 42) -> np.ndarray:
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
        if n_points <= 0:
            raise ValueError("n_points 必须为正")

        rng = np.random.default_rng(seed)

        generators = rng.uniform(
            [self.xmin, self.ymin], [self.xmax, self.ymax], size=(n_points, 2)
        )

        for _ in range(n_iterations):

            samples = rng.uniform(
                [self.xmin, self.ymin], [self.xmax, self.ymax], size=(n_samples, 2)
            )


            dists = np.linalg.norm(samples[:, None, :] - generators[None, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)


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

                    generators[j] = rng.uniform(
                        [self.xmin, self.ymin], [self.xmax, self.ymax]
                    )

        self.points = generators
        return generators

    def delaunay_triangulation(self) -> np.ndarray:
        if len(self.points) < 3:
            raise ValueError("至少需要 3 个节点才能进行三角剖分")
        tri = Delaunay(self.points)
        self.triangles = tri.simplices
        return tri.simplices

    def upgrade_to_quadratic(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.triangles) == 0:
            raise ValueError("先进行三角剖分")

        n_tri = len(self.triangles)
        n_orig = len(self.points)


        nodes = [p for p in self.points]
        tri_quad = np.zeros((n_tri, 6), dtype=int)
        tri_quad[:, :3] = self.triangles


        edge_to_mid = {}
        next_node_idx = n_orig

        for t in range(n_tri):
            tri = self.triangles[t]
            for e in range(3):
                i1 = tri[e]
                i2 = tri[(e + 1) % 3]
                edge = tuple(sorted([i1, i2]))

                if edge not in edge_to_mid:

                    mid = 0.5 * (self.points[i1] + self.points[i2])
                    nodes.append(mid)
                    edge_to_mid[edge] = next_node_idx
                    next_node_idx += 1

                tri_quad[t, 3 + e] = edge_to_mid[edge]

        self.quadratic_nodes = np.array(nodes)
        self.quadratic_triangles = tri_quad
        return self.quadratic_nodes, self.quadratic_triangles

    def refine_mesh(self, n_refinements: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if n_refinements < 0:
            raise ValueError("n_refinements 必须为非负")

        points = self.points.copy()
        triangles = self.triangles.copy()

        for _ in range(n_refinements):
            n_points = len(points)
            n_tri = len(triangles)


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


            if area < 1e-12:
                qualities.append(0.0)
                continue
            R = (a * b * c) / (4.0 * area)


            r = area / s

            q = 2.0 * r / R if R > 1e-12 else 0.0
            qualities.append(min(max(q, 0.0), 1.0))

        return np.array(qualities)

    def mesh_statistics(self) -> dict:
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
