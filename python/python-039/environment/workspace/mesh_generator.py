
import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import Delaunay


class MeshGenerator:

    def __init__(self, max_area: float = 0.5, min_angle: float = 25.0):
        self.max_area = max_area
        self.min_angle_deg = min_angle
        self.min_angle_rad = np.deg2rad(min_angle)

    def generate_uniform_grid(self, xlim: Tuple[float, float],
                              ylim: Tuple[float, float],
                              nx: int = 40, ny: int = 40) -> Tuple[np.ndarray, np.ndarray]:
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

        x = np.linspace(xlim[0], xlim[1], n_base)
        y = np.linspace(ylim[0], ylim[1], n_base)
        X, Y = np.meshgrid(x, y)
        base_points = np.column_stack((X.ravel(), Y.ravel()))


        densities = np.array([density_func(p[0], p[1]) for p in base_points])
        max_rho = np.max(densities)
        if max_rho < 1e-15:
            return base_points, Delaunay(base_points).simplices


        extra_points = []
        threshold = 0.3 * max_rho
        for i, p in enumerate(base_points):
            if densities[i] > threshold:

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
        adjacency = [set() for _ in range(node_num)]
        for tri in triangles:
            i, j, k = tri
            adjacency[i].update([j, k])
            adjacency[j].update([i, k])
            adjacency[k].update([i, j])
        counts = np.array([len(s) for s in adjacency])
        return counts

    def boundary_nodes(self, triangles: np.ndarray) -> np.ndarray:
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
        areas = self.triangle_area(points, triangles)

        tri_vals = scalar_values[triangles]
        avg_vals = np.mean(tri_vals, axis=1)
        integral = np.sum(avg_vals * areas)
        return float(integral)

    def gradient_scalar(self, points: np.ndarray, triangles: np.ndarray,
                        scalar_values: np.ndarray) -> np.ndarray:
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
