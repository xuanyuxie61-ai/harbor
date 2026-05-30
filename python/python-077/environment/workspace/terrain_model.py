
import numpy as np
from typing import List, Tuple, Optional


class TerrainProfile:

    def __init__(self):
        self.points = np.zeros((0, 2))
        self.closed = False

    def add_point(self, x: float, z: float):
        self.points = np.vstack([self.points, [x, z]])

    def close_profile(self):
        if len(self.points) > 0 and not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])
        self.closed = True

    def polygon_area(self) -> float:
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
        p1, p2, p3 = self.points[i], self.points[j], self.points[k]
        area = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        return area

    def elevation_at(self, x: float) -> float:
        if len(self.points) == 0:
            return 0.0
        pts = self.points
        if x <= pts[0, 0]:
            return pts[0, 1]
        if x >= pts[-1, 0]:
            return pts[-1, 1]


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
        delta = 1.0
        z_plus = self.elevation_at(x + delta)
        z_minus = self.elevation_at(x - delta)
        return (z_plus - z_minus) / (2.0 * delta)


class FEM2DMesh:

    def __init__(self, xmin: float = 0.0, xmax: float = 5000.0,
                 ymin: float = 0.0, ymax: float = 5000.0,
                 nx: int = 20, ny: int = 20):
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
        elem = self.elements[eidx]
        n1, n2, n3, n4 = elem
        p1 = self.nodes[n1]
        p2 = self.nodes[n2]
        p3 = self.nodes[n3]
        p4 = self.nodes[n4]


        area1 = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )

        area2 = 0.5 * abs(
            p1[0] * (p3[1] - p4[1]) +
            p3[0] * (p4[1] - p1[1]) +
            p4[0] * (p1[1] - p3[1])
        )
        return area1 + area2

    def total_domain_area(self) -> float:
        return sum(self.element_area(i) for i in range(self.n_elements()))

    def find_element_containing(self, x: float, y: float) -> int:
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
        p = self.nodes[node_idx]
        neighbors = []
        for i, q in enumerate(self.nodes):
            if i != node_idx and np.linalg.norm(p - q) <= radius:
                neighbors.append(i)
        return neighbors
