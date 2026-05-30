
import numpy as np
from typing import Tuple, List


class TriangularLattice:

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
        self.neighbors = [[] for _ in range(self.nsites)]

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
        a = self.a
        b1 = np.array([2.0 * np.pi / a, -2.0 * np.pi / (a * np.sqrt(3.0))])
        b2 = np.array([0.0, 4.0 * np.pi / (a * np.sqrt(3.0))])
        self.b1 = b1
        self.b2 = b2

        self.bz_vertices = np.array([
            (2.0/3.0) * b1 + (1.0/3.0) * b2,
            (1.0/3.0) * b1 + (2.0/3.0) * b2,
            (-1.0/3.0) * b1 + (1.0/3.0) * b2,
            (-2.0/3.0) * b1 + (-1.0/3.0) * b2,
            (-1.0/3.0) * b1 + (-2.0/3.0) * b2,
            (1.0/3.0) * b1 + (-1.0/3.0) * b2,
        ])

    def reciprocal_lattice_points(self) -> np.ndarray:



        raise NotImplementedError("HOLE 1: 请实现 reciprocal_lattice_points")

    def site_index(self, ix: int, iy: int) -> int:
        ix = ix % self.nx
        iy = iy % self.ny
        return ix + iy * self.nx


def hex_grid_in_brillouin_zone(n_layers: int, bz_vertices: np.ndarray) -> np.ndarray:
    if n_layers < 1:
        return bz_vertices[:1]

    center = np.mean(bz_vertices, axis=0)

    R = np.linalg.norm(bz_vertices[0] - center)

    points = [center]
    hx = R / n_layers
    hy = hx * np.sqrt(3.0) / 2.0
    for layer in range(1, n_layers + 1):

        for dir_idx in range(6):
            angle = dir_idx * np.pi / 3.0
            base = np.array([np.cos(angle), np.sin(angle)]) * hy * layer * (2.0 / np.sqrt(3.0))

            for step in range(layer):
                offset_angle = (dir_idx + 2) * np.pi / 3.0
                offset = np.array([np.cos(offset_angle), np.sin(offset_angle)]) * hx * step
                pt = center + base + offset

                if _point_in_hexagon(pt, bz_vertices):
                    points.append(pt)
    return np.array(points)


def _point_in_hexagon(pt: np.ndarray, vertices: np.ndarray) -> bool:
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
    from scipy.spatial import Delaunay
    if len(k_points) < 3:
        raise ValueError("k_points 数量必须 >= 3")
    tri = Delaunay(k_points)
    triangles = tri.simplices

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
