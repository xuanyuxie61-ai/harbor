
import numpy as np
from typing import Tuple, List


class TriangulatedMembrane:

    def __init__(self, vertices: np.ndarray, elements: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_v = self.vertices.shape[0]
        self.n_e = self.elements.shape[0]
        self.e_order = self.elements.shape[1]
        self._etoe = None
        self._areas = None
        self._normals = None
        self._curvature = None

    def compute_etoe(self) -> np.ndarray:
        e_order = self.e_order
        e_num = self.n_e
        etov = self.elements

        records = []
        for e in range(e_num):
            for s in range(e_order):
                v1 = etov[e, s]
                v2 = etov[e, (s + 1) % e_order]
                vmin = min(v1, v2)
                vmax = max(v1, v2)
                records.append((vmin, vmax, s, e))

        records.sort(key=lambda r: (r[0], r[1]))
        etoe = np.full((e_num, e_order), -1, dtype=np.int64)
        i = 0
        while i < len(records):
            j = i + 1
            while j < len(records) and records[j][0] == records[i][0] and records[j][1] == records[i][1]:
                j += 1
            if j - i == 2:

                _, _, s1, e1 = records[i]
                _, _, s2, e2 = records[i + 1]
                etoe[e1, s1] = e2
                etoe[e2, s2] = e1
            elif j - i > 2:

                _, _, s1, e1 = records[i]
                _, _, s2, e2 = records[i + 1]
                etoe[e1, s1] = e2
                etoe[e2, s2] = e1
            i = j
        self._etoe = etoe
        return etoe

    def compute_element_areas(self) -> np.ndarray:
        v = self.vertices
        e = self.elements
        x1 = v[e[:, 0]]
        x2 = v[e[:, 1]]
        x3 = v[e[:, 2]]
        cross = np.cross(x2 - x1, x3 - x1)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        self._areas = areas
        return areas

    def compute_normals(self) -> np.ndarray:
        v = self.vertices
        e = self.elements
        x1 = v[e[:, 0]]
        x2 = v[e[:, 1]]
        x3 = v[e[:, 2]]
        normals = np.cross(x2 - x1, x3 - x1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normals = normals / norms
        return self._normals

    def compute_mean_curvature(self) -> np.ndarray:
        adj = [set() for _ in range(self.n_v)]
        for tri in self.elements:
            for k in range(3):
                i = tri[k]
                j = tri[(k + 1) % 3]
                adj[i].add(j)
                adj[j].add(i)
        H = np.zeros(self.n_v, dtype=np.float64)
        for i in range(self.n_v):
            neighbors = list(adj[i])
            if len(neighbors) == 0:
                H[i] = 0.0
                continue
            diff = self.vertices[i] - self.vertices[neighbors]
            laplacian = np.mean(diff, axis=0)
            H[i] = 0.5 * np.linalg.norm(laplacian)
        self._curvature = H
        return H

    def bending_energy(self, kappa: float = 20.0) -> float:
        if self._areas is None:
            self.compute_element_areas()
        if self._curvature is None:
            self.compute_mean_curvature()

        H0 = 0.0
        H_elem = np.mean(self._curvature[self.elements], axis=1)
        energy = 0.5 * kappa * np.sum(self._areas * (H_elem - H0) ** 2)
        return float(energy)

    @classmethod
    def create_planar_sheet(cls, nx: int = 16, ny: int = 16,
                            lx: float = 10.0, ly: float = 10.0) -> "TriangulatedMembrane":
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        xv, yv = np.meshgrid(x, y)
        vertices = np.column_stack((xv.ravel(), yv.ravel(), np.zeros(nx * ny)))
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = j * nx + i
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v2 + 1
                elements.append([v0, v1, v2])
                elements.append([v1, v3, v2])
        elements = np.array(elements, dtype=np.int64)
        return cls(vertices, elements)
