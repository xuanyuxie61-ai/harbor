# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Dict
from utils import ensure_positive, clip_to_unit





def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def cartesian_to_spherical(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = v
    r = np.linalg.norm(v)
    if r < 1e-15:
        return 0.0, 0.0
    theta = np.arccos(clip_to_unit(z / r))
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2.0 * np.pi
    return theta, phi


def normalize_to_sphere(v: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(v)
    if r < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return v / r





def create_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1.0,  phi, 0.0], [1.0,  phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0,  phi], [0.0, 1.0,  phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ], dtype=float)

    verts = np.array([normalize_to_sphere(v) for v in verts])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    return verts, faces





def subdivide_mesh(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_faces = []
    edge_midpoint: Dict[Tuple[int, int], int] = {}
    vert_list = verts.tolist()

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in edge_midpoint:
            return edge_midpoint[key]
        mid = normalize_to_sphere(0.5 * (verts[i] + verts[j]))
        idx = len(vert_list)
        vert_list.append(mid)
        edge_midpoint[key] = idx
        return idx

    for tri in faces:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(vert_list), np.array(new_faces, dtype=int)


class SphericalMesh:

    def __init__(self, nsides: int = 2):
        self.nsides = ensure_positive(nsides, "nsides")
        self.vertices, self.faces = create_icosahedron()
        for _ in range(nsides):
            self.vertices, self.faces = subdivide_mesh(self.vertices, self.faces)
        self.n_vertices = len(self.vertices)
        self.n_faces = len(self.faces)
        self._compute_neighbors()

    def _compute_neighbors(self):
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, tri in enumerate(self.faces):
            for ei in range(3):
                v1 = tri[ei]
                v2 = tri[(ei + 1) % 3]
                key = (min(v1, v2), max(v1, v2))
                edge_to_faces.setdefault(key, []).append(fi)

        self.neighbors = np.full((self.n_faces, 3), -1, dtype=int)
        for fi, tri in enumerate(self.faces):
            for ei in range(3):
                v1 = tri[ei]
                v2 = tri[(ei + 1) % 3]
                key = (min(v1, v2), max(v1, v2))
                faces_sharing = edge_to_faces[key]
                for fj in faces_sharing:
                    if fj != fi:
                        self.neighbors[fi, ei] = fj
                        break

    def face_area(self, face_idx: int) -> float:
        tri = self.faces[face_idx]
        v0, v1, v2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]

        a = np.arccos(clip_to_unit(np.dot(v1, v2)))
        b = np.arccos(clip_to_unit(np.dot(v2, v0)))
        c = np.arccos(clip_to_unit(np.dot(v0, v1)))
        s = 0.5 * (a + b + c)

        tan_s2 = np.tan(max(s / 2.0, 1e-12))
        tan_sa = np.tan(max((s - a) / 2.0, 1e-12))
        tan_sb = np.tan(max((s - b) / 2.0, 1e-12))
        tan_sc = np.tan(max((s - c) / 2.0, 1e-12))
        tan_E4 = np.sqrt(tan_s2 * tan_sa * tan_sb * tan_sc)
        E = 4.0 * np.arctan(tan_E4)
        return E

    def total_area(self) -> float:
        return sum(self.face_area(i) for i in range(self.n_faces))

    def node_angles(self, node_idx: int) -> Tuple[float, float]:
        return cartesian_to_spherical(self.vertices[node_idx])

    def write_mesh(self, prefix: str):
        np.savetxt(f"{prefix}_nodes.txt", self.vertices, fmt="%.12e")
        np.savetxt(f"{prefix}_elements.txt", self.faces + 1, fmt="%d")
        np.savetxt(f"{prefix}_neighbors.txt", self.neighbors + 1, fmt="%d")
