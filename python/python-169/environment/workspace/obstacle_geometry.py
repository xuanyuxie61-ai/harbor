
import numpy as np
from typing import List, Tuple






def triangle_signed_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    p1, p2, p3 = map(np.asarray, (p1, p2, p3))
    return 0.5 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))


def triangle_signed_area_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    p1, p2, p3 = map(np.asarray, (p1, p2, p3))
    cross = np.cross(p2 - p1, p3 - p1)
    return 0.5 * np.linalg.norm(cross)


def orient_triangles_ccw(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    if nodes.shape[1] not in (2, 3):
        raise ValueError("节点必须为2D或3D坐标")
    corrected = elements.copy()
    for idx, tri in enumerate(elements):
        if nodes.shape[1] == 2:
            area = triangle_signed_area_2d(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]])
        else:

            area = triangle_signed_area_2d(
                nodes[tri[0], :2], nodes[tri[1], :2], nodes[tri[2], :2]
            )
        if area < 0:

            corrected[idx, 1], corrected[idx, 2] = corrected[idx, 2], corrected[idx, 1]
    return corrected






def _prism_witherden_rule_precision(p: int) -> Tuple[np.ndarray, np.ndarray]:
    if p < 0 or p > 5:
        p = 5


    if p == 0:
        pts = np.array([[1.0/3.0, 1.0/3.0, 0.5]])
        w = np.array([1.0])

    elif p == 1:
        pts = np.array([[1.0/3.0, 1.0/3.0, 0.5]])
        w = np.array([1.0])

    elif p == 2:
        a = 1.0 / 3.0
        b = 0.059715871789770
        c = 0.797426985353087
        pts = np.array([
            [a, a, 0.5],
            [b, 0.5*(1-b), 0.211324865405187],
            [b, 0.5*(1-b), 0.788675134594813],
            [c, 0.5*(1-c), 0.211324865405187],
            [c, 0.5*(1-c), 0.788675134594813],
        ])
        w = np.array([0.225, 0.132394152788506, 0.132394152788506,
                      0.125939180544827, 0.125939180544827])

    elif p == 3:
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        z1 = 0.330009478207572
        z2 = 0.669990521792428
        pts = np.array([
            [a, b, z1], [b, a, z1], [b, b, z1],
            [a, b, z2], [b, a, z2], [b, b, z2],
            [d, c, 0.5], [c, d, 0.5],
        ])
        w = np.array([0.053167620283302, 0.053167620283302, 0.053167620283302,
                      0.053167620283302, 0.053167620283302, 0.053167620283302,
                      0.111690794839006, 0.111690794839006])

    elif p == 4:
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        z1 = 0.221962689113754
        z2 = 0.5
        z3 = 0.778037310886246
        pts = np.array([
            [a, b, z1], [b, a, z1], [b, b, z1],
            [a, b, z2], [b, a, z2], [b, b, z2],
            [a, b, z3], [b, a, z3], [b, b, z3],
            [d, c, z2], [c, d, z2],
        ])
        w = np.array([0.036848902546363, 0.036848902546363, 0.036848902546363,
                      0.046046366595935, 0.046046366595935, 0.046046366595935,
                      0.036848902546363, 0.036848902546363, 0.036848902546363,
                      0.077667095375523, 0.077667095375523])

    else:
        a1 = 0.333333333333333
        a2 = 0.170569307751760
        a3 = 0.050547228317031
        a4 = 0.459292588292723
        b4 = 0.728492392955404
        c4 = 0.263112829634638
        z1 = 0.169990521792428
        z2 = 0.380003113463505
        z3 = 0.619996886536495
        z4 = 0.830009478207572
        pts = np.array([
            [a1, a1, z2], [a1, a1, z3],
            [a2, a2, z1], [a2, a2, z4],
            [a3, a3, z2], [a3, a3, z3],
            [a4, b4, z1], [b4, a4, z1], [a4, b4, z4], [b4, a4, z4],
            [a4, c4, z2], [c4, a4, z2], [a4, c4, z3], [c4, a4, z3],
            [a3, a3, z1], [a3, a3, z4],
        ])
        w = np.array([0.065783135440355, 0.065783135440355,
                      0.034437368688912, 0.034437368688912,
                      0.028609231658563, 0.028609231658563,
                      0.027231240701046, 0.027231240701046,
                      0.027231240701046, 0.027231240701046,
                      0.032261482794736, 0.032261482794736,
                      0.032261482794736, 0.032261482794736,
                      0.010389256501586, 0.010389256501586])
    return pts, w


def integrate_over_prism(f, precision: int = 5) -> float:
    pts, w = _prism_witherden_rule_precision(precision)
    vals = f(pts)
    vals = np.asarray(vals).reshape(-1)
    return 0.5 * np.sum(w * vals)


class PolyhedralObstacle:

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, density: float = 1.0):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.density = float(density)
        if self.triangles.max() >= self.vertices.shape[0]:
            raise ValueError("三角形索引超出顶点范围")

        self.triangles = orient_triangles_ccw(self.vertices, self.triangles)

        self._precompute_faces()

        self.mass, self.centroid, self.inertia = self._compute_mass_properties()

    def _precompute_faces(self):
        n_tri = self.triangles.shape[0]
        self.face_normals = np.zeros((n_tri, 3), dtype=float)
        self.face_areas = np.zeros(n_tri, dtype=float)
        self.face_centers = np.zeros((n_tri, 3), dtype=float)
        for i in range(n_tri):
            tri = self.triangles[i]
            p0, p1, p2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]
            n_vec = np.cross(p1 - p0, p2 - p0)
            area = 0.5 * np.linalg.norm(n_vec)
            if area > 1e-14:
                normal = n_vec / (2.0 * area)
            else:
                normal = np.array([0.0, 0.0, 1.0])
            self.face_normals[i] = normal
            self.face_areas[i] = area
            self.face_centers[i] = (p0 + p1 + p2) / 3.0

    def _compute_mass_properties(self) -> Tuple[float, np.ndarray, np.ndarray]:
        ref = np.mean(self.vertices, axis=0)
        total_vol = 0.0
        total_mass = 0.0
        centroid_sum = np.zeros(3, dtype=float)
        inertia_sum = np.zeros((3, 3), dtype=float)
        for i in range(self.triangles.shape[0]):
            tri = self.triangles[i]
            p0 = self.vertices[tri[0]] - ref
            p1 = self.vertices[tri[1]] - ref
            p2 = self.vertices[tri[2]] - ref

            vol = np.dot(p0, np.cross(p1, p2)) / 6.0
            if vol < 0:

                p1, p2 = p2.copy(), p1.copy()
                vol = -vol
            if vol < 1e-14:
                continue

            c_tet = 0.25 * (p0 + p1 + p2)



            def f_inertia(pts_local):



                pass
            total_vol += vol
            centroid_sum += vol * c_tet
        if total_vol < 1e-14:
            total_vol = 1e-14
        centroid = ref + centroid_sum / total_vol
        mass = self.density * total_vol

        inertia = np.eye(3) * mass * 0.1
        return mass, centroid, inertia

    def signed_distance(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float).reshape(3)
        min_dist_sq = np.inf
        min_sign = 1.0
        for i in range(self.triangles.shape[0]):
            tri = self.triangles[i]
            p0, p1, p2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]
            n = self.face_normals[i]

            d_plane = np.dot(n, point - p0)

            proj = point - d_plane * n

            v0 = p2 - p0
            v1 = p1 - p0
            v2 = proj - p0
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < 1e-14:
                continue
            u = (dot11 * dot02 - dot01 * dot12) / denom
            v = (dot00 * dot12 - dot01 * dot02) / denom
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9:

                dist = abs(d_plane)
                if dist * dist < min_dist_sq:
                    min_dist_sq = dist * dist
                    min_sign = 1.0 if d_plane >= 0 else -1.0
            else:

                edges = [(p0, p1), (p1, p2), (p2, p0)]
                for a, b in edges:
                    ab = b - a
                    t_proj = np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-14)
                    t_proj = np.clip(t_proj, 0.0, 1.0)
                    closest = a + t_proj * ab
                    diff = point - closest
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq

                        min_sign = 1.0 if np.dot(diff, n) >= 0 else -1.0

                for p in (p0, p1, p2):
                    diff = point - p
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_sign = 1.0
        if min_dist_sq == np.inf:
            return 1e6
        dist = np.sqrt(min_dist_sq)


        return min_sign * dist

    def collision_check(self, point: np.ndarray, safety_margin: float = 0.05) -> bool:
        return self.signed_distance(point) < safety_margin


def generate_box_obstacle(center: np.ndarray, size: np.ndarray,
                          density: float = 1.0) -> PolyhedralObstacle:
    c = np.asarray(center, dtype=float).reshape(3)
    s = np.asarray(size, dtype=float).reshape(3) * 0.5
    vertices = np.array([
        [c[0]-s[0], c[1]-s[1], c[2]-s[2]],
        [c[0]+s[0], c[1]-s[1], c[2]-s[2]],
        [c[0]+s[0], c[1]+s[1], c[2]-s[2]],
        [c[0]-s[0], c[1]+s[1], c[2]-s[2]],
        [c[0]-s[0], c[1]-s[1], c[2]+s[2]],
        [c[0]+s[0], c[1]-s[1], c[2]+s[2]],
        [c[0]+s[0], c[1]+s[1], c[2]+s[2]],
        [c[0]-s[0], c[1]+s[1], c[2]+s[2]],
    ], dtype=float)
    triangles = np.array([
        [0,1,2], [0,2,3],
        [4,6,5], [4,7,6],
        [0,4,5], [0,5,1],
        [2,6,7], [2,7,3],
        [0,3,7], [0,7,4],
        [1,5,6], [1,6,2],
    ], dtype=int)
    return PolyhedralObstacle(vertices, triangles, density)


def generate_sphere_obstacle(center: np.ndarray, radius: float,
                             n_segments: int = 16, density: float = 1.0) -> PolyhedralObstacle:
    c = np.asarray(center, dtype=float).reshape(3)
    r = float(radius)
    vertices = []
    triangles = []

    vertices.append([0.0, 0.0, r])
    for i in range(1, n_segments):
        theta = np.pi * i / n_segments
        z = r * np.cos(theta)
        ring_r = r * np.sin(theta)
        for j in range(n_segments):
            phi = 2 * np.pi * j / n_segments
            x = ring_r * np.cos(phi)
            y = ring_r * np.sin(phi)
            vertices.append([x, y, z])
    vertices.append([0.0, 0.0, -r])
    vertices = np.array(vertices, dtype=float) + c

    for j in range(n_segments):
        j1 = (j + 1) % n_segments
        triangles.append([0, 1 + j, 1 + j1])

    for i in range(n_segments - 2):
        base = 1 + i * n_segments
        next_base = 1 + (i + 1) * n_segments
        for j in range(n_segments):
            j1 = (j + 1) % n_segments
            a = base + j
            b = base + j1
            c_idx = next_base + j
            d_idx = next_base + j1
            triangles.append([a, c_idx, b])
            triangles.append([b, c_idx, d_idx])

    south = len(vertices) - 1
    base = 1 + (n_segments - 2) * n_segments
    for j in range(n_segments):
        j1 = (j + 1) % n_segments
        triangles.append([south, base + j1, base + j])
    triangles = np.array(triangles, dtype=int)
    return PolyhedralObstacle(vertices, triangles, density)
