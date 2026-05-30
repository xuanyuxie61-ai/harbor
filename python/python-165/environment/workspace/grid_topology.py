
import numpy as np
from typing import List, Tuple, Optional
from utils import circle_arc_grid, triangle_area_2d, triangle_angles


class GridTopology:

    def __init__(self, nodes: np.ndarray, elements: Optional[np.ndarray] = None):
        self.nodes = np.array(nodes, dtype=np.float64)
        self.n_nodes = self.nodes.shape[0]
        if elements is None:
            self.elements = self.delaunay_triangulate()
        else:
            self.elements = np.array(elements, dtype=np.int32)
        self.n_elements = self.elements.shape[0]
        self.adjacency = self._build_adjacency()

    def delaunay_triangulate(self) -> np.ndarray:
        pts = self.nodes
        n = pts.shape[0]
        if n < 3:
            return np.zeros((0, 3), dtype=np.int32)


        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        dx, dy = maxx - minx, maxy - miny
        dmax = max(dx, dy)
        midx, midy = (minx + maxx) * 0.5, (miny + maxy) * 0.5

        super_tri = np.array([
            [midx - 20 * dmax, midy - dmax],
            [midx + 20 * dmax, midy - dmax],
            [midx, midy + 20 * dmax]
        ], dtype=np.float64)

        triangles = [(0, 1, 2)]
        all_pts = np.vstack([pts, super_tri])

        def in_circumcircle(p, a, b, c):
            ax, ay = a[0] - p[0], a[1] - p[1]
            bx, by = b[0] - p[0], b[1] - p[1]
            cx, cy = c[0] - p[0], c[1] - p[1]
            det = (ax * ax + ay * ay) * (bx * cy - by * cx) \
                - (bx * bx + by * by) * (ax * cy - ay * cx) \
                + (cx * cx + cy * cy) * (ax * by - ay * bx)
            return det > 0

        for p_idx in range(n):
            p = all_pts[p_idx]
            bad_triangles = []
            for t in triangles:
                a, b, c = all_pts[t[0]], all_pts[t[1]], all_pts[t[2]]
                if in_circumcircle(p, a, b, c):
                    bad_triangles.append(t)

            polygon = []
            for t in bad_triangles:
                edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                for e in edges:
                    shared = False
                    for ot in bad_triangles:
                        if t == ot:
                            continue
                        o_edges = [(ot[0], ot[1]), (ot[1], ot[2]), (ot[2], ot[0])]
                        if any((e[0] == oe[1] and e[1] == oe[0]) for oe in o_edges):
                            shared = True
                            break
                    if not shared:
                        polygon.append(e)


            bad_set = set(bad_triangles)
            triangles = [t for t in triangles if t not in bad_set]
            for e in polygon:
                triangles.append((e[0], e[1], p_idx))


        filtered = []
        super_idx = {n, n + 1, n + 2}
        for t in triangles:
            if not any(v in super_idx for v in t):
                filtered.append(t)
        if not filtered:
            return np.zeros((0, 3), dtype=np.int32)
        return np.array(filtered, dtype=np.int32)

    def _build_adjacency(self) -> List[List[int]]:
        adj = [[] for _ in range(self.n_nodes)]
        for tri in self.elements:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = int(tri[i]), int(tri[j])
                    if v not in adj[u]:
                        adj[u].append(v)
                    if u not in adj[v]:
                        adj[v].append(u)
        return adj

    def get_edge_list(self) -> np.ndarray:
        edges = set()
        for tri in self.elements:
            for i in range(3):
                u, v = int(tri[i]), int(tri[(i + 1) % 3])
                if u == v:
                    continue
                edges.add((min(u, v), max(u, v)))
        return np.array(sorted(edges), dtype=np.int32)

    def compute_mesh_quality(self) -> dict:
        if self.n_elements == 0:
            return {"min_angle_deg": 0.0, "area_ratio": 0.0,
                    "mean_area": 0.0, "std_area": 0.0}
        min_angles = []
        area_ratios = []
        areas = []
        for tri in self.elements:
            a = self.nodes[tri[0]]
            b = self.nodes[tri[1]]
            c = self.nodes[tri[2]]
            area = triangle_area_2d(a, b, c)
            areas.append(area)
            angles = triangle_angles(a, b, c)
            min_angles.append(np.degrees(angles.min()))
            lab = np.linalg.norm(b - a)
            lbc = np.linalg.norm(c - b)
            lca = np.linalg.norm(a - c)
            denom = lab**2 + lbc**2 + lca**2
            if denom > 1e-12:
                area_ratios.append(4.0 * np.sqrt(3.0) * area / denom)
            else:
                area_ratios.append(0.0)
        return {
            "min_angle_deg": float(np.min(min_angles)),
            "area_ratio": float(np.mean(area_ratios)),
            "mean_area": float(np.mean(areas)),
            "std_area": float(np.std(areas))
        }

    @staticmethod
    def generate_ring_radial_topology(n_ring: int, n_radial: int,
                                       r_inner: float, r_outer: float,
                                       rng_seed: int = 42) -> "GridTopology":
        rng = np.random.default_rng(rng_seed)
        nodes = []

        inner = circle_arc_grid(0.0, 0.0, r_inner, 0.0, 360.0, n_ring)
        inner = inner[:-1]
        nodes.extend(inner.tolist())

        outer = circle_arc_grid(0.0, 0.0, r_outer, 0.0, 360.0, n_ring)
        outer = outer[:-1]
        nodes.extend(outer.tolist())

        for i in range(n_ring):
            angle = 2.0 * np.pi * i / n_ring
            for j in range(1, n_radial + 1):
                t = j / (n_radial + 1)
                r = r_inner + t * (r_outer - r_inner)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                nodes.append([x, y])
        pts = np.array(nodes, dtype=np.float64)

        pts += rng.normal(0.0, 0.05 * r_inner, pts.shape)
        return GridTopology(pts)
