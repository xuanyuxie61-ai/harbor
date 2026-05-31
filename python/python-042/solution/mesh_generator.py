"""
mesh_generator.py

Unstructured mesh generation and topological utilities for mantle geometry.

Core seed mappings:
- 890_polygon_triangulate -> ear-clipping triangulation of polygonal cross-sections
- 1394_voronoi_city       -> Voronoi tessellation for surface plate partitioning
- 1372_unicycle           -> cyclic permutation indexing for boundary node ordering

Scientific formulas:
- Triangle area (signed): A = 0.5 * [(xb−xa)(yc−ya) − (xc−xa)(yb−ya)]
- Polygon area: A = 0.5 * Σ_{i=1}^{n} (x_i y_{i+1} − x_{i+1} y_i)
- Voronoi cell area for generator p_i: A_i = ∫_{V_i} dA
"""

import numpy as np
from typing import List, Tuple, Optional


class PolygonTriangulator:
    """
    Ear-clipping triangulation for arbitrary simple polygons.
    Adapted from seed 890_polygon_triangulate (O'Rourke algorithm).
    """
    def __init__(self, angle_tol_deg: float = 5.7e-05):
        self.angle_tol = angle_tol_deg * np.pi / 180.0  # convert to radians

    def _triangle_area(self, xa, ya, xb, yb, xc, yc) -> float:
        return 0.5 * ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))

    def _collinear(self, xa, ya, xb, yb, xc, yc) -> bool:
        area = abs(self._triangle_area(xa, ya, xb, yb, xc, yc))
        side_ab_sq = (xa - xb) ** 2 + (ya - yb) ** 2
        side_bc_sq = (xb - xc) ** 2 + (yb - yc) ** 2
        side_ca_sq = (xc - xa) ** 2 + (yc - ya) ** 2
        side_max_sq = max(side_ab_sq, max(side_bc_sq, side_ca_sq))
        eps = np.finfo(float).eps
        if side_max_sq <= eps:
            return True
        return 2.0 * area <= eps * side_max_sq

    def _between(self, xa, ya, xb, yb, xc, yc) -> bool:
        if not self._collinear(xa, ya, xb, yb, xc, yc):
            return False
        if abs(ya - yb) < abs(xa - xb):
            return min(xa, xb) <= xc <= max(xa, xb)
        else:
            return min(ya, yb) <= yc <= max(ya, yb)

    def _intersect_prop(self, xa, ya, xb, yb, xc, yc, xd, yd) -> bool:
        if self._collinear(xa, ya, xb, yb, xc, yc):
            return False
        if self._collinear(xa, ya, xb, yb, xd, yd):
            return False
        if self._collinear(xc, yc, xd, yd, xa, ya):
            return False
        if self._collinear(xc, yc, xd, yd, xb, yb):
            return False
        t1 = self._triangle_area(xa, ya, xb, yb, xc, yc)
        t2 = self._triangle_area(xa, ya, xb, yb, xd, yd)
        t3 = self._triangle_area(xc, yc, xd, yd, xa, ya)
        t4 = self._triangle_area(xc, yc, xd, yd, xb, yb)
        return (bool(t1 > 0) != bool(t2 > 0)) and (bool(t3 > 0) != bool(t4 > 0))

    def _intersect(self, xa, ya, xb, yb, xc, yc, xd, yd) -> bool:
        if self._intersect_prop(xa, ya, xb, yb, xc, yc, xd, yd):
            return True
        if self._between(xa, ya, xb, yb, xc, yc):
            return True
        if self._between(xa, ya, xb, yb, xd, yd):
            return True
        if self._between(xc, yc, xd, yd, xa, ya):
            return True
        if self._between(xc, yc, xd, yd, xb, yb):
            return True
        return False

    def _in_cone(self, im1, ip1, prev_node, next_node, x, y) -> bool:
        n = len(x)
        im2 = prev_node[im1]
        i = next_node[im1]
        t1 = self._triangle_area(x[im1], y[im1], x[i], y[i], x[im2], y[im2])
        t2 = self._triangle_area(x[im1], y[im1], x[ip1], y[ip1], x[im2], y[im2])
        t3 = self._triangle_area(x[ip1], y[ip1], x[im1], y[im1], x[i], y[i])
        if t1 >= 0.0:
            return (t2 > 0.0) and (t3 > 0.0)
        else:
            t4 = self._triangle_area(x[im1], y[im1], x[ip1], y[ip1], x[i], y[i])
            t5 = self._triangle_area(x[ip1], y[ip1], x[im1], y[im1], x[im2], y[im2])
            return not ((t4 >= 0.0) and (t5 >= 0.0))

    def _diagonalie(self, im1, ip1, next_node, x, y) -> bool:
        first = im1
        j = first
        jp1 = next_node[first]
        n = len(x)
        while True:
            if not (j == im1 or j == ip1 or jp1 == im1 or jp1 == ip1):
                if self._intersect(x[im1], y[im1], x[ip1], y[ip1],
                                   x[j], y[j], x[jp1], y[jp1]):
                    return False
            j = jp1
            jp1 = next_node[j]
            if j == first:
                break
        return True

    def _diagonal(self, im1, ip1, prev_node, next_node, x, y) -> bool:
        v1 = self._in_cone(im1, ip1, prev_node, next_node, x, y)
        v2 = self._in_cone(ip1, im1, prev_node, next_node, x, y)
        v3 = self._diagonalie(im1, ip1, next_node, x, y)
        return v1 and v2 and v3

    def triangulate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Triangulate a simple polygon with vertices (x, y).
        Returns triangles array of shape (n-2, 3) with 1-based vertex indices.
        """
        n = len(x)
        if n < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        # Check for consecutive duplicate vertices
        for i in range(n):
            im1 = (i - 1) % n
            if abs(x[i] - x[im1]) < 1e-14 and abs(y[i] - y[im1]) < 1e-14:
                raise ValueError("Two consecutive nodes are identical")
        # Check positive area (counter-clockwise)
        area = 0.0
        im1 = n - 1
        for i in range(n):
            area += x[im1] * y[i] - x[i] * y[im1]
            im1 = i
        area = 0.5 * area
        if area <= 0.0:
            raise ValueError("Polygon has zero or negative area; ensure CCW ordering")
        # Build linked list
        prev_node = np.zeros(n, dtype=int)
        next_node = np.zeros(n, dtype=int)
        prev_node[0] = n - 1
        next_node[0] = 1
        for i in range(1, n - 1):
            prev_node[i] = i - 1
            next_node[i] = i + 1
        prev_node[n - 1] = n - 2
        next_node[n - 1] = 0
        ear = np.zeros(n, dtype=bool)
        for i in range(n):
            ear[i] = self._diagonal(prev_node[i], next_node[i], prev_node, next_node, x, y)
        triangles = np.zeros((n - 2, 3), dtype=int)
        triangle_num = 0
        i2 = 0
        while triangle_num < n - 3:
            if ear[i2]:
                i3 = next_node[i2]
                i4 = next_node[i3]
                i1 = prev_node[i2]
                i0 = prev_node[i1]
                next_node[i1] = i3
                prev_node[i3] = i1
                ear[i1] = self._diagonal(i0, i3, prev_node, next_node, x, y)
                ear[i3] = self._diagonal(i1, i4, prev_node, next_node, x, y)
                triangles[triangle_num, 0] = i3 + 1
                triangles[triangle_num, 1] = i1 + 1
                triangles[triangle_num, 2] = i2 + 1
                triangle_num += 1
            i2 = next_node[i2]
        i3 = next_node[i2]
        i1 = prev_node[i2]
        triangles[triangle_num, 0] = i3 + 1
        triangles[triangle_num, 1] = i1 + 1
        triangles[triangle_num, 2] = i2 + 1
        return triangles


class VoronoiTessellator:
    """
    2D Voronoi diagram computation for surface plate partitioning.
    Adapted from seed 1394_voronoi_city (perpendicular bisector concept).

    For N generators p_i in ℝ², the Voronoi cell V_i is:
        V_i = { x ∈ ℝ² : ‖x − p_i‖ ≤ ‖x − p_j‖, ∀ j ≠ i }
    """
    def __init__(self, bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds

    def compute_cells(self, generators: np.ndarray, grid_res: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute approximate Voronoi cells on a regular grid using
        the generator-distance criterion.

        Returns
        -------
        labels : np.ndarray of shape (grid_res, grid_res)
            Index of nearest generator for each grid point.
        areas : np.ndarray of shape (n_generators,)
            Approximate cell areas.
        """
        generators = np.asarray(generators, dtype=float)
        if generators.ndim != 2 or generators.shape[1] != 2:
            raise ValueError("generators must be array of shape (n, 2)")
        n_gen = generators.shape[0]
        if n_gen < 2:
            raise ValueError("Need at least 2 generators")
        x = np.linspace(self.xmin, self.xmax, grid_res)
        y = np.linspace(self.ymin, self.ymax, grid_res)
        dx = (self.xmax - self.xmin) / (grid_res - 1)
        dy = (self.ymax - self.ymin) / (grid_res - 1)
        cell_area = dx * dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        # Compute distances to all generators
        diffs = points[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1).reshape((grid_res, grid_res))
        areas = np.array([np.count_nonzero(labels == i) * cell_area for i in range(n_gen)])
        return labels, areas

    def perpendicular_bisector(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float]:
        """
        Return line ax + by + c = 0 of the perpendicular bisector of segment p1-p2.
        """
        mid = 0.5 * (p1 + p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # Normal vector to segment is (dx, dy); bisector line is dx*(x-mx) + dy*(y-my) = 0
        a = dx
        b = dy
        c = -(dx * mid[0] + dy * mid[1])
        return a, b, c


class UnicycleIndexer:
    """
    Cyclic permutation (unicycle) indexing for boundary node ordering.
    Adapted from seed 1372_unicycle.

    A unicycle on n elements is a cyclic permutation σ with a single orbit.
    The index form maps i → σ(i). Here we use it for cyclic renumbering
    of nodes along polygonal boundaries.
    """
    @staticmethod
    def index_to_sequence(n: int, u_index: np.ndarray) -> np.ndarray:
        """
        Convert unicycle index vector to sequence.
        u_index[i] = next node after i in the cycle.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        u_index = np.asarray(u_index, dtype=int)
        if len(u_index) != n:
            raise ValueError("u_index length must equal n")
        u = np.zeros(n, dtype=int)
        u[0] = 0
        i = 0
        for j in range(1, n):
            i = u_index[i]
            u[j] = i
            if i == 0 and j < n - 1:
                raise ValueError("Index vector does not represent a single unicycle")
        return u

    @staticmethod
    def create_cycle(n: int, shift: int = 1) -> np.ndarray:
        """
        Create a simple cyclic permutation index: each node points to (i+shift) mod n.
        Ensures gcd(n, shift) = 1 to form a single unicycle.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        import math
        shift = shift % n
        if shift == 0:
            shift = 1
        # Ensure single cycle by finding coprime shift
        while math.gcd(n, shift) != 1 and shift < n:
            shift += 1
        if math.gcd(n, shift) != 1:
            shift = 1
        return (np.arange(n) + shift) % n


class MantleMesh2D:
    """
    2D annular cross-section mesh for mantle convection.
    Combines triangulation and Voronoi concepts.
    """
    def __init__(self, R_inner: float = 0.5, R_outer: float = 1.0):
        if R_inner <= 0 or R_outer <= R_inner:
            raise ValueError("Require 0 < R_inner < R_outer")
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.triangulator = PolygonTriangulator()
        self.voronoi = VoronoiTessellator(bounds=(-R_outer, R_outer, -R_outer, R_outer))

    def generate_annular_sector_mesh(self, n_r: int = 10, n_theta: int = 20,
                                     theta_min: float = 0.0,
                                     theta_max: float = np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a structured triangular mesh for an annular sector.
        Returns nodes (n_nodes, 2), triangles (n_tri, 3), and boundary_nodes.
        """
        if n_r < 2 or n_theta < 2:
            raise ValueError("Grid dimensions must be >= 2")
        r = np.linspace(self.R_inner, self.R_outer, n_r)
        theta = np.linspace(theta_min, theta_max, n_theta)
        nodes = []
        for ri in r:
            for tj in theta:
                nodes.append([ri * np.cos(tj), ri * np.sin(tj)])
        nodes = np.array(nodes)
        # Boundary nodes: inner arc, outer arc, and radial edges
        boundary = set()
        for j in range(n_theta):
            boundary.add(j)  # inner arc
            boundary.add((n_r - 1) * n_theta + j)  # outer arc
        for i in range(n_r):
            boundary.add(i * n_theta)  # left radial
            boundary.add(i * n_theta + n_theta - 1)  # right radial
        boundary_nodes = np.array(sorted(boundary), dtype=int)
        # Structured triangulation
        triangles = []
        for i in range(n_r - 1):
            for j in range(n_theta - 1):
                n0 = i * n_theta + j
                n1 = n0 + 1
                n2 = (i + 1) * n_theta + j
                n3 = n2 + 1
                triangles.append([n0, n2, n1])
                triangles.append([n1, n2, n3])
        triangles = np.array(triangles, dtype=int)
        return nodes, triangles, boundary_nodes

    def triangulate_complex_polygon(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Wrapper for ear-clipping triangulation."""
        return self.triangulator.triangulate(x, y)
