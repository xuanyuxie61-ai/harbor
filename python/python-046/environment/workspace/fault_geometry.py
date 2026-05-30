
import numpy as np
from utils import check_finite, compute_triangle_area, clip_to_range


class FaultMesh:

    def __init__(self, length, width, strike_deg, dip_deg, num_strike, num_dip,
                 adaptivity=False):
        self.length = length
        self.width = width
        self.strike_deg = strike_deg
        self.dip_deg = dip_deg
        self.num_strike = num_strike
        self.num_dip = num_dip
        self.adaptivity = adaptivity


        if adaptivity:
            self.nodes, self.elements = self._build_cvt_adaptive_mesh()
        else:
            self.nodes, self.elements = self._build_regular_quadrilateral_mesh()

        self.num_nodes = self.nodes.shape[0]
        self.num_elements = self.elements.shape[0]


        self.boundary_flags = self._mark_boundary_nodes()

    def _build_regular_quadrilateral_mesh(self):
        nx = self.num_strike + 1
        ny = self.num_dip + 1
        dx = self.length / self.num_strike
        dy = self.width / self.num_dip

        nodes = np.zeros((nx * ny, 2))
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                nodes[idx, 0] = i * dx
                nodes[idx, 1] = j * dy


        elements = []
        for j in range(self.num_dip):
            for i in range(self.num_strike):
                n1 = j * nx + i
                n2 = j * nx + (i + 1)
                n3 = (j + 1) * nx + i
                n4 = (j + 1) * nx + (i + 1)

                elements.append([n1, n2, n4])

                elements.append([n1, n4, n3])

        return nodes, np.array(elements, dtype=int)

    def _build_cvt_adaptive_mesh(self):
        nx = self.num_strike + 1
        ny = self.num_dip + 1
        n_total = nx * ny


        np.random.seed(42)
        pts = np.zeros((n_total, 2))
        idx = 0
        for j in range(ny):
            for i in range(nx):
                pts[idx, 0] = i * self.length / self.num_strike
                pts[idx, 1] = j * self.width / self.num_dip
                idx += 1


        def density(x, y):

            cx, cy = 0.5 * self.length, 0.5 * self.width
            sx, sy = 0.3 * self.length, 0.2 * self.width
            d = np.exp(-((x - cx) ** 2) / (2 * sx ** 2) -
                       ((y - cy) ** 2) / (2 * sy ** 2))
            return 0.5 + 2.0 * d


        n_lloyd = 5
        for _ in range(n_lloyd):



            new_pts = np.zeros_like(pts)
            for k in range(n_total):
                xk, yk = pts[k]

                dists = np.sum((pts - pts[k]) ** 2, axis=1)
                neigh = np.argsort(dists)[1:min(9, n_total)]

                xmin = max(0.0, np.min(pts[neigh, 0]))
                xmax = min(self.length, np.max(pts[neigh, 0]))
                ymin = max(0.0, np.min(pts[neigh, 1]))
                ymax = min(self.width, np.max(pts[neigh, 1]))

                best_pt = pts[k]
                best_w = density(*best_pt)
                for _trial in range(20):
                    rx = xmin + np.random.rand() * (xmax - xmin)
                    ry = ymin + np.random.rand() * (ymax - ymin)
                    w = density(rx, ry)
                    if w > best_w:
                        best_pt = np.array([rx, ry])
                        best_w = w
                new_pts[k] = best_pt
            pts = new_pts.copy()


        from scipy.spatial import Delaunay

        boundary_pts = []
        for i in range(nx):
            boundary_pts.append([i * self.length / self.num_strike, 0.0])
            boundary_pts.append([i * self.length / self.num_strike, self.width])
        for j in range(1, ny - 1):
            boundary_pts.append([0.0, j * self.width / self.num_dip])
            boundary_pts.append([self.length, j * self.width / self.num_dip])
        boundary_pts = np.array(boundary_pts)
        all_pts = np.vstack([pts, boundary_pts])
        tri = Delaunay(all_pts)
        nodes = all_pts
        elements = tri.simplices


        valid = []
        for tri_idx in range(elements.shape[0]):
            c = np.mean(nodes[elements[tri_idx]], axis=0)
            if 0.0 <= c[0] <= self.length and 0.0 <= c[1] <= self.width:
                valid.append(elements[tri_idx])
        elements = np.array(valid, dtype=int)

        return nodes, elements

    def _mark_boundary_nodes(self):
        flags = np.zeros(self.num_nodes, dtype=bool)
        tol = 1e-6
        for i in range(self.num_nodes):
            x, y = self.nodes[i]
            if (abs(x) < tol or abs(x - self.length) < tol or
                    abs(y) < tol or abs(y - self.width) < tol):
                flags[i] = True
        return flags

    def get_element_centroids(self):
        centroids = np.zeros((self.num_elements, 2))
        for e in range(self.num_elements):
            nids = self.elements[e]
            centroids[e] = np.mean(self.nodes[nids], axis=0)
        return centroids

    def map_to_3d(self, origin=np.array([0.0, 0.0, 0.0])):
        strike_rad = np.deg2rad(self.strike_deg)
        dip_rad = np.deg2rad(self.dip_deg)
        nodes_3d = np.zeros((self.num_nodes, 3))
        for i in range(self.num_nodes):
            x2d, y2d = self.nodes[i]
            nodes_3d[i, 0] = origin[0] + x2d * np.cos(strike_rad)
            nodes_3d[i, 1] = origin[1] + x2d * np.sin(strike_rad)
            nodes_3d[i, 2] = origin[2] - y2d * np.sin(dip_rad)
        return nodes_3d

    def element_areas(self):
        areas = np.zeros(self.num_elements)
        for e in range(self.num_elements):
            n1, n2, n3 = self.elements[e]
            areas[e] = compute_triangle_area(
                self.nodes[n1], self.nodes[n2], self.nodes[n3])
        return areas


class SurfaceGrid:

    def __init__(self, x_range, y_range, nx, ny):
        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        self.points = np.column_stack([X.ravel(), Y.ravel()])
        self.num_points = self.points.shape[0]
