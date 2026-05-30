
import numpy as np


def point_in_triangle_2d(A, B, C, P):
    A, B, C, P = map(np.asarray, (A, B, C, P))
    cross1 = np.cross(B - A, P - A)
    cross2 = np.cross(C - B, P - B)
    cross3 = np.cross(A - C, P - C)

    eps = 1e-12
    s1 = np.sign(cross1)
    s2 = np.sign(cross2)
    s3 = np.sign(cross3)

    if abs(cross1) < eps:
        s1 = 0
    if abs(cross2) < eps:
        s2 = 0
    if abs(cross3) < eps:
        s3 = 0

    signs = [s for s in [s1, s2, s3] if s != 0]
    if len(signs) == 0:
        return True
    return all(s == signs[0] for s in signs)


def point_in_polygon_2d(vertices, P):
    vertices = np.asarray(vertices, dtype=float)
    P = np.asarray(P, dtype=float)
    n = len(vertices)
    inside = False
    x, y = P
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if xinters > x:
                inside = not inside
    return inside


def hexagonal_grid_points(center, radius, n_layers):
    center = np.asarray(center, dtype=float)
    points = [center.copy()]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    for layer in range(1, n_layers + 1):

        for side in range(6):
            for step in range(layer):

                start = center + radius * layer * directions[side]
                end = center + radius * layer * directions[(side + 1) % 6]

                t = step / layer
                p = start + t * (end - start)
                points.append(p)
    return np.array(points, dtype=float)


def lloyd_cvt_iteration(points, domain, n_samples=1000, density_func=None):
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    xmin, xmax, ymin, ymax = domain

    samples = np.column_stack([
        np.random.uniform(xmin, xmax, n_samples),
        np.random.uniform(ymin, ymax, n_samples)
    ])
    if density_func is not None:
        weights = np.array([density_func(s[0], s[1]) for s in samples])
        weights = np.maximum(weights, 1e-10)
    else:
        weights = np.ones(n_samples)

    new_points = np.zeros_like(points)
    point_weights = np.zeros(n_points)
    for s, w in zip(samples, weights):
        dists = np.sum((points - s) ** 2, axis=1)
        idx = np.argmin(dists)
        new_points[idx] += w * s
        point_weights[idx] += w

    point_weights = np.maximum(point_weights, 1e-15)
    new_points = new_points / point_weights[:, None]
    return new_points


class CorticalSurfaceGeometry:

    def __init__(self, curvature_radius=80.0, patch_radius=5.0):
        self.R = curvature_radius
        self.patch_r = patch_radius

    def surface_height(self, x, y):
        r2 = x ** 2 + y ** 2
        if r2 >= self.R ** 2:
            return 0.0
        return self.R - np.sqrt(self.R ** 2 - r2)

    def surface_normal(self, x, y):
        r2 = x ** 2 + y ** 2
        if r2 >= self.R ** 2 - 1e-6:
            return np.array([0.0, 0.0, 1.0])
        dz_dx = x / np.sqrt(self.R ** 2 - r2)
        dz_dy = y / np.sqrt(self.R ** 2 - r2)
        n = np.array([-dz_dx, -dz_dy, 1.0])
        n = n / np.linalg.norm(n)
        return n

    def generate_electrode_positions(self, layout='hex', n_electrodes=64,
                                     n_layers=None):
        if n_layers is None:


            n_layers = max(1, int(np.ceil((-3 + np.sqrt(9 + 12 * (n_electrodes - 1))) / 6)))

            while True:
                test_points = hexagonal_grid_points(
                    center=[0.0, 0.0],
                    radius=self.patch_r / n_layers,
                    n_layers=n_layers)
                if len(test_points) >= n_electrodes:
                    break
                n_layers += 1
        if layout == 'hex':
            points_2d = hexagonal_grid_points(
                center=[0.0, 0.0],
                radius=self.patch_r / n_layers,
                n_layers=n_layers)

            points_2d = points_2d[:n_electrodes]
        elif layout == 'cvt':

            points_2d = hexagonal_grid_points(
                center=[0.0, 0.0],
                radius=self.patch_r / n_layers,
                n_layers=n_layers)
            points_2d = points_2d[:n_electrodes]
            domain = (-self.patch_r, self.patch_r, -self.patch_r, self.patch_r)
            for _ in range(20):
                points_2d = lloyd_cvt_iteration(points_2d, domain, n_samples=2000)

                dists = np.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)
                mask = dists > self.patch_r
                if np.any(mask):
                    angle = np.arctan2(points_2d[mask, 1], points_2d[mask, 0])
                    points_2d[mask, 0] = self.patch_r * np.cos(angle) * 0.95
                    points_2d[mask, 1] = self.patch_r * np.sin(angle) * 0.95
        else:
            raise ValueError(f"Unknown layout: {layout}")


        positions_3d = []
        for p in points_2d:
            x, y = p
            z = self.surface_height(x, y)
            positions_3d.append([x, y, z])
        return np.array(positions_3d, dtype=float)

    def compute_inter_electrode_distances(self, positions):
        n = len(positions)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def triangulate_electrode_patch(self, positions, max_edge_length=None):
        from scipy.spatial import Delaunay

        pts_2d = positions[:, :2]
        if len(pts_2d) < 3:
            return np.zeros((0, 3), dtype=int)
        tri = Delaunay(pts_2d)
        triangles = tri.simplices
        if max_edge_length is not None:

            filtered = []
            for tri_idx in triangles:
                p0, p1, p2 = positions[tri_idx]
                e0 = np.linalg.norm(p0 - p1)
                e1 = np.linalg.norm(p1 - p2)
                e2 = np.linalg.norm(p2 - p0)
                if max(e0, e1, e2) <= max_edge_length:
                    filtered.append(tri_idx)
            triangles = np.array(filtered, dtype=int)
        return triangles


class ElectrodeArray:

    def __init__(self, n_electrodes=64, geometry=None):
        if geometry is None:
            geometry = CorticalSurfaceGeometry()
        self.geometry = geometry
        self.n_electrodes = n_electrodes
        self.positions = None
        self.triangles = None

    def generate_layout(self, layout='cvt'):
        self.positions = self.geometry.generate_electrode_positions(
            layout=layout, n_electrodes=self.n_electrodes)
        self.triangles = self.geometry.triangulate_electrode_patch(
            self.positions, max_edge_length=2.0)
        return self.positions

    def sample_neural_field(self, field_func):
        if self.positions is None:
            self.generate_layout()
        vals = np.array([field_func(p[0], p[1], p[2]) for p in self.positions], dtype=float)
        return vals

    def compute_spatial_coverage(self):
        if self.triangles is None or len(self.triangles) == 0:
            return 0.0
        total_area = 0.0
        for tri in self.triangles:
            p0, p1, p2 = self.positions[tri]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p0 - p2)
            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
            total_area += area
        return total_area
