
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class CochleaGeometry:

    def __init__(self, r0=3.5, b=0.15, theta_max=4.5 * np.pi,
                 scala_height=1.2, scala_width=2.0):
        self.r0 = float(r0)
        self.b = float(b)
        self.theta_max = float(theta_max)
        self.scala_height = float(scala_height)
        self.scala_width = float(scala_width)
        self._centerline = None
        self._normals = None
        self._build_centerline()

    def _build_centerline(self, n_points=400):
        theta = np.linspace(0.0, self.theta_max, n_points)
        r = self.r0 * np.exp(-self.b * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)


        dr_dtheta = -self.b * r
        dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
        dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
        norm = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
        norm = np.where(norm < 1e-14, 1.0, norm)
        tangent = np.column_stack((dx_dtheta / norm, dy_dtheta / norm))


        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

        self._centerline = {
            'theta': theta,
            'r': r,
            'points': np.column_stack((x, y)),
            'tangent': tangent,
            'normal': normal,
        }

    def centerline_at(self, theta):
        r = self.r0 * np.exp(-self.b * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y])

    def signed_distance_to_modiolar_axis(self, points):
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be of shape (N, 2)")

        cl = self._centerline['points']

        diffs = cl[np.newaxis, :, :] - points[:, np.newaxis, :]
        dists_sq = np.sum(diffs**2, axis=2)
        closest_idx = np.argmin(dists_sq, axis=1)


        n_points = points.shape[0]
        signed_dists = np.empty(n_points)
        for i in range(n_points):
            idx = closest_idx[i]
            p = points[i]
            p1 = cl[idx]
            if idx + 1 < len(cl):
                p2 = cl[idx + 1]
            else:
                p2 = cl[idx]


            dv = p2 - p1
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 1e-14:
                dist_vec = p - p1
                signed_dists[i] = np.linalg.norm(dist_vec)
                continue


            nv = np.array([-dv[1], dv[0]]) / dv_norm

            signed_dists[i] = np.dot(nv, p - p1)

        return signed_dists, closest_idx

    def interpolate_patient_geometry(self, known_thetas, known_radii):
        known_thetas = np.asarray(known_thetas, dtype=float)
        known_radii = np.asarray(known_radii, dtype=float)
        if len(known_thetas) < 4:
            raise ValueError("至少需要 4 个数据点进行三次样条插值")
        if not np.all(np.diff(known_thetas) > 0):
            raise ValueError("known_thetas 必须严格递增")
        if np.any(known_radii <= 0):
            raise ValueError("半径必须为正")

        cs = CubicSpline(known_thetas, known_radii)
        return cs

    def build_sgn_graph(self, n_neurons=200):
        theta = np.linspace(0.0, self.theta_max, n_neurons)
        r = self.r0 * np.exp(-self.b * theta)

        offset = 0.5
        nodes = np.column_stack((
            (r + offset) * np.cos(theta),
            (r + offset) * np.sin(theta)
        ))

        edges = []
        edge_weights = []
        for i in range(n_neurons):

            for j in [i - 1, i + 1]:
                if 0 <= j < n_neurons:
                    edges.append((i, j))
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    w = 1.0 / max(dist, 1e-6)
                    edge_weights.append(w)

        return nodes, edges, np.array(edge_weights)

    def get_scala_cross_section(self, theta):
        center = self.centerline_at(theta)

        dr = -self.b * self.r0 * np.exp(-self.b * theta)
        r_val = self.r0 * np.exp(-self.b * theta)
        dx = dr * np.cos(theta) - r_val * np.sin(theta)
        dy = dr * np.sin(theta) + r_val * np.cos(theta)
        tangent = np.array([dx, dy])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
        normal = np.array([-tangent[1], tangent[0]])


        perp = np.array([-normal[1], normal[0]])
        polygon = np.array([
            center + normal * self.scala_width / 2 + perp * self.scala_height / 2,
            center + normal * self.scala_width / 2 - perp * self.scala_height / 2,
            center - normal * self.scala_width / 2 - perp * self.scala_height / 2,
            center - normal * self.scala_width / 2 + perp * self.scala_height / 2,
        ])
        return polygon
