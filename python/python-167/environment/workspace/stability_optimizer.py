
import numpy as np
from typing import Tuple, List, Optional
from utils import clip_to_bounds


class SupportPolygon:

    def __init__(self, foot_positions: np.ndarray):
        pts = np.asarray(foot_positions, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        elif pts.shape[1] >= 3:
            pts = pts[:, :2]
        self.points = pts
        self.hull = self._convex_hull_graham(pts)

    def _convex_hull_graham(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 1:
            return pts.copy()

        start_idx = np.argmin(pts[:, 1])
        p0 = pts[start_idx]

        others = np.delete(pts, start_idx, axis=0)
        angles = np.arctan2(others[:, 1] - p0[1], others[:, 0] - p0[0])
        order = np.argsort(angles)
        sorted_pts = others[order]

        hull = [p0, sorted_pts[0]]
        for i in range(1, len(sorted_pts)):
            while len(hull) > 1:
                cross = self._cross(hull[-2], hull[-1], sorted_pts[i])
                if cross <= 1e-12:
                    hull.pop()
                else:
                    break
            hull.append(sorted_pts[i])
        return np.array(hull)

    @staticmethod
    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def contains_point(self, p: np.ndarray) -> bool:
        p = np.asarray(p, dtype=float)[:2]
        hull = self.hull
        n = len(hull)
        if n == 0:
            return False
        if n == 1:
            return np.allclose(p, hull[0])
        if n == 2:

            cross = self._cross(hull[0], hull[1], p)
            if abs(cross) > 1e-9:
                return False
            dot = np.dot(p - hull[0], p - hull[1])
            return dot <= 1e-9

        sign = None
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]
            c = self._cross(a, b, p)
            if abs(c) < 1e-9:
                continue
            curr_sign = c > 0
            if sign is None:
                sign = curr_sign
            elif sign != curr_sign:
                return False
        return True

    def distance_to_boundary(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)[:2]
        hull = self.hull
        n = len(hull)
        if n < 3:
            return -np.inf
        min_dist = float('inf')
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]

            ab = b - a
            t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-14)
            t = clip_to_bounds(np.array([t]), np.array([0.0]), np.array([1.0]))[0]
            closest = a + t * ab
            dist = np.linalg.norm(p - closest)

            cross = self._cross(a, b, p)
            if cross < 0:
                dist = -dist
            min_dist = min(min_dist, dist)
        return min_dist


class StabilityMargin:

    def __init__(self, robot_mass: float = 5.0, gravity: float = 9.81):
        self.m = robot_mass
        self.g = gravity

    def static_margin(self, com: np.ndarray, support: SupportPolygon) -> float:
        return support.distance_to_boundary(com)

    def zmp_position(self, com: np.ndarray, com_acceleration: np.ndarray,
                     angular_momentum_rate: np.ndarray,
                     foot_forces: np.ndarray, foot_positions: np.ndarray) -> np.ndarray:
        com = np.asarray(com, dtype=float)
        a = np.asarray(com_acceleration, dtype=float)
        z_com = com[2] if len(com) > 2 else 0.3
        x_zmp = com[0] - (z_com * a[0]) / self.g
        y_zmp = com[1] - (z_com * a[1]) / self.g
        return np.array([x_zmp, y_zmp])


class SupportGraphCentrality:

    def __init__(self, n_legs: int = 6, alpha: float = 0.85):
        self.n = n_legs
        self.alpha = alpha

    def build_transition_matrix(self, stance_state: np.ndarray,
                                coupling_matrix: np.ndarray) -> np.ndarray:
        M = np.zeros((self.n, self.n))
        for j in range(self.n):
            col_sum = 0.0
            for i in range(self.n):
                if i == j:

                    w = 2.0 if stance_state[i] else 0.5
                else:
                    w = coupling_matrix[i, j]
                M[i, j] = w
                col_sum += w
            if col_sum > 1e-12:
                M[:, j] /= col_sum
            else:
                M[:, j] = 1.0 / self.n
        return M

    def pagerank(self, M: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
        v = np.ones(self.n) / self.n
        r = v.copy()
        for _ in range(max_iter):
            r_new = self.alpha * M @ r + (1.0 - self.alpha) * v
            if np.linalg.norm(r_new - r, 1) < tol:
                break
            r = r_new
        return r


class LinearStabilityConstraint:

    def __init__(self):
        pass

    def polygon_constraints(self, hull: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(hull)
        A = np.zeros((n, 2))
        b = np.zeros(n)
        for i in range(n):
            v_i = hull[i]
            v_j = hull[(i + 1) % n]
            edge = v_j - v_i

            normal = np.array([edge[1], -edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len
            A[i] = normal
            b[i] = np.dot(normal, v_i)
        return A, b

    def com_feasible_region(self, support: SupportPolygon,
                            margin: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        A, b = self.polygon_constraints(support.hull)
        b_shrunk = b - margin
        return A, b_shrunk
