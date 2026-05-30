
import numpy as np


class Lidar2D:

    def __init__(self, num_beams=360, max_range=10.0, fov=2.0 * np.pi,
                 sigma_range=0.02, angular_resolution=None):
        self.num_beams = max(int(num_beams), 1)
        self.max_range = max(float(max_range), 1e-6)
        self.fov = max(float(fov), 1e-6)
        self.sigma_range = max(float(sigma_range), 0.0)
        if angular_resolution is None:
            self.angular_resolution = self.fov / self.num_beams
        else:
            self.angular_resolution = max(float(angular_resolution), 1e-12)


        self.angles = np.linspace(-self.fov / 2.0, self.fov / 2.0,
                                  self.num_beams, endpoint=False)

    def scan_environment(self, robot_pose, obstacles):
        rx, ry, rtheta = robot_pose
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float64)
        points = np.zeros((self.num_beams, 2), dtype=np.float64)

        for i, angle in enumerate(self.angles):
            beam_angle = rtheta + angle

            dx = np.cos(beam_angle)
            dy = np.sin(beam_angle)

            min_dist = self.max_range
            for obs in obstacles:
                if obs.get('type') == 'circle':
                    dist = self._ray_circle_intersection(
                        rx, ry, dx, dy,
                        obs['center'][0], obs['center'][1], obs['radius']
                    )
                elif obs.get('type') == 'segment':
                    dist = self._ray_segment_intersection(
                        rx, ry, dx, dy,
                        obs['p1'][0], obs['p1'][1],
                        obs['p2'][0], obs['p2'][1]
                    )
                else:
                    continue
                if dist is not None and 1e-6 < dist < min_dist:
                    min_dist = dist


            if self.sigma_range > 0:
                noise = np.random.normal(0.0, self.sigma_range)
            else:
                noise = 0.0
            measured = min_dist + noise
            measured = max(1e-6, min(measured, self.max_range))
            ranges[i] = measured
            points[i, 0] = rx + measured * dx
            points[i, 1] = ry + measured * dy

        return ranges, points

    @staticmethod
    def _ray_circle_intersection(ox, oy, dx, dy, cx, cy, r):
        fx = ox - cx
        fy = oy - cy
        a = dx * dx + dy * dy
        b = 2.0 * (dx * fx + dy * fy)
        c = fx * fx + fy * fy - r * r
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None
        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        t_valid = [t for t in (t1, t2) if t > 1e-8]
        return min(t_valid) if t_valid else None

    @staticmethod
    def _ray_segment_intersection(ox, oy, dx, dy, x1, y1, x2, y2):
        rx = x2 - x1
        ry = y2 - y1

        denom = dx * (-ry) - dy * (-rx)
        if abs(denom) < 1e-12:
            return None

        t = ((x1 - ox) * (-ry) - (y1 - oy) * (-rx)) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / (-denom)

        if t > 1e-8 and 0.0 <= u <= 1.0:
            return t
        return None

    @staticmethod
    def transform_points_to_local(points, robot_pose):
        x, y, theta = robot_pose
        c = np.cos(theta)
        s = np.sin(theta)
        R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)
        translated = points - np.array([x, y])
        return translated @ R_inv.T

    @staticmethod
    def transform_points_to_world(points_local, robot_pose):
        x, y, theta = robot_pose
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return points_local @ R.T + np.array([x, y])


class PointCloudRegistration:

    def __init__(self, max_iterations=50, tolerance=1e-6):
        self.max_iterations = max(int(max_iterations), 1)
        self.tolerance = max(float(tolerance), 1e-15)

    def icp_2d(self, source, target):
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape[0] == 0 or target.shape[0] == 0:
            return np.eye(2), np.zeros(2), np.inf

        R = np.eye(2, dtype=np.float64)
        t = np.zeros(2, dtype=np.float64)
        prev_error = np.inf

        for _ in range(self.max_iterations):

            transformed = source @ R.T + t


            correspondences = self._nearest_neighbors(transformed, target)
            matched_target = target[correspondences]


            R_new, t_new = self._svd_transform(transformed, matched_target)


            R = R_new @ R
            t = R_new @ t + t_new


            det_R = np.linalg.det(R)
            if det_R < 0:

                U, S, Vt = np.linalg.svd(R)
                S_mat = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
                R = U @ S_mat @ Vt


            error = np.mean(np.sum((transformed - matched_target) ** 2, axis=1))
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        return R, t, prev_error

    @staticmethod
    def _nearest_neighbors(src, tgt):
        indices = np.zeros(src.shape[0], dtype=np.int64)
        for i, s in enumerate(src):
            dists = np.sum((tgt - s) ** 2, axis=1)
            indices[i] = np.argmin(dists)
        return indices

    @staticmethod
    def _svd_transform(src, tgt):
        mu_src = np.mean(src, axis=0)
        mu_tgt = np.mean(tgt, axis=0)
        src_centered = src - mu_src
        tgt_centered = tgt - mu_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T


        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src
        return R, t
