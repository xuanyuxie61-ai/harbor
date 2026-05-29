"""
lidar_observation_model.py
二维激光雷达观测模型与点云处理

核心数学模型：
1. 激光测距模型：
   z_k = sqrt((x_{obs} - x_{robot})^2 + (y_{obs} - y_{robot})^2) + ε
   其中 ε ~ N(0, σ_r^2)
2. 角度分辨率：Δφ = 2π / N_beams
3. 坐标变换：
   [x_{world}]   [cosθ, -sinθ, x] [r*cosφ]
   [y_{world}] = [sinθ,  cosθ, y] [r*sinφ]
                                    [1    ]
4. 点云配准误差（ICP）：
   E(R,t) = Σ || (R*p_i + t) - q_{c(i)} ||^2
   其中 c(i) 为最近邻对应
"""

import numpy as np


class Lidar2D:
    """
    二维激光雷达模拟器
    """

    def __init__(self, num_beams=360, max_range=10.0, fov=2.0 * np.pi,
                 sigma_range=0.02, angular_resolution=None):
        """
        Parameters
        ----------
        num_beams : int
            激光束数量
        max_range : float
            最大测距范围 (m)
        fov : float
            视场角 (rad)
        sigma_range : float
            测距噪声标准差 (m)
        angular_resolution : float or None
            角度分辨率，若为None则自动计算
        """
        self.num_beams = max(int(num_beams), 1)
        self.max_range = max(float(max_range), 1e-6)
        self.fov = max(float(fov), 1e-6)
        self.sigma_range = max(float(sigma_range), 0.0)
        if angular_resolution is None:
            self.angular_resolution = self.fov / self.num_beams
        else:
            self.angular_resolution = max(float(angular_resolution), 1e-12)

        # 激光束角度数组
        self.angles = np.linspace(-self.fov / 2.0, self.fov / 2.0,
                                  self.num_beams, endpoint=False)

    def scan_environment(self, robot_pose, obstacles):
        """
        对环境进行激光扫描
        
        Parameters
        ----------
        robot_pose : array_like, shape (3,)
            [x, y, theta]
        obstacles : list of dict
            每个障碍物为 {'type': 'circle'|'segment', ...}
            circle: {'type':'circle', 'center':(cx,cy), 'radius':r}
            segment: {'type':'segment', 'p1':(x1,y1), 'p2':(x2,y2)}
        
        Returns
        -------
        ranges : ndarray, shape (num_beams,)
            测距值
        points : ndarray, shape (num_beams, 2)
            世界坐标系下的点云
        """
        rx, ry, rtheta = robot_pose
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float64)
        points = np.zeros((self.num_beams, 2), dtype=np.float64)

        for i, angle in enumerate(self.angles):
            beam_angle = rtheta + angle
            # 射线方向
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

            # 添加测距噪声
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
        """
        射线与圆的交点距离
        
        射线: P = O + t*d, t >= 0
        圆: ||P - C||^2 = r^2
        
        => (d·d) t^2 + 2d·(O-C) t + ||O-C||^2 - r^2 = 0
        """
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
        """
        射线与线段的交点距离
        
        射线: O + t*d
        线段: P1 + u*(P2-P1), u∈[0,1]
        
        求解: O + t*d = P1 + u*(P2-P1)
        => t*d - u*(P2-P1) = P1 - O
        """
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
        """
        将世界坐标系点云转换到机器人局部坐标系
        
        p_local = R(-θ) * (p_world - t)
        """
        x, y, theta = robot_pose
        c = np.cos(theta)
        s = np.sin(theta)
        R_inv = np.array([[c, s], [-s, c]], dtype=np.float64)
        translated = points - np.array([x, y])
        return translated @ R_inv.T

    @staticmethod
    def transform_points_to_world(points_local, robot_pose):
        """
        将局部坐标系点云转换到世界坐标系
        
        p_world = R(θ) * p_local + t
        """
        x, y, theta = robot_pose
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return points_local @ R.T + np.array([x, y])


class PointCloudRegistration:
    """
    点云配准算法（ICP 变体）
    """

    def __init__(self, max_iterations=50, tolerance=1e-6):
        self.max_iterations = max(int(max_iterations), 1)
        self.tolerance = max(float(tolerance), 1e-15)

    def icp_2d(self, source, target):
        """
        2D ICP 配准
        
        最小化目标函数：
        E(R, t) = Σ_{i=1}^N || R * s_i + t - q_{c(i)} ||^2
        
        其中 c(i) = argmin_j ||R*s_i + t - q_j||
        
        闭式解（SVD方法）：
        1. 计算质心: μ_s = mean(s), μ_q = mean(q)
        2. 去中心化: s' = s - μ_s, q' = q - μ_q
        3. 协方差: H = Σ s'_i * q'_i^T
        4. SVD: H = U * Σ * V^T
        5. R = V * U^T, t = μ_q - R * μ_s
        
        Parameters
        ----------
        source : ndarray, shape (N, 2)
        target : ndarray, shape (M, 2)
        
        Returns
        -------
        R : ndarray, shape (2, 2)
        t : ndarray, shape (2,)
        error : float
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape[0] == 0 or target.shape[0] == 0:
            return np.eye(2), np.zeros(2), np.inf

        R = np.eye(2, dtype=np.float64)
        t = np.zeros(2, dtype=np.float64)
        prev_error = np.inf

        for _ in range(self.max_iterations):
            # 变换源点云
            transformed = source @ R.T + t

            # 最近邻对应
            correspondences = self._nearest_neighbors(transformed, target)
            matched_target = target[correspondences]

            # SVD 求解最优变换
            R_new, t_new = self._svd_transform(transformed, matched_target)

            # 更新
            R = R_new @ R
            t = R_new @ t + t_new

            # 检查正交性（数值鲁棒性）
            det_R = np.linalg.det(R)
            if det_R < 0:
                # 反射修正为旋转
                U, S, Vt = np.linalg.svd(R)
                S_mat = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
                R = U @ S_mat @ Vt

            # 计算误差
            error = np.mean(np.sum((transformed - matched_target) ** 2, axis=1))
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        return R, t, prev_error

    @staticmethod
    def _nearest_neighbors(src, tgt):
        """暴力最近邻（为保证无外部依赖）"""
        indices = np.zeros(src.shape[0], dtype=np.int64)
        for i, s in enumerate(src):
            dists = np.sum((tgt - s) ** 2, axis=1)
            indices[i] = np.argmin(dists)
        return indices

    @staticmethod
    def _svd_transform(src, tgt):
        """
        SVD 求解 2D 刚体变换
        """
        mu_src = np.mean(src, axis=0)
        mu_tgt = np.mean(tgt, axis=0)
        src_centered = src - mu_src
        tgt_centered = tgt - mu_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 保证旋转矩阵（行列式为+1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src
        return R, t
