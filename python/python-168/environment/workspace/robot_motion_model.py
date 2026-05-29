"""
robot_motion_model.py
移动机器人差分驱动运动模型与误差传播

核心数学模型：
1. 状态向量: x = [x, y, theta]^T ∈ SE(2)
2. 运动学方程（差分驱动）：
   x_{t+1} = x_t + v_t * cos(theta_t) * Δt
   y_{t+1} = y_t + v_t * sin(theta_t) * Δt
   theta_{t+1} = theta_t + ω_t * Δt
3. 误差传播（一阶泰勒展开）：
   Σ_{t+1} = F_t * Σ_t * F_t^T + G_t * Q * G_t^T
   其中 F_t = ∂f/∂x 为状态雅可比，G_t = ∂f/∂u 为控制雅可比
4. 协方差矩阵 Q = diag(σ_v^2, σ_ω^2) 描述控制噪声
"""

import numpy as np


class DifferentialDriveRobot:
    """
    差分驱动移动机器人模型
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0,
                 sigma_v=0.05, sigma_w=0.02, dt=0.1):
        """
        初始化机器人状态
        
        Parameters
        ----------
        x, y, theta : float
            初始位姿 (x, y, 朝向角)
        sigma_v : float
            线速度噪声标准差 (m/s)
        sigma_w : float
            角速度噪声标准差 (rad/s)
        dt : float
            时间步长 (s)
        """
        self.pose = np.array([float(x), float(y), float(theta)], dtype=np.float64)
        self.sigma_v = max(float(sigma_v), 1e-12)
        self.sigma_w = max(float(sigma_w), 1e-12)
        self.dt = max(float(dt), 1e-12)
        # 初始协方差 3x3
        self.covariance = np.zeros((3, 3), dtype=np.float64)

    def motion_model(self, v, w):
        """
        确定性运动模型（无噪声）
        
        x_{t+1} = x_t + v * cos(theta) * dt       (若 |w| < epsilon)
                = x_t + (v/w) * (sin(theta+w*dt) - sin(theta))  (否则)
        y_{t+1} = y_t + v * sin(theta) * dt       (若 |w| < epsilon)
                = y_t - (v/w) * (cos(theta+w*dt) - cos(theta))  (否则)
        theta_{t+1} = theta + w * dt
        
        使用精确圆弧运动模型避免直线近似误差
        """
        x, y, theta = self.pose
        v = float(v)
        w = float(w)
        dt = self.dt

        if abs(w) < 1e-8:
            # 直线运动极限
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = 0.0
        else:
            # 圆弧运动
            r = v / w
            dtheta = w * dt
            dx = r * (np.sin(theta + dtheta) - np.sin(theta))
            dy = -r * (np.cos(theta + dtheta) - np.cos(theta))

        new_pose = np.array([
            x + dx,
            y + dy,
            theta + dtheta
        ], dtype=np.float64)

        # 角度归一化到 [-pi, pi]
        new_pose[2] = self._normalize_angle(new_pose[2])
        return new_pose

    def compute_jacobians(self, v, w):
        """
        计算运动模型雅可比矩阵
        
        F = ∂f/∂x = [[1, 0, -v*sin(theta)*dt],
                     [0, 1,  v*cos(theta)*dt],
                     [0, 0,  1]]        (直线近似)
        
        更精确的圆弧雅可比：
        F = [[1, 0, (v/w)*(cos(theta+w*dt)-cos(theta))],
             [0, 1, (v/w)*(sin(theta+w*dt)-sin(theta))],
             [0, 0, 1]]
        
        G = ∂f/∂u = 根据控制输入计算
        """
        x, y, theta = self.pose
        v = float(v)
        w = float(w)
        dt = self.dt

        F = np.eye(3, dtype=np.float64)
        G = np.zeros((3, 2), dtype=np.float64)

        if abs(w) < 1e-8:
            # 直线雅可比
            F[0, 2] = -v * np.sin(theta) * dt
            F[1, 2] =  v * np.cos(theta) * dt
            G[0, 0] =  np.cos(theta) * dt
            G[1, 0] =  np.sin(theta) * dt
            G[2, 1] =  dt
        else:
            # 圆弧雅可比
            r = v / w
            th_new = theta + w * dt
            F[0, 2] = r * (np.cos(th_new) - np.cos(theta))
            F[1, 2] = r * (np.sin(th_new) - np.sin(theta))

            # 对 v 的偏导
            dv_x = (np.sin(th_new) - np.sin(theta)) / w
            dv_y = -(np.cos(th_new) - np.cos(theta)) / w
            # 对 w 的偏导 (更复杂，含 r 对 w 的依赖)
            dw_x = (v * dt * np.cos(th_new)) / w - (v * (np.sin(th_new) - np.sin(theta))) / (w * w)
            dw_y = (v * dt * np.sin(th_new)) / w + (v * (np.cos(th_new) - np.cos(theta))) / (w * w)

            G[0, 0] = dv_x
            G[1, 0] = dv_y
            G[2, 1] = dt
            G[0, 1] = dw_x
            G[1, 1] = dw_y

        return F, G

    def propagate(self, v, w):
        """
        执行一步运动并传播协方差
        
        Σ_{t+1} = F_t * Σ_t * F_t^T + G_t * Q * G_t^T
        """
        v_noisy = v + np.random.normal(0.0, self.sigma_v)
        w_noisy = w + np.random.normal(0.0, self.sigma_w)

        F, G = self.compute_jacobians(v_noisy, w_noisy)
        Q = np.diag([self.sigma_v ** 2, self.sigma_w ** 2])

        self.pose = self.motion_model(v_noisy, w_noisy)
        self.covariance = F @ self.covariance @ F.T + G @ Q @ G.T

        # 数值稳定性：对称化
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        # 保证半正定性
        eigvals = np.linalg.eigvalsh(self.covariance)
        if np.min(eigvals) < 0:
            self.covariance += (-np.min(eigvals) + 1e-12) * np.eye(3)

        return self.pose.copy(), self.covariance.copy()

    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def se2_to_matrix(self):
        """
        将位姿转换为 SE(2) 齐次变换矩阵
        
        T = [[cosθ, -sinθ, x],
             [sinθ,  cosθ, y],
             [0,     0,    1]]
        """
        # HOLE 2a: 实现位姿到 SE(2) 齐次变换矩阵的转换
        raise NotImplementedError("Hole 2a: se2_to_matrix not implemented")

    @staticmethod
    def matrix_to_se2(T):
        """SE(2) 矩阵转换为位姿向量"""
        # HOLE 2b: 实现 SE(2) 矩阵到 [x, y, theta] 位姿向量的转换
        raise NotImplementedError("Hole 2b: matrix_to_se2 not implemented")

    def relative_transform(self, other_pose):
        """
        计算从当前位姿到其他位姿的相对变换
        
        ξ_ij = T_i^{-1} ⊕ T_j
        """
        # HOLE 2c: 实现 SE(2) 相对变换计算
        # 要求：使用 se2_to_matrix 和 matrix_to_se2，计算 T_i^{-1} @ T_j
        # 注意：此处的输出格式必须与 graph_slam_optimizer.py 中
        # _compute_error_and_jacobians 对观测值 z 的期望一致。
        raise NotImplementedError("Hole 2c: relative_transform not implemented")
