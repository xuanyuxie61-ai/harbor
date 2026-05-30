
import numpy as np


class DifferentialDriveRobot:

    def __init__(self, x=0.0, y=0.0, theta=0.0,
                 sigma_v=0.05, sigma_w=0.02, dt=0.1):
        self.pose = np.array([float(x), float(y), float(theta)], dtype=np.float64)
        self.sigma_v = max(float(sigma_v), 1e-12)
        self.sigma_w = max(float(sigma_w), 1e-12)
        self.dt = max(float(dt), 1e-12)

        self.covariance = np.zeros((3, 3), dtype=np.float64)

    def motion_model(self, v, w):
        x, y, theta = self.pose
        v = float(v)
        w = float(w)
        dt = self.dt

        if abs(w) < 1e-8:

            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = 0.0
        else:

            r = v / w
            dtheta = w * dt
            dx = r * (np.sin(theta + dtheta) - np.sin(theta))
            dy = -r * (np.cos(theta + dtheta) - np.cos(theta))

        new_pose = np.array([
            x + dx,
            y + dy,
            theta + dtheta
        ], dtype=np.float64)


        new_pose[2] = self._normalize_angle(new_pose[2])
        return new_pose

    def compute_jacobians(self, v, w):
        x, y, theta = self.pose
        v = float(v)
        w = float(w)
        dt = self.dt

        F = np.eye(3, dtype=np.float64)
        G = np.zeros((3, 2), dtype=np.float64)

        if abs(w) < 1e-8:

            F[0, 2] = -v * np.sin(theta) * dt
            F[1, 2] =  v * np.cos(theta) * dt
            G[0, 0] =  np.cos(theta) * dt
            G[1, 0] =  np.sin(theta) * dt
            G[2, 1] =  dt
        else:

            r = v / w
            th_new = theta + w * dt
            F[0, 2] = r * (np.cos(th_new) - np.cos(theta))
            F[1, 2] = r * (np.sin(th_new) - np.sin(theta))


            dv_x = (np.sin(th_new) - np.sin(theta)) / w
            dv_y = -(np.cos(th_new) - np.cos(theta)) / w

            dw_x = (v * dt * np.cos(th_new)) / w - (v * (np.sin(th_new) - np.sin(theta))) / (w * w)
            dw_y = (v * dt * np.sin(th_new)) / w + (v * (np.cos(th_new) - np.cos(theta))) / (w * w)

            G[0, 0] = dv_x
            G[1, 0] = dv_y
            G[2, 1] = dt
            G[0, 1] = dw_x
            G[1, 1] = dw_y

        return F, G

    def propagate(self, v, w):
        v_noisy = v + np.random.normal(0.0, self.sigma_v)
        w_noisy = w + np.random.normal(0.0, self.sigma_w)

        F, G = self.compute_jacobians(v_noisy, w_noisy)
        Q = np.diag([self.sigma_v ** 2, self.sigma_w ** 2])

        self.pose = self.motion_model(v_noisy, w_noisy)
        self.covariance = F @ self.covariance @ F.T + G @ Q @ G.T


        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        eigvals = np.linalg.eigvalsh(self.covariance)
        if np.min(eigvals) < 0:
            self.covariance += (-np.min(eigvals) + 1e-12) * np.eye(3)

        return self.pose.copy(), self.covariance.copy()

    @staticmethod
    def _normalize_angle(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def se2_to_matrix(self):

        raise NotImplementedError("Hole 2a: se2_to_matrix not implemented")

    @staticmethod
    def matrix_to_se2(T):

        raise NotImplementedError("Hole 2b: matrix_to_se2 not implemented")

    def relative_transform(self, other_pose):




        raise NotImplementedError("Hole 2c: relative_transform not implemented")
