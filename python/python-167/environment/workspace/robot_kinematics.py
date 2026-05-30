
import numpy as np
from typing import Tuple, List
from utils import clip_to_bounds, robust_sqrt


class SerialLegKinematics:

    def __init__(self, dh_params: np.ndarray, joint_limits: np.ndarray):
        self.dh = np.asarray(dh_params, dtype=float)
        self.limits = np.asarray(joint_limits, dtype=float)

    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,      ca,      d  ],
            [0.0,    0.0,     0.0,     1.0 ]
        ], dtype=float)
        return T

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = clip_to_bounds(q, self.limits[:, 0], self.limits[:, 1])
        T = np.eye(4)
        transforms = [T.copy()]
        for i in range(3):
            theta = self.dh[i, 0] + q[i]
            d = self.dh[i, 1]
            a = self.dh[i, 2]
            alpha = self.dh[i, 3]
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i
            transforms.append(T.copy())
        return T, transforms

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        T_ee, transforms = self.forward_kinematics(q)
        p_ee = T_ee[:3, 3]
        J = np.zeros((3, 3))
        for i in range(3):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]
            J[:, i] = np.cross(z_i, p_ee - p_i)
        return J

    def inverse_kinematics_numerical(self, target: np.ndarray, q0: np.ndarray,
                                     max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        q = q0.copy()
        lam = 0.01
        for _ in range(max_iter):
            T_ee, _ = self.forward_kinematics(q)
            err = target - T_ee[:3, 3]
            if np.linalg.norm(err) < tol:
                break
            J = self.jacobian(q)

            JJT = J @ J.T
            damp = lam ** 2 * np.eye(3)
            delta_q = J.T @ np.linalg.solve(JJT + damp, err)
            q = clip_to_bounds(q + delta_q, self.limits[:, 0], self.limits[:, 1])
        return q


class JointLimitConstraint:

    def __init__(self, limits: np.ndarray, safety_margin: float = 0.05):
        self.limits = limits
        self.margin = safety_margin

    def geodesic_distance_to_limit(self, q: np.ndarray) -> np.ndarray:
        dist_lower = np.abs(q - self.limits[:, 0])
        dist_upper = np.abs(self.limits[:, 1] - q)
        return np.minimum(dist_lower, dist_upper)

    def penalty_gradient(self, q: np.ndarray, k_p: float = 10.0) -> np.ndarray:
        sigma = self.margin
        d = self.geodesic_distance_to_limit(q)
        mid = 0.5 * (self.limits[:, 0] + self.limits[:, 1])
        sign = np.sign(q - mid)
        grad = -(k_p / sigma) * np.exp(-d / sigma) * sign

        grad[d > 3 * sigma] = 0.0
        return grad


class FootContactGeometry:

    def __init__(self, mu: float = 0.8, contact_radius: float = 0.02):
        self.mu = mu
        self.r = contact_radius

    def friction_cone_residual(self, force: np.ndarray, normal: np.ndarray) -> float:
        fn = np.dot(force, normal)
        if fn < 0:
            return np.linalg.norm(force)
        ft = force - fn * normal
        return np.linalg.norm(ft) - self.mu * fn

    def contact_moment(self, force: np.ndarray, center_offset: np.ndarray) -> np.ndarray:
        return np.cross(center_offset, force)
