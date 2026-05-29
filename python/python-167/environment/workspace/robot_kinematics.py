"""
robot_kinematics.py
多足机器人运动学与接触几何模块。
融入种子项目：
  - 1115_sphere_distance（球面距离约束，映射为关节旋转空间测地距离）

科学背景：六足机器人每条腿为 3-DOF 串联链，基坐标系 {B} 下足端位置
由正向运动学映射 FK: θ ∈ R^3 → p ∈ R^3 给出。

核心数学：
  1. Denavit-Hartenberg (DH) 参数化正向运动学
  2. 球面关节极限约束（球面距离 ≤ 关节限位角）
  3. 接触面三角形投影与摩擦锥约束
"""

import numpy as np
from typing import Tuple, List
from utils import clip_to_bounds, robust_sqrt


class SerialLegKinematics:
    """
    3-DOF 串联腿运动学模型。
    采用改进 DH 参数法（Craig 约定）：
        关节角 θ_i，连杆偏距 d_i，连杆长度 a_i，扭角 α_i
    变换矩阵：
        ^{i-1}T_i = Rot_z(θ_i)·Trans_z(d_i)·Trans_x(a_i)·Rot_x(α_i)
    """

    def __init__(self, dh_params: np.ndarray, joint_limits: np.ndarray):
        """
        dh_params: (3, 4) 每行 [θ_offset, d, a, α]
        joint_limits: (3, 2) 每行 [lower, upper] (rad)
        """
        self.dh = np.asarray(dh_params, dtype=float)
        self.limits = np.asarray(joint_limits, dtype=float)

    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        单个 DH 变换矩阵（4×4 齐次变换）。

        ^{i-1}T_i =
        [ cosθ  -sinθ·cosα   sinθ·sinα   a·cosθ ]
        [ sinθ   cosθ·cosα  -cosθ·sinα   a·sinθ ]
        [   0        sinα        cosα        d   ]
        [   0          0           0         1   ]
        """
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
        """
        正向运动学：给定关节角 q，返回足端位姿 T_ee（4×4）与各连杆变换列表。
        """
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
        """
        几何 Jacobian（3×3），仅计算线速度部分。

        数学公式：
        J_v = [ z_0 × (p_ee - p_0) , z_1 × (p_ee - p_1) , z_2 × (p_ee - p_2) ]
        其中 z_i 为第 i 关节轴在基坐标系中的方向，p_i 为关节原点位置。
        """
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
        """
        数值逆运动学：阻尼最小二乘法（Damped Least Squares）。

        迭代公式：
            Δq = J^T · (J·J^T + λ^2·I)^{-1} · Δx
        其中 Δx = target - FK(q)，λ 为阻尼系数，用于处理奇异构型附近
        的数值稳定性问题。
        """
        q = q0.copy()
        lam = 0.01
        for _ in range(max_iter):
            T_ee, _ = self.forward_kinematics(q)
            err = target - T_ee[:3, 3]
            if np.linalg.norm(err) < tol:
                break
            J = self.jacobian(q)
            # 阻尼最小二乘
            JJT = J @ J.T
            damp = lam ** 2 * np.eye(3)
            delta_q = J.T @ np.linalg.solve(JJT + damp, err)
            q = clip_to_bounds(q + delta_q, self.limits[:, 0], self.limits[:, 1])
        return q


class JointLimitConstraint:
    """
    源自 sphere_distance 的球面距离概念，映射到关节空间约束。

    科学原理：
    旋转关节的位形空间是一个三维环面 T^3 = S^1 × S^1 × S^1。
    每个关节的极限可视为在圆周 S^1 上定义了一个弧段。
    两个关节角 θ_a, θ_b 之间的"距离"采用测地距离：
        d(θ_a, θ_b) = |θ_a - θ_b|  (mod 2π，取最短弧)
    在球面 S^2 上，两点间测地距离为
        d_g = R · arccos( p_a · p_b / R^2 )
    这里我们将其类比到关节极限判定：若当前关节角距离限位边界
    的测地距离小于安全阈值，则触发软约束惩罚力。
    """

    def __init__(self, limits: np.ndarray, safety_margin: float = 0.05):
        self.limits = limits
        self.margin = safety_margin

    def geodesic_distance_to_limit(self, q: np.ndarray) -> np.ndarray:
        """
        计算每个关节到最近限位边界的测地距离。
        返回 (n,) 数组，值越小表示越接近极限。
        """
        dist_lower = np.abs(q - self.limits[:, 0])
        dist_upper = np.abs(self.limits[:, 1] - q)
        return np.minimum(dist_lower, dist_upper)

    def penalty_gradient(self, q: np.ndarray, k_p: float = 10.0) -> np.ndarray:
        """
        基于势函数的软约束惩罚梯度。

        势函数定义（类似排斥势）：
            U_i(q_i) = k_p · exp( -d_i / σ )
        其中 d_i 为到最近限位的测地距离，σ 为衰减尺度。
        梯度：∂U/∂q_i = -(k_p/σ) · exp(-d_i/σ) · sign(q_i - mid_i)
        """
        sigma = self.margin
        d = self.geodesic_distance_to_limit(q)
        mid = 0.5 * (self.limits[:, 0] + self.limits[:, 1])
        sign = np.sign(q - mid)
        grad = -(k_p / sigma) * np.exp(-d / sigma) * sign
        # 对远离限位的关节置零
        grad[d > 3 * sigma] = 0.0
        return grad


class FootContactGeometry:
    """
    足部接触几何：将足端近似为三角形接触面，计算接触力与力矩。

    科学公式（Coulomb 摩擦锥）：
    接触力 f_c 分解为法向分量 f_n 与切向分量 f_t：
        ||f_t|| ≤ μ · f_n
    其中 μ 为摩擦系数。该不等式定义了一个半角为 arctan(μ) 的圆锥。
    """

    def __init__(self, mu: float = 0.8, contact_radius: float = 0.02):
        self.mu = mu
        self.r = contact_radius

    def friction_cone_residual(self, force: np.ndarray, normal: np.ndarray) -> float:
        """
        计算给定力相对于摩擦锥的违反量。
        返回负值表示在锥内，正值表示在锥外。
        """
        fn = np.dot(force, normal)
        if fn < 0:
            return np.linalg.norm(force)  # 拉力视为极大违反
        ft = force - fn * normal
        return np.linalg.norm(ft) - self.mu * fn

    def contact_moment(self, force: np.ndarray, center_offset: np.ndarray) -> np.ndarray:
        """
        接触力对足部中心的力矩：τ = r × f。
        """
        return np.cross(center_offset, force)
