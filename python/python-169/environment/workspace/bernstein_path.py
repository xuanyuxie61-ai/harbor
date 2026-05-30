
import numpy as np
from typing import List, Tuple


def bernstein_basis(n: int, a: float, b: float, t: float) -> np.ndarray:
    if b <= a:
        raise ValueError("区间右端点b必须大于左端点a")
    if t < a - 1e-12 or t > b + 1e-12:

        t = np.clip(t, a, b)

    s = (t - a) / (b - a)

    B = np.array([1.0 - s, s], dtype=float)
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return B

    for degree in range(2, n + 1):
        B_new = np.zeros(degree + 1, dtype=float)
        B_new[0] = (1.0 - s) * B[0]
        for i in range(1, degree):
            B_new[i] = (1.0 - s) * B[i] + s * B[i - 1]
        B_new[degree] = s * B[degree - 1]
        B = B_new
    return B


def de_casteljau(control_points: np.ndarray, a: float, b: float, t: float) -> np.ndarray:
    if b <= a:
        raise ValueError("区间右端点b必须大于左端点a")
    t = np.clip(t, a, b)
    s = (t - a) / (b - a)

    P = np.array(control_points, dtype=float)
    n = P.shape[0] - 1
    if n < 0:
        raise ValueError("控制点数量不能为零")

    for r in range(1, n + 1):
        for i in range(n - r + 1):
            P[i] = (1.0 - s) * P[i] + s * P[i + 1]
    return P[0]


def bezier_derivative(control_points: np.ndarray, a: float, b: float,
                      order: int = 1) -> np.ndarray:
    P = np.array(control_points, dtype=float)
    n = P.shape[0] - 1
    if n < order:
        raise ValueError(f"阶数{n}不足以求{order}阶导数")
    for _ in range(order):
        n = P.shape[0] - 1
        scale = n / (b - a)
        D = np.zeros((n, P.shape[1]), dtype=float)
        for i in range(n):
            D[i] = scale * (P[i + 1] - P[i])
        P = D
    return P


class JointSpaceBezierTrajectory:

    def __init__(self, control_points: np.ndarray, t0: float = 0.0, tf: float = 1.0):
        self.P = np.array(control_points, dtype=float)
        self.n = self.P.shape[0] - 1
        self.n_dof = self.P.shape[1]
        self.t0 = float(t0)
        self.tf = float(tf)
        if self.tf <= self.t0:
            raise ValueError("终止时间tf必须大于起始时间t0")

        self.dP = bezier_derivative(self.P, self.t0, self.tf, order=1)
        self.ddP = bezier_derivative(self.P, self.t0, self.tf, order=2)

    def position(self, t: float) -> np.ndarray:
        return de_casteljau(self.P, self.t0, self.tf, t)

    def velocity(self, t: float) -> np.ndarray:
        return de_casteljau(self.dP, self.t0, self.tf, t)

    def acceleration(self, t: float) -> np.ndarray:
        return de_casteljau(self.ddP, self.t0, self.tf, t)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.clip(t, self.t0, self.tf)
        return self.position(t), self.velocity(t), self.acceleration(t)

    def max_velocity_bound(self) -> np.ndarray:
        return np.max(np.abs(self.dP), axis=0)

    def max_acceleration_bound(self) -> np.ndarray:
        return np.max(np.abs(self.ddP), axis=0)

    def split_at(self, t_split: float) -> Tuple['JointSpaceBezierTrajectory', 'JointSpaceBezierTrajectory']:
        t_split = np.clip(t_split, self.t0, self.tf)
        s = (t_split - self.t0) / (self.tf - self.t0)
        P_left = []
        P_right = []
        P = np.array(self.P, dtype=float)
        n = self.n
        P_left.append(P[0].copy())
        P_right.append(P[n].copy())
        for r in range(1, n + 1):
            for i in range(n - r + 1):
                P[i] = (1.0 - s) * P[i] + s * P[i + 1]
            P_left.append(P[0].copy())
            P_right.append(P[n - r].copy())

        P_right.reverse()
        traj_left = JointSpaceBezierTrajectory(np.array(P_left), self.t0, t_split)
        traj_right = JointSpaceBezierTrajectory(np.array(P_right), t_split, self.tf)
        return traj_left, traj_right


def generate_minimum_jerk_bezier(q_start: np.ndarray, q_end: np.ndarray,
                                  t0: float = 0.0, tf: float = 1.0,
                                  degree: int = 5) -> JointSpaceBezierTrajectory:
    q0 = np.asarray(q_start, dtype=float).reshape(-1)
    q1 = np.asarray(q_end, dtype=float).reshape(-1)
    n_dof = q0.size
    if degree < 3:
        raise ValueError("minimum-jerk轨迹至少需要3次")

    if degree != 5:
        degree = 5


    T = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 2, 6, 12, 20],
    ], dtype=float)

    P_list = []
    for j in range(n_dof):
        rhs = np.array([q0[j], 0.0, 0.0, q1[j], 0.0, 0.0], dtype=float)
        a = np.linalg.solve(T, rhs)



        P = np.zeros(6, dtype=float)
        for i in range(6):




            for k in range(i + 1):
                if k <= 5:
                    P[i] += (np.math.comb(k, i) / np.math.comb(5, i)) * a[k] if i <= k else 0.0




        pass


    ts = np.linspace(0.0, 1.0, 6)
    V = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            V[i, j] = ts[i] ** j

    B_mat = np.zeros((6, 6), dtype=float)
    for idx, t in enumerate(ts):
        B_mat[idx, :] = bernstein_basis(5, 0.0, 1.0, t)


    try:
        B_inv = np.linalg.inv(B_mat)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_mat)
    P_all = np.zeros((6, n_dof), dtype=float)
    for j in range(n_dof):
        rhs = np.array([q0[j], 0.0, 0.0, q1[j], 0.0, 0.0], dtype=float)
        a = np.linalg.solve(T, rhs)
        y_samples = V @ a
        P_all[:, j] = B_inv @ y_samples
    return JointSpaceBezierTrajectory(P_all, t0, tf)


def clamp_control_points_to_joint_limits(P: np.ndarray,
                                         q_min: np.ndarray,
                                         q_max: np.ndarray) -> np.ndarray:
    q_min = np.asarray(q_min).reshape(1, -1)
    q_max = np.asarray(q_max).reshape(1, -1)
    return np.clip(P, q_min, q_max)
