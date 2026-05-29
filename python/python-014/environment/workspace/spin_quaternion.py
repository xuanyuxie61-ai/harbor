"""
spin_quaternion.py
==================
自旋矢量的四元数表示与三维旋转模块。
融合来源：quaternions（四元数旋转矩阵）。

在凝聚态物理中，经典 Heisenberg 自旋 S 是三维单位矢量。
四元数表示提供：
- 无奇异点的球面插值（slerp）
- 数值稳定的旋转合成
- 避免欧拉角的万向节锁问题

核心公式：
    单位四元数 q = [cos(θ/2), n̂ sin(θ/2)]，其中 n̂ 为旋转轴。
    矢量 v 的旋转：v' = q ⊗ [0, v] ⊗ q*。
    对应的旋转矩阵 R(q) 为 3×3 正交矩阵，det(R)=1。
"""

import numpy as np
from typing import Tuple, Optional
from utils import EPS_MACHINE, safe_sqrt


def q_normalize(q: np.ndarray) -> np.ndarray:
    """归一化四元数。"""
    norm = np.linalg.norm(q)
    if norm < EPS_MACHINE:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def q_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭：q* = [q0, -q1, -q2, -q3]。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    四元数乘法（Hamilton 积）。
    对于 p = [p0, p_vec], q = [q0, q_vec]：
        p ⊗ q = [p0 q0 - p_vec·q_vec, p0 q_vec + q0 p_vec + p_vec × q_vec]
    """
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
    return np.array([r0, r1, r2, r3])


def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    使用单位四元数 q 旋转三维矢量 v。
    v' = q ⊗ [0, v] ⊗ q*。
    """
    # TODO: Hole_1 — 实现四元数旋转向量公式
    # 提示：先归一化 q，将 v 嵌入为纯虚四元数 [0, v]，然后计算 q ⊗ [0,v] ⊗ q*
    raise NotImplementedError("Hole_1: 请实现 q_rotate_vector 函数")


def q_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转 3×3 旋转矩阵。
    融合来源：rotation_quat2mat。

    对于单位四元数 q = [q0, q1, q2, q3]：
        R = [[1-2(q2^2+q3^2), 2(q1 q2 - q0 q3), 2(q1 q3 + q0 q2)],
             [2(q1 q2 + q0 q3), 1-2(q1^2+q3^2), 2(q2 q3 - q0 q1)],
             [2(q1 q3 - q0 q2), 2(q2 q3 + q0 q1), 1-2(q1^2+q2^2)]]
    """
    q = q_normalize(q)
    q0, q1, q2, q3 = q
    R = np.zeros((3, 3), dtype=float)
    R[0, 0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    R[0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    R[0, 2] = 2.0 * (q1 * q3 + q0 * q2)
    R[1, 0] = 2.0 * (q1 * q2 + q0 * q3)
    R[1, 1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    R[1, 2] = 2.0 * (q2 * q3 - q0 * q1)
    R[2, 0] = 2.0 * (q1 * q3 - q0 * q2)
    R[2, 1] = 2.0 * (q2 * q3 + q0 * q1)
    R[2, 2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    return R


def rotation_matrix_to_q(R: np.ndarray) -> np.ndarray:
    """
    3×3 旋转矩阵转四元数（Shepperd 方法）。
    数值鲁棒：选择最大迹的分支以避免 cancellation。
    """
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / safe_sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2, 1] - R[1, 2]) * s
        q2 = (R[0, 2] - R[2, 0]) * s
        q3 = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * safe_sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * safe_sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * safe_sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s
    return q_normalize(np.array([q0, q1, q2, q3]))


def axis_angle_to_q(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转轴 + 旋转角 -> 四元数。
    angle 单位为弧度。
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < EPS_MACHINE:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = 0.5 * angle
    s = np.sin(half)
    c = np.cos(half)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])


def q_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    球面线性插值 (Spherical Linear Interpolation)。
    用于自旋构型在能量景观中的平滑路径参数化。
    """
    q1 = q_normalize(q1)
    q2 = q_normalize(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    # 若夹角过大，取反以避免长路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 1.0 - EPS_MACHINE:
        # 线性插值退避
        res = q1 + t * (q2 - q1)
        return q_normalize(res)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return q_normalize(q1 * s1 + q2 * s2)


def random_spin_quaternion(seed: Optional[int] = None) -> np.ndarray:
    """
    生成均匀随机分布的三维单位矢量对应的四元数。
    采用 Marsaglia 方法：在单位球面上均匀采样。
    """
    if seed is not None:
        np.random.seed(seed)
    while True:
        x1, x2 = np.random.uniform(-1.0, 1.0, size=2)
        s1 = x1 * x1 + x2 * x2
        if s1 < 1.0:
            break
    while True:
        x3, x4 = np.random.uniform(-1.0, 1.0, size=2)
        s2 = x3 * x3 + x4 * x4
        if s2 < 1.0 and s2 > EPS_MACHINE:
            break
    # 四元数表示随机旋转，等价于随机自旋方向
    q = np.array([
        x1,
        x2,
        x3 * np.sqrt((1.0 - s1) / s2),
        x4 * np.sqrt((1.0 - s1) / s2),
    ], dtype=float)
    return q_normalize(q)


def spin_vector_to_q(s: np.ndarray) -> np.ndarray:
    """
    将单位自旋矢量 s 映射到四元数。
    约定：s 沿 z 轴时 q = [1,0,0,0]；绕轴 n̂ 旋转 θ 使 s 对齐。
    此处简单映射：取旋转轴 n̂ = (ẑ × s) / |ẑ × s|，θ = arccos(ẑ·s)。
    """
    s = np.array(s, dtype=float)
    norm = np.linalg.norm(s)
    if norm < EPS_MACHINE:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = s / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(z, s), -1.0, 1.0)
    angle = np.arccos(dot)
    axis = np.cross(z, s)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < EPS_MACHINE:
        # 平行或反平行
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / ax_norm
    return axis_angle_to_q(axis, angle)


def q_to_spin_vector(q: np.ndarray) -> np.ndarray:
    """将四元数对应的旋转作用于 z 轴单位矢量，得到自旋方向。"""
    return q_rotate_vector(q, np.array([0.0, 0.0, 1.0]))



