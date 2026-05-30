
import numpy as np
from typing import Tuple, Optional
from utils import EPS_MACHINE, safe_sqrt


def q_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < EPS_MACHINE:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def q_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
    return np.array([r0, r1, r2, r3])


def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:


    raise NotImplementedError("Hole_1: 请实现 q_rotate_vector 函数")


def q_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
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
    q1 = q_normalize(q1)
    q2 = q_normalize(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)

    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 1.0 - EPS_MACHINE:

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

    q = np.array([
        x1,
        x2,
        x3 * np.sqrt((1.0 - s1) / s2),
        x4 * np.sqrt((1.0 - s1) / s2),
    ], dtype=float)
    return q_normalize(q)


def spin_vector_to_q(s: np.ndarray) -> np.ndarray:
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

        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / ax_norm
    return axis_angle_to_q(axis, angle)


def q_to_spin_vector(q: np.ndarray) -> np.ndarray:
    return q_rotate_vector(q, np.array([0.0, 0.0, 1.0]))



