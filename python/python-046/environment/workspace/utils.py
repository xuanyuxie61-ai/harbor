"""
utils.py
通用工具函数模块：边界检查、数值稳定性保障、向量/矩阵操作辅助函数。
"""

import numpy as np


def safe_divide(a, b, fill_value=0.0):
    """
    安全除法，避免除以零。
    当 |b| < eps 时返回 fill_value。
    """
    eps = np.finfo(float).eps * 100.0
    result = np.where(np.abs(b) < eps, fill_value, a / b)
    return result


def clip_to_range(x, x_min, x_max):
    """
    将数值裁剪到指定范围，增强数值稳定性。
    """
    return np.clip(x, x_min, x_max)


def check_finite(arr, name="array"):
    """
    检查数组中是否存在 nan 或 inf，若存在则抛出 ValueError。
    """
    if not np.all(np.isfinite(arr)):
        bad_count = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {bad_count} non-finite values.")


def normalize_vector(v):
    """
    对向量进行 L2 归一化，若范数接近零则返回零向量。
    """
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def rotation_matrix_3d(axis, angle):
    """
    3D 旋转矩阵（Rodrigues 公式）。
    axis: 旋转轴单位向量 (3,)
    angle: 旋转角度（弧度）
    """
    axis = normalize_vector(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def compute_triangle_area(p1, p2, p3):
    """
    计算三角形面积（2D）。
    p1, p2, p3: shape (2,)
    """
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1]) +
        p2[0] * (p3[1] - p1[1]) +
        p3[0] * (p1[1] - p2[1])
    )
    return area


def wrap_to_pi(angle):
    """
    将角度归化到 [-pi, pi] 区间。
    """
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def check_symmetric(A, tol=1e-8):
    """
    检查矩阵是否对称。
    """
    return np.allclose(A, A.T, atol=tol)


def ensure_positive_definite(A, min_eig=1e-10):
    """
    通过对特征值截断确保矩阵正定。
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, min_eig)
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A_pd
