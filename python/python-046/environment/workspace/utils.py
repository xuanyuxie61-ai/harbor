
import numpy as np


def safe_divide(a, b, fill_value=0.0):
    eps = np.finfo(float).eps * 100.0
    result = np.where(np.abs(b) < eps, fill_value, a / b)
    return result


def clip_to_range(x, x_min, x_max):
    return np.clip(x, x_min, x_max)


def check_finite(arr, name="array"):
    if not np.all(np.isfinite(arr)):
        bad_count = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {bad_count} non-finite values.")


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def rotation_matrix_3d(axis, angle):
    axis = normalize_vector(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def compute_triangle_area(p1, p2, p3):
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1]) +
        p2[0] * (p3[1] - p1[1]) +
        p3[0] * (p1[1] - p2[1])
    )
    return area


def wrap_to_pi(angle):
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def check_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)


def ensure_positive_definite(A, min_eig=1e-10):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, min_eig)
    A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A_pd
