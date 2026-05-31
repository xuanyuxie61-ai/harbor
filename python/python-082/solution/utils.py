"""
utils.py
通用工具模块：矩阵/向量打印、参数管理、数值校验、文件I/O辅助函数。
原项目映射：
  - 508_hb_to_mm 的格式解析与文件读写思想
  - 891_polygonal_surface_display 的数据表打印辅助
  - 972_r8but 的边界检查与数值鲁棒性策略
"""

import numpy as np
import sys


def r8mat_print_some(m, n, a, ilo, jlo, ihi, jhi, title):
    """打印矩阵的一部分（从MATLAB r8mat_print_some迁移）。"""
    if 0 < len(title.strip()):
        print(title)
    incx = 5
    for i2lo in range(max(ilo, 1), min(ihi, m) + 1, incx):
        i2hi = min(i2lo + incx - 1, m, ihi)
        print("  Row: ", end="")
        for i in range(i2lo, i2hi + 1):
            print(f"{i:7d}       ", end="")
        print()
        print("  Col")
        for j in range(max(jlo, 1), min(jhi, n) + 1):
            print(f"{j:5d} ", end="")
            for i in range(i2lo, i2hi + 1):
                print(f"{a[i - 1, j - 1]:12.6f}", end="")
            print()


def r8vec_print(n, a, title):
    """打印向量。"""
    if 0 < len(title.strip()):
        print(title)
    for i in range(n):
        print(f"  a[{i}] = {a[i]:14.6f}")


def validate_positive(val, name, strict=True):
    """校验正数参数，保障数值鲁棒性。"""
    if strict:
        if val <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {val}")
    else:
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative, got {val}")


def validate_matrix_nonsingular(a, tol=1e-12):
    """检查矩阵是否接近奇异。"""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")
    det = np.linalg.det(a)
    if abs(det) < tol:
        raise ValueError(f"Matrix is numerically singular (det={det:.3e}).")


def safe_inverse(a, rcond=1e-15):
    """带安全校验的矩阵求逆。"""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return vh.T @ np.diag(s_inv) @ u.T


def file_row_count(filepath):
    """统计文件中的非空非注释数据行数（从polygonal_surface_display/file_row_count迁移）。"""
    try:
        with open(filepath, 'r') as f:
            row_num = 0
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                row_num += 1
        return row_num
    except FileNotFoundError:
        return 0


def file_column_count(filepath):
    """统计文件第一行数据列数。"""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                return len(line.split())
        return 0
    except FileNotFoundError:
        return 0


def compute_condition_number(a, norm_type=2):
    """计算矩阵条件数，用于评估刚度矩阵数值稳定性。"""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return np.inf
    return np.linalg.cond(a, norm_type)


def compute_residual_norm(A, x, b, norm_type=np.inf):
    """计算线性系统残差 ||b - A x||，从 linpack_bench_backslash 迁移。"""
    r = b - A @ x
    return np.linalg.norm(r, ord=norm_type)


def compute_normalized_residual(A, x, b):
    """计算归一化残差 ratio = ||r|| / (||A|| * ||x|| * eps)。"""
    eps = np.finfo(float).eps
    r_norm = compute_residual_norm(A, x, b, np.inf)
    A_norm = np.linalg.norm(A, ord=np.inf)
    x_norm = np.linalg.norm(x, ord=np.inf)
    if A_norm == 0.0 or x_norm == 0.0:
        return np.inf
    return r_norm / (A_norm * x_norm * eps)
