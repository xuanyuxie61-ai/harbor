"""
utils.py
工具函数集合：数值稳定性、文件 I/O、统计计算
基于 gmsh_to_fem 的网格 I/O 与 gpl_display 的数据解析思想重构
"""

import numpy as np


def safe_divide(a, b, fill_value=0.0):
    """安全除法，避免除以零。"""
    b = np.asarray(b, dtype=float)
    result = np.empty_like(np.asarray(a, dtype=float))
    mask = np.abs(b) > 1e-14
    result[mask] = np.asarray(a, dtype=float)[mask] / b[mask]
    result[~mask] = fill_value
    return result


def normalize_vector(v):
    """向量归一化，处理零向量。"""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.zeros_like(v)
    return v / norm


def compute_mean_free_path(volume, surface_area):
    """
    平均自由程（Mean Free Path）：
        Λ = 4V / S
    其中 V 为体积，S 为总表面积。
    """
    if surface_area < 1e-14:
        return 0.0
    return 4.0 * volume / surface_area


def sabine_absorption_to_t60(volume, total_absorption):
    """
    Sabine 公式：T60 = 0.161 * V / A
    """
    if total_absorption < 1e-14:
        total_absorption = 1e-14
    return 0.161 * volume / total_absorption


def eyring_absorption_to_t60(volume, surface_area, avg_absorption):
    """
    Eyring 公式（更精确的混响时间）：
        T60 = 0.161 * V / (-S * ln(1 - α_avg))
    """
    if avg_absorption >= 1.0:
        avg_absorption = 0.999
    if avg_absorption < 1e-14:
        return sabine_absorption_to_t60(volume, surface_area * avg_absorption)
    denom = -surface_area * np.log(1.0 - avg_absorption)
    if abs(denom) < 1e-14:
        denom = 1e-14
    return 0.161 * volume / denom


def write_matrix_file(filename, matrix, fmt='%.6f'):
    """
    将矩阵写入文本文件（基于 gmsh_to_fem 的文件输出格式）。
    """
    np.savetxt(filename, matrix, fmt=fmt)


def read_node_element_files(node_file, element_file):
    """
    读取节点和单元文件（基于 gmsh_to_fem 的网格读取思想）。
    """
    nodes = np.loadtxt(node_file, dtype=float)
    elements = np.loadtxt(element_file, dtype=int)
    return nodes, elements


def compute_bounding_box(points):
    """计算点集的包围盒。"""
    return np.min(points, axis=0), np.max(points, axis=0)


def is_point_inside_box(point, box_min, box_max, tol=1e-10):
    """判断点是否在轴对齐包围盒内。"""
    point = np.asarray(point)
    return np.all(point >= box_min - tol) and np.all(point <= box_max + tol)


def linear_regression(x, y):
    """
    一元线性回归：y = a*x + b
    使用最小二乘法。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if abs(ss_xx) < 1e-14:
        return 0.0, y_mean
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def db_to_linear(db):
    """dB 转线性幅值。"""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear):
    """线性幅值转 dB。"""
    linear = np.maximum(np.asarray(linear, dtype=float), 1e-15)
    return 20.0 * np.log10(linear)


def energy_to_db(energy):
    """能量转 dB。"""
    energy = np.maximum(np.asarray(energy, dtype=float), 1e-15)
    return 10.0 * np.log10(energy)


def check_finite_and_real(arr, name="array"):
    """检查数组是否全为有限实数。"""
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        bad_count = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {bad_count} non-finite values")
    return True


def compute_statistics(data):
    """计算基本统计量。"""
    data = np.asarray(data, dtype=float)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
    }
