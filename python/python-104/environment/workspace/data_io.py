"""
data_io.py — 自适应光学波前数据文件I/O库

融合原项目: 1420_xy_io (2D几何数据文件读写) + 718_matlab_commandline (文件操作)

功能:
  - 波前相位数据 (XY格式) 的读写
  - Zernike系数向量 (XYF格式关联) 的读写
  - 子孔径斜率数据 (XYL格式关联) 的读写
  - 系统参数日志记录
"""

import os
import numpy as np


def write_xy_data(filepath, x, y, header_comment="Wavefront phase data"):
    """
    写入二维坐标数据 (XY格式)
    格式: 头部注释 + 每行 "x y"
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Cannot write empty data.")
    with open(filepath, 'w') as f:
        f.write(f"# {header_comment}\n")
        f.write(f"# N_POINTS: {len(x)}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.12e} {yi:.12e}\n")


def read_xy_data(filepath):
    """
    读取二维坐标数据 (XY格式)
    返回: x数组, y数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    x_list, y_list = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            x_list.append(float(parts[0]))
            y_list.append(float(parts[1]))
    if len(x_list) == 0:
        raise ValueError("No valid data found in file.")
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def write_zernike_coefficients(filepath, coeffs, modes_labels=None):
    """
    写入Zernike系数向量 (类XYF格式: 系数 + 模式标签)
    coeffs: 1D numpy array
    """
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D array.")
    n = len(coeffs)
    with open(filepath, 'w') as f:
        f.write("# Zernike mode coefficients\n")
        f.write(f"# N_MODES: {n}\n")
        for i in range(n):
            label = modes_labels[i] if modes_labels and i < len(modes_labels) else f"Z{i}"
            f.write(f"{i:4d} {coeffs[i]:.16e} {label}\n")


def read_zernike_coefficients(filepath):
    """
    读取Zernike系数向量
    返回: coeffs数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    coeffs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            coeffs.append(float(parts[1]))
    if len(coeffs) == 0:
        raise ValueError("No valid coefficients found.")
    return np.array(coeffs, dtype=np.float64)


def write_subaperture_slopes(filepath, sx, sy, subap_indices=None):
    """
    写入子孔径斜率数据 (类XYL格式: 斜率 + 子孔径索引)
    sx, sy: 1D arrays of x- and y-slopes
    """
    if len(sx) != len(sy):
        raise ValueError("sx and sy must have the same length.")
    if len(sx) == 0:
        raise ValueError("Cannot write empty slope data.")
    n = len(sx)
    with open(filepath, 'w') as f:
        f.write("# Shack-Hartmann subaperture slopes\n")
        f.write(f"# N_SUBAPERTURES: {n}\n")
        for i in range(n):
            idx = subap_indices[i] if subap_indices is not None and i < len(subap_indices) else i
            f.write(f"{idx:4d} {sx[i]:.12e} {sy[i]:.12e}\n")


def read_subaperture_slopes(filepath):
    """
    读取子孔径斜率数据
    返回: indices, sx, sy
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    indices, sx, sy = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            indices.append(int(parts[0]))
            sx.append(float(parts[1]))
            sy.append(float(parts[2]))
    if len(sx) == 0:
        raise ValueError("No valid slope data found.")
    return np.array(indices, dtype=int), np.array(sx, dtype=np.float64), np.array(sy, dtype=np.float64)


def log_system_parameters(filepath, params_dict):
    """
    记录系统参数到日志文件 (源自718_matlab_commandline的文件操作思想)
    """
    with open(filepath, 'w') as f:
        f.write("# Adaptive Optics System Parameters Log\n")
        f.write("# ====================================\n")
        for key, value in params_dict.items():
            if isinstance(value, float):
                f.write(f"{key:<30s} = {value:.12e}\n")
            elif isinstance(value, int):
                f.write(f"{key:<30s} = {value:d}\n")
            else:
                f.write(f"{key:<30s} = {value}\n")


def read_system_parameters(filepath):
    """
    读取系统参数字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val_str = line.split('=', 1)
                key = key.strip()
                val_str = val_str.strip()
                try:
                    if '.' in val_str or 'e' in val_str.lower():
                        params[key] = float(val_str)
                    else:
                        params[key] = int(val_str)
                except ValueError:
                    params[key] = val_str
    return params
