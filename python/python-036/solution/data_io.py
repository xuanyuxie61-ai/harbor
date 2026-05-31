"""
data_io.py
数据文件输入输出与索引操作

基于 file_increment 的核心算法:
    - 整数矩阵的读写
    - 数据文件列数和行数统计
    - 索引偏移 (0-based <-> 1-based 转换)
    - 网格和矩阵数据的持久化

物理应用:
    1. 保存中微子振荡概率计算结果
    2. 读写地球密度模型网格数据
    3. 实验数据索引管理
"""

import os
import numpy as np


def file_column_count(filename):
    """
    统计文件中第一行数据的列数。
    (源自 file_increment)

    参数:
        filename: 文件路径

    返回:
        column_num: 列数, -1 表示无数据
    """
    if not os.path.exists(filename):
        return -1

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            return len(parts)

    return -1


def file_row_count(filename):
    """
    统计文件中的数据行数 (跳过空行和注释行)。
    (源自 file_increment)

    参数:
        filename: 文件路径

    返回:
        row_num: 数据行数
    """
    if not os.path.exists(filename):
        return 0

    row_num = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            row_num += 1

    return row_num


def read_matrix_file(filename, dtype=np.float64):
    """
    从文本文件读取矩阵数据。

    参数:
        filename: 文件路径
        dtype:    数据类型

    返回:
        data: (n_rows, n_cols) 数组
    """
    n_cols = file_column_count(filename)
    n_rows = file_row_count(filename)

    if n_cols <= 0 or n_rows <= 0:
        return np.array([])

    data = np.zeros((n_rows, n_cols), dtype=dtype)

    with open(filename, 'r') as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == n_cols:
                data[i, :] = [dtype(x) for x in parts]
                i += 1
                if i >= n_rows:
                    break

    return data


def write_matrix_file(filename, data, fmt='%.6e', header=None):
    """
    将矩阵数据写入文本文件。

    参数:
        filename: 文件路径
        data:     数组
        fmt:      格式字符串
        header:   头部注释 (可选)
    """
    data = np.asarray(data)

    with open(filename, 'w') as f:
        if header is not None:
            f.write(f"# {header}\n")
        if data.ndim == 1:
            for val in data:
                f.write(fmt % val + "\n")
        else:
            for row in data:
                f.write(" ".join(fmt % x for x in row) + "\n")


def increment_indices(indices, increment=1):
    """
    对索引数组进行偏移操作。
    (源自 file_increment)

    参数:
        indices:   索引数组
        increment: 偏移量 (通常为 1 或 -1)

    返回:
        new_indices: 偏移后的索引
    """
    arr = np.asarray(indices, dtype=np.int64)
    return arr + increment


def convert_index_base(indices, from_base, to_base):
    """
    在 0-based 和 1-based 索引之间转换。

    参数:
        indices:  索引数组
        from_base: 0 或 1
        to_base:   0 或 1

    返回:
        converted: 转换后的索引
    """
    if from_base not in (0, 1) or to_base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    diff = to_base - from_base
    return np.asarray(indices, dtype=np.int64) + diff


def save_oscillation_results(results, filename_prefix):
    """
    保存中微子振荡计算结果到文件。

    参数:
        results:       结果字典
        filename_prefix: 文件名前缀
    """
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            fname = f"{filename_prefix}_{key}.txt"
            write_matrix_file(fname, value, header=key)
        elif isinstance(value, (int, float, complex)):
            fname = f"{filename_prefix}_{key}.txt"
            with open(fname, 'w') as f:
                f.write(f"# {key}\n")
                if isinstance(value, complex):
                    f.write(f"{value.real:.10e} {value.imag:.10e}\n")
                else:
                    f.write(f"{value:.10e}\n")


def load_density_profile(filename):
    """
    加载密度剖面数据。

    期望文件格式:
        # radius_km  density_g_cm3
        0.0          13.0
        ...

    参数:
        filename: 文件路径

    返回:
        radius:   (n,) 半径 [km]
        density:  (n,) 密度 [g/cm^3]
    """
    data = read_matrix_file(filename)
    if len(data) == 0:
        return np.array([]), np.array([])
    if data.shape[1] < 2:
        raise ValueError("Density profile file must have at least 2 columns")
    return data[:, 0], data[:, 1]
