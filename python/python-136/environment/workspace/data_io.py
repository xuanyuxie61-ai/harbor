"""
data_io.py
==========
催化剂孔扩散与表面反应模拟系统的数据读写模块。

基于种子项目 1420_xy_io 的核心思想重构：
- 原项目负责 XY 坐标点集的文本读写；
- 在本系统中承担催化剂孔结构节点数据、浓度-温度剖面数据、
  以及二维截面采样点数据的持久化与加载职责。

所有读写操作均包含边界校验与异常处理，确保工程鲁棒性。
"""

import os
import numpy as np


class DataIOException(Exception):
    """数据 I/O 异常基类。"""
    pass


def read_xy_profile(filename):
    """
    从文本文件中读取一维剖面数据（如径向坐标与浓度/温度）。

    文件格式：每行两个浮点数，以空格/制表符分隔，
    第一列为位置坐标 x，第二列为物理量 y。
    以 '#' 开头的行为注释，空行自动跳过。

    Parameters
    ----------
    filename : str
        输入文件路径。

    Returns
    -------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)

    Raises
    ------
    DataIOException
        文件不存在、格式错误或数据为空时抛出。
    """
    if not os.path.isfile(filename):
        raise DataIOException(f"文件不存在: {filename}")

    x_list, y_list = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise DataIOException(
                    f"{filename}:{line_num} 格式错误，需要至少两列数据"
                )
            try:
                xv = float(parts[0])
                yv = float(parts[1])
            except ValueError as exc:
                raise DataIOException(
                    f"{filename}:{line_num} 无法解析为浮点数"
                ) from exc
            x_list.append(xv)
            y_list.append(yv)

    if not x_list:
        raise DataIOException(f"{filename} 中未找到有效数据")

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)

    if not np.all(np.diff(x) >= 0):
        # 允许非严格递增，但给出警告性质的排序
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    return x, y


def write_xy_profile(filename, x, y, header=None):
    """
    将一维剖面数据写入文本文件。

    Parameters
    ----------
    filename : str
        输出文件路径。
    x, y : ndarray
        坐标与物理量数组，长度必须一致。
    header : str, optional
        文件头部注释（自动添加 '#' 前缀）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise DataIOException("x 与 y 的形状不一致")
    if x.size == 0:
        raise DataIOException("空数组不可写入")

    with open(filename, 'w', encoding='utf-8') as f:
        if header is not None:
            for hline in header.splitlines():
                f.write(f"# {hline}\n")
        f.write("# x          y\n")
        for xv, yv in zip(x, y):
            f.write(f"{xv:24.16e}  {yv:24.16e}\n")


def read_pore_structure(filename):
    """
    读取二维催化剂孔截面采样点数据。

    文件格式：每行两个浮点数 (x, y)，表示孔壁或活性位点坐标。
    基于 xy_io 的思想扩展为二维数据结构。

    Returns
    -------
    points : ndarray, shape (n, 2)
    """
    if not os.path.isfile(filename):
        raise DataIOException(f"文件不存在: {filename}")

    points = []
    with open(filename, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            points.append([float(parts[0]), float(parts[1])])

    if not points:
        raise DataIOException(f"{filename} 中未找到有效二维数据")

    return np.array(points, dtype=float)


def write_pore_structure(filename, points, header=None):
    """
    写入二维孔结构采样点数据。

    Parameters
    ----------
    points : ndarray, shape (n, 2)
    header : str, optional
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise DataIOException("points 必须是形状为 (n, 2) 的数组")

    with open(filename, 'w', encoding='utf-8') as f:
        if header is not None:
            for hline in header.splitlines():
                f.write(f"# {hline}\n")
        f.write("# x          y\n")
        for p in points:
            f.write(f"{p[0]:24.16e}  {p[1]:24.16e}\n")


def ensure_dir(path):
    """确保目录存在，若不存在则递归创建。"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
