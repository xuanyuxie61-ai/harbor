
import os
import numpy as np


class DataIOException(Exception):
    pass


def read_xy_profile(filename):
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

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    return x, y


def write_xy_profile(filename, x, y, header=None):
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
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
