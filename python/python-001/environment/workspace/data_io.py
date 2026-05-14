"""
data_io.py

基于 xyz_io (XYZ 格式数据读写) 与 filum (字符串/文件处理工具集)
核心算法，实现小行星几何数据与轨道数据的输入输出。

科学背景：
小行星形状模型通常以 XYZ 顶点坐标和 OBJ/OFF 面片格式存储。
本项目提供标准化的数据解析、校验与转换工具，
支持从原始测距/雷达数据生成多面体模型所需的中间格式。
"""

import numpy as np
import os
from typing import Tuple, List, Optional


class DataIOError(Exception):
    pass


def s_len_trim(s: str) -> int:
    """
    返回字符串去掉尾部空白后的长度。
    基于 s_len_trim.m。
    """
    return len(s.rstrip())


def s_word_count(s: str) -> int:
    """
    统计字符串中非空单词数。
    基于 s_word_count.m。
    """
    return len(s.split())


def filename_ext_get(filename: str) -> str:
    """
    提取文件扩展名。
    """
    _, ext = os.path.splitext(filename)
    return ext


def filename_ext_swap(filename: str, new_ext: str) -> str:
    """
    替换文件扩展名。
    """
    base, _ = os.path.splitext(filename)
    if not new_ext.startswith("."):
        new_ext = "." + new_ext
    return base + new_ext


def read_xyz_data(filepath: str) -> np.ndarray:
    """
    读取 XYZ 格式的三维点云数据。
    格式：每行 "x y z"，支持 # 注释和空行。
    基于 xyz_data_read.m。

    参数:
        filepath: 文件路径

    返回:
        points: (n, 3) 的 numpy 数组
    """
    if not os.path.exists(filepath):
        raise DataIOError(f"文件不存在: {filepath}")

    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
            else:
                continue

    if len(points) == 0:
        raise DataIOError(f"未从 {filepath} 读取到有效数据")
    return np.array(points)


def write_xyz_data(filepath: str, points: np.ndarray, header: Optional[str] = None) -> None:
    """
    将三维点云写入 XYZ 格式文件。
    基于 xyz_data_write.m。
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise DataIOError("points 必须是 (n, 3) 数组")

    with open(filepath, "w") as f:
        if header is not None:
            f.write(f"# {header}\n")
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.12e}  {points[i, 1]:.12e}  {points[i, 2]:.12e}\n")


def read_face_indices(filepath: str) -> np.ndarray:
    """
    读取面片索引文件。
    格式：每行 "v1 v2 v3"（0-based 或 1-based 索引）。
    自动检测并转换为 0-based。
    """
    if not os.path.exists(filepath):
        raise DataIOError(f"文件不存在: {filepath}")

    faces = []
    min_idx = np.inf
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    idx = [int(parts[0]), int(parts[1]), int(parts[2])]
                    faces.append(idx)
                    min_idx = min(min_idx, min(idx))
                except ValueError:
                    continue

    faces = np.array(faces, dtype=int)
    if min_idx == 1:
        faces -= 1
    return faces


def write_face_indices(filepath: str, faces: np.ndarray, zero_based: bool = True) -> None:
    """
    写入面片索引文件。
    """
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise DataIOError("faces 必须是 (n, 3) 数组")
    offset = 0 if zero_based else 1
    with open(filepath, "w") as f:
        for i in range(faces.shape[0]):
            f.write(f"{faces[i,0]+offset} {faces[i,1]+offset} {faces[i,2]+offset}\n")


def generate_synthetic_asteroid_pointcloud(
    a: float = 2.0,
    b: float = 1.5,
    c: float = 1.0,
    n_theta: int = 32,
    n_phi: int = 32,
    noise_amp: float = 0.05,
    seed: int = 42
) -> np.ndarray:
    """
    生成合成小行星的 XYZ 点云数据。
    基于椭圆体 + 随机扰动模型。

    参数:
        a, b, c: 椭球半轴 (km)
        n_theta, n_phi: 经纬度采样数
        noise_amp: 表面粗糙度相对振幅
        seed: 随机种子

    返回:
        points: (n_theta*n_phi, 3)
    """
    np.random.seed(seed)
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    points = []
    for t in theta:
        for p in phi:
            r_base = 1.0 / np.sqrt(
                (np.sin(t) * np.cos(p) / a) ** 2 +
                (np.sin(t) * np.sin(p) / b) ** 2 +
                (np.cos(t) / c) ** 2
            )
            noise = r_base * noise_amp * (2.0 * np.random.rand() - 1.0)
            r = r_base + noise
            x = r * np.sin(t) * np.cos(p)
            y = r * np.sin(t) * np.sin(p)
            z = r * np.cos(t)
            points.append([x, y, z])
    return np.array(points)


def write_obj_file(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray
) -> None:
    """
    将多面体模型写入 Wavefront OBJ 格式（纯文本，无可视化）。
    """
    with open(filepath, "w") as f:
        f.write("# Synthetic asteroid model\n")
        for v in vertices:
            f.write(f"v {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def read_obj_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 Wavefront OBJ 格式文件，提取顶点与面片。
    """
    if not os.path.exists(filepath):
        raise DataIOError(f"文件不存在: {filepath}")

    vertices = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()
                if len(parts) >= 4:
                    # 处理 "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
                    idx = []
                    for p in parts[1:4]:
                        idx.append(int(p.split("/")[0]) - 1)
                    faces.append(idx)

    if len(vertices) == 0:
        raise DataIOError("未读取到顶点数据")
    return np.array(vertices), np.array(faces, dtype=int)
