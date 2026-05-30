
import numpy as np
import os
from typing import Tuple, List, Optional


class DataIOError(Exception):
    pass


def s_len_trim(s: str) -> int:
    return len(s.rstrip())


def s_word_count(s: str) -> int:
    return len(s.split())


def filename_ext_get(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext


def filename_ext_swap(filename: str, new_ext: str) -> str:
    base, _ = os.path.splitext(filename)
    if not new_ext.startswith("."):
        new_ext = "." + new_ext
    return base + new_ext


def read_xyz_data(filepath: str) -> np.ndarray:
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
    if points.ndim != 2 or points.shape[1] != 3:
        raise DataIOError("points 必须是 (n, 3) 数组")

    with open(filepath, "w") as f:
        if header is not None:
            f.write(f"# {header}\n")
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.12e}  {points[i, 1]:.12e}  {points[i, 2]:.12e}\n")


def read_face_indices(filepath: str) -> np.ndarray:
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
    with open(filepath, "w") as f:
        f.write("# Synthetic asteroid model\n")
        for v in vertices:
            f.write(f"v {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def read_obj_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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

                    idx = []
                    for p in parts[1:4]:
                        idx.append(int(p.split("/")[0]) - 1)
                    faces.append(idx)

    if len(vertices) == 0:
        raise DataIOError("未读取到顶点数据")
    return np.array(vertices), np.array(faces, dtype=int)
