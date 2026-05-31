"""
mesh_triangulation.py
基于种子项目 1336_triangulation_display (triangulation data processing)
和 873_ply_io (PLY file I/O)
改造为钙钛矿太阳能电池三维多晶薄膜网格生成与数据处理器。

在钙钛矿太阳能电池中，光吸收层由大量晶粒组成，每个晶粒可近似为
空间中的多面体。PLY 格式用于存储晶粒顶点与面片信息，
三角剖分用于生成有限元/有限体积法所需的二维截面网格。

核心公式：
  1. 二维 Delaunay 三角剖分：
       对于点集 P = {p_i}，构造三角形集合 T，使得
       每个三角形的外接圆内不含其他点。
  2. 三角形面积（Heron 公式）：
       A = sqrt(s(s-a)(s-b)(s-c)), s = (a+b+c)/2
  3. PLY 文件头格式：
       element vertex N
       property float x / y / z
       element face M
       property list uchar int vertex_indices
"""

import numpy as np
from typing import Tuple, List


class TriMesh:
    """
    二维/三维三角网格数据结构。
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Parameters
        ----------
        vertices : (n_v, 2) or (n_v, 3) array
        faces : (n_f, 3) array of int
            每个面的顶点索引（0-based）
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("面片必须是三角形（每行3个顶点索引）")
        if self.vertices.shape[0] < 3:
            raise ValueError("顶点数必须 ≥ 3")

    def compute_areas(self) -> np.ndarray:
        """计算每个三角形的面积。"""
        areas = np.zeros(len(self.faces))
        for i, f in enumerate(self.faces):
            v = self.vertices[f]
            if v.shape[1] == 2:
                # 二维：叉积的模的一半
                a = v[1] - v[0]
                b = v[2] - v[0]
                areas[i] = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
            else:
                # 三维：叉积模的一半
                a = v[1] - v[0]
                b = v[2] - v[0]
                cross = np.cross(a, b)
                areas[i] = 0.5 * np.linalg.norm(cross)
        return areas

    def compute_centroids(self) -> np.ndarray:
        """计算每个三角形的重心。"""
        centroids = np.zeros((len(self.faces), self.vertices.shape[1]))
        for i, f in enumerate(self.faces):
            centroids[i] = self.vertices[f].mean(axis=0)
        return centroids


def generate_grain_mesh(
    n_grains_x: int = 4,
    n_grains_y: int = 4,
    length: float = 1.0e-4,  # cm
    thickness: float = 5.0e-5,  # cm
    randomness: float = 0.1,
) -> TriMesh:
    """
    生成钙钛矿薄膜多晶结构的简化三角网格。
    在 xy 平面生成 Voronoi 风格的随机晶粒顶点，
    然后对每个晶粒进行三角剖分。
    """
    if n_grains_x < 2 or n_grains_y < 2:
        raise ValueError("晶粒数必须 ≥ 2")

    # 生成规则网格 + 随机扰动
    nx, ny = n_grains_x + 1, n_grains_y + 1
    xs = np.linspace(0, length, nx)
    ys = np.linspace(0, length, ny)
    X, Y = np.meshgrid(xs, ys)
    # 边界点不扰动，内部点随机扰动
    rng = np.random.default_rng(42)
    dx = length / n_grains_x
    dy = length / n_grains_y
    interior = (X > 0) & (X < length) & (Y > 0) & (Y < length)
    X[interior] += rng.uniform(-dx * randomness, dx * randomness, size=X[interior].shape)
    Y[interior] += rng.uniform(-dy * randomness, dy * randomness, size=Y[interior].shape)

    vertices = np.column_stack([X.ravel(), Y.ravel()])

    # 简单地对每个矩形单元进行对角线剖分
    faces = []
    for j in range(n_grains_y):
        for i in range(n_grains_x):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + i
            n3 = (j + 1) * nx + (i + 1)
            faces.append([n0, n1, n2])
            faces.append([n1, n3, n2])

    return TriMesh(vertices, np.array(faces, dtype=int))


def write_ply(mesh: TriMesh, filename: str, is_3d: bool = False) -> None:
    """
    将网格写入 PLY 格式（对应原项目 ply_write）。
    若 is_3d=True，则 z 坐标设为 0（二维网格）。
    """
    nv = mesh.vertices.shape[0]
    nf = mesh.faces.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Perovskite grain mesh\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in mesh.vertices:
            if is_3d and mesh.vertices.shape[1] == 3:
                f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            else:
                f.write(f"{v[0]:.6e} {v[1]:.6e} 0.0\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def read_ply(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 PLY 文件（对应原项目 ply_read 的简化版）。
    返回 (vertices, faces)。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and lines[idx].strip() != "end_header":
        idx += 1
    idx += 1

    # 从头信息解析顶点数和面数
    nv, nf = 0, 0
    for line in lines[:idx]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "element":
            if parts[1] == "vertex":
                nv = int(parts[2])
            elif parts[1] == "face":
                nf = int(parts[2])

    vertices = []
    faces = []
    for i in range(nv):
        parts = lines[idx + i].strip().split()
        vertices.append([float(p) for p in parts[:3]])

    for i in range(nf):
        parts = lines[idx + nv + i].strip().split()
        n_vert = int(parts[0])
        faces.append([int(p) for p in parts[1:1 + n_vert]])

    return np.array(vertices), np.array(faces)


def tri_mesh_to_ply(mesh: TriMesh, filename: str) -> None:
    """兼容原项目 tri_mesh_to_ply 接口。"""
    write_ply(mesh, filename, is_3d=(mesh.vertices.shape[1] == 3))


def ply_to_tri_mesh(filename: str) -> TriMesh:
    """兼容原项目 ply_to_tri_mesh 接口。"""
    vertices, faces = read_ply(filename)
    # 只保留三角形
    tri_faces = [f for f in faces if len(f) == 3]
    return TriMesh(vertices[:, :2] if vertices.shape[1] >= 2 else vertices,
                   np.array(tri_faces, dtype=int))


if __name__ == "__main__":
    mesh = generate_grain_mesh(4, 4)
    areas = mesh.compute_areas()
    print(f"生成网格：{mesh.vertices.shape[0]} 顶点, {mesh.faces.shape[0]} 三角形")
    print(f"总面积: {areas.sum():.6e} cm^2")
    fname = "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/161_synth_project/test_mesh.ply"
    write_ply(mesh, fname)
    verts, faces = read_ply(fname)
    print(f"读取 PLY：{len(verts)} 顶点, {len(faces)} 面")
    import os
    os.remove(fname)
