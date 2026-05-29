"""
mesh_io.py
有限元网格与稀疏矩阵 I/O 工具 — FEM/MEDIT 格式与压缩存储

基于种子项目 770_mm_to_hb (Matrix Market to Harwell-Boeing 转换)
与 379_fem_to_medit (FEM to MEDIT mesh 转换) 的格式处理思想，
实现冰盖有限元模拟中的网格与稀疏刚度矩阵的读写与格式转换。

支持格式:
  1. 简单 FEM 文本格式 (节点 + 单元)
  2. MEDIT .mesh 格式 (Triangles/Tetrahedra)
  3. 坐标格式稀疏矩阵 (COO)
  4. Harwell-Boeing 风格压缩列存储 (CSC)

核心数学:
  - 稀疏刚度矩阵 K 的组装:
      K = \sum_e K_e,  \quad K_e = \int_{\Omega_e} B^T D B \, d\Omega

  - 压缩存储:
      CSC: val, row_ind, col_ptr
      其中 col_ptr[j] 表示第 j 列的起始索引
"""

import numpy as np
from typing import List, Tuple, Optional


def read_fem_nodes(filepath: str) -> np.ndarray:
    """
    读取 FEM 节点文件 (每行: x y z [可选边界标记])。
    跳过空行与 # 注释。
    """
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    coords = [float(p) for p in parts[:3]] if len(parts) >= 3 else [float(parts[0]), float(parts[1]), 0.0]
                    rows.append(coords)
    except FileNotFoundError:
        # 若无文件，返回空数组
        return np.zeros((0, 3), dtype=np.float64)

    return np.array(rows, dtype=np.float64)


def read_fem_elements(filepath: str) -> np.ndarray:
    """
    读取 FEM 单元文件 (每行: n1 n2 n3 [n4] [区域标记])。
    """
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    nodes = [int(p) for p in parts[:3]]
                    rows.append(nodes)
    except FileNotFoundError:
        return np.zeros((0, 3), dtype=np.int64)

    return np.array(rows, dtype=np.int64)


def write_medit_mesh(nodes: np.ndarray,
                     elements: np.ndarray,
                     boundary_nodes: Optional[np.ndarray] = None,
                     filepath: str = "ice_mesh.mesh") -> None:
    """
    写入 MEDIT .mesh 格式文件。

    格式规范 (Pascal Frey):
        MeshVersionFormatted 1
        Dimension 3
        Vertices
        N
        x y z ref
        Triangles
        M
        n1 n2 n3 ref
        End

    参数:
        nodes: (N, 3) 节点坐标
        elements: (M, 3) 三角形单元 (1-based 索引)
        boundary_nodes: (N,) 边界标记 (0/1)
        filepath: 输出路径
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if elements.size > 0 and np.min(elements) == 0:
        elements = elements + 1  # 转为 1-based

    n_nodes = len(nodes)
    n_elems = len(elements)

    refs = np.ones(n_nodes, dtype=np.int64)
    if boundary_nodes is not None:
        refs = np.asarray(boundary_nodes, dtype=np.int64)

    with open(filepath, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n")
        f.write("Vertices\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{nodes[i, 0]:.6e} {nodes[i, 1]:.6e} {nodes[i, 2]:.6e} {refs[i]}\n")

        if n_elems > 0:
            f.write("Triangles\n")
            f.write(f"{n_elems}\n")
            for e in elements:
                f.write(f"{e[0]} {e[1]} {e[2]} 1\n")

        f.write("End\n")


def read_medit_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 MEDIT .mesh 文件。

    返回:
        nodes: (N, 3)
        elements: (M, 3) 0-based 索引
    """
    nodes = []
    elements = []
    mode = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                lower = line.lower()

                if lower == 'vertices':
                    mode = 'vertices_count'
                    continue
                elif lower == 'triangles':
                    mode = 'triangles_count'
                    continue
                elif lower == 'end':
                    mode = None
                    continue

                if mode == 'vertices_count':
                    mode = 'vertices'
                    continue
                elif mode == 'triangles_count':
                    mode = 'triangles'
                    continue

                parts = line.split()
                if mode == 'vertices' and len(parts) >= 3:
                    nodes.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif mode == 'triangles' and len(parts) >= 3:
                    elements.append([int(parts[0]), int(parts[1]), int(parts[2])])
    except FileNotFoundError:
        pass

    nodes_arr = np.array(nodes, dtype=np.float64)
    elems_arr = np.array(elements, dtype=np.int64)
    if elems_arr.size > 0:
        elems_arr = elems_arr - 1  # 转为 0-based
    return nodes_arr, elems_arr


def coo_to_csc(values: np.ndarray, row_indices: np.ndarray,
               col_indices: np.ndarray, n: int, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 COO 格式稀疏矩阵转换为 CSC (Compressed Sparse Column) 格式。

    CSC 格式:
        data: 非零元值
        row_ind: 非零元的行索引
        col_ptr: 第 j 列起始位置指针 (长度 m+1)

    参数:
        values: COO 非零值
        row_indices: COO 行索引
        col_indices: COO 列索引
        n: 行数
        m: 列数

    返回:
        data, row_ind, col_ptr
    """
    nnz = len(values)
    if nnz == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64), np.zeros(m + 1, dtype=np.int64)

    # 按 (列, 行) 排序
    order = np.lexsort((row_indices, col_indices))
    data = values[order]
    rows = row_indices[order]
    cols = col_indices[order]

    col_ptr = np.zeros(m + 1, dtype=np.int64)
    for j in range(m):
        col_ptr[j] = np.searchsorted(cols, j, side='left')
    col_ptr[m] = nnz

    return data, rows, col_ptr


def csc_to_dense(data: np.ndarray, row_ind: np.ndarray,
                 col_ptr: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    将 CSC 格式转换回稠密矩阵。
    """
    A = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        for idx in range(col_ptr[j], col_ptr[j + 1]):
            i = row_ind[idx]
            A[i, j] = data[idx]
    return A


def assemble_ice_stiffness_matrix_2d(nodes: np.ndarray,
                                      elements: np.ndarray,
                                      diffusivity: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    组装二维冰扩散问题的有限元刚度矩阵 (线性三角形元)。

    弱形式:
        \int_{\Omega} D \nabla \phi \cdot \nabla \psi \, d\Omega

    单元刚度:
        K_e = D \cdot A_e \cdot B_e^T B_e

    其中 B_e 为梯度算子矩阵 (3x2)，A_e 为三角形面积。

    参数:
        nodes: (N, 2) 或 (N, 3) 节点坐标
        elements: (M, 3) 三角形单元 (0-based)
        diffusivity: 扩散系数 D

    返回:
        data, rows, cols, n, n (COO 格式)
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]

    n_nodes = len(nodes)
    vals = []
    rows = []
    cols = []

    for e in elements:
        n1, n2, n3 = e
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]

        # 三角形面积
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area < 1e-15:
            continue

        # 形函数梯度 (常数)
        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * area)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * area)

        # Ke[i,j] = D * area * (b_i*b_j + c_i*c_j)
        for i_local in range(3):
            for j_local in range(3):
                gi = e[i_local]
                gj = e[j_local]
                ke_ij = diffusivity * area * (b[i_local] * b[j_local] + c[i_local] * c[j_local])
                vals.append(ke_ij)
                rows.append(gi)
                cols.append(gj)

    return np.array(vals, dtype=np.float64), np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), n_nodes, n_nodes


def write_harwell_boeing_csc(data: np.ndarray, row_ind: np.ndarray,
                              col_ptr: np.ndarray, n: int, m: int,
                              filepath: str = "ice_matrix.hb") -> None:
    """
    写入简化版 Harwell-Boeing 格式 (仅实数非对称)。
    """
    nnz = len(data)
    with open(filepath, 'w') as f:
        f.write(f"{n} {m} {nnz}\n")
        f.write(" ".join(str(v) for v in col_ptr) + "\n")
        f.write(" ".join(str(v) for v in row_ind) + "\n")
        f.write(" ".join(f"{v:.12e}" for v in data) + "\n")


def read_harwell_boeing_csc(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    读取简化版 Harwell-Boeing 格式。
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return (np.zeros(0), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64), 0, 0)

    header = lines[0].strip().split()
    n, m, nnz = int(header[0]), int(header[1]), int(header[2])
    col_ptr = np.array([int(x) for x in lines[1].strip().split()], dtype=np.int64)
    row_ind = np.array([int(x) for x in lines[2].strip().split()], dtype=np.int64)
    data = np.array([float(x) for x in lines[3].strip().split()], dtype=np.float64)
    return data, row_ind, col_ptr, n, m
