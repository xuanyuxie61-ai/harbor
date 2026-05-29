"""
mesh_io.py
网格数据与稀疏矩阵 I/O 模块

科学背景:
海洋数值模拟需要处理大规模网格数据和稀疏矩阵:
  - 三维海洋网格 (ICE/NETCDF 格式) 存储顶点、边、面、体信息
  - 稀疏矩阵 (Harwell-Boeing / Matrix Market) 存储离散算子

本模块实现:
  - 简化版 NETCDF-like 网格数据读写
  - 稀疏矩阵 CSC 格式与坐标格式转换
  - 矩阵属性存储与读取

融合来源:
- 570_ice_io: NETCDF 网格数据 I/O 思想
- 508_hb_to_mm: Harwell-Boeing / Matrix Market 格式转换
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


# ============================================================
# 1. 简化网格数据结构
# ============================================================

class OceanGrid:
    """
    简化版三维海洋网格数据结构.

    存储:
      - vertices: (n_v, 3) 顶点坐标
      - edges: (n_e, 2) 边连接
      - triangles: (n_t, 3) 三角形面
      - tetrahedra: (n_tet, 4) 四面体单元
      - labels: 单元标签 (用于标识不同区域)
    """

    def __init__(self):
        self.vertices = np.zeros((0, 3))
        self.edges = np.zeros((0, 2), dtype=int)
        self.triangles = np.zeros((0, 3), dtype=int)
        self.tetrahedra = np.zeros((0, 4), dtype=int)
        self.vertex_labels = np.array([], dtype=int)
        self.cell_labels = np.array([], dtype=int)
        self.attributes = {}

    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    def n_cells(self) -> int:
        return max(self.tetrahedra.shape[0], self.triangles.shape[0])

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_vertices() == 0:
            return np.zeros(3), np.ones(3)
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def cell_volumes(self) -> np.ndarray:
        """计算四面体单元体积."""
        if self.tetrahedra.shape[0] == 0:
            return np.array([])
        vols = []
        for tet in self.tetrahedra:
            v0 = self.vertices[tet[0]]
            v1 = self.vertices[tet[1]]
            v2 = self.vertices[tet[2]]
            v3 = self.vertices[tet[3]]
            vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
            vols.append(vol)
        return np.array(vols)

    def to_dict(self) -> Dict:
        return {
            "vertices": self.vertices,
            "edges": self.edges,
            "triangles": self.triangles,
            "tetrahedra": self.tetrahedra,
            "vertex_labels": self.vertex_labels,
            "cell_labels": self.cell_labels,
            "attributes": self.attributes
        }

    @classmethod
    def from_dict(cls, d: Dict):
        g = cls()
        g.vertices = np.asarray(d.get("vertices", np.zeros((0, 3))))
        g.edges = np.asarray(d.get("edges", np.zeros((0, 2), dtype=int)), dtype=int)
        g.triangles = np.asarray(d.get("triangles", np.zeros((0, 3), dtype=int)), dtype=int)
        g.tetrahedra = np.asarray(d.get("tetrahedra", np.zeros((0, 4), dtype=int)), dtype=int)
        g.vertex_labels = np.asarray(d.get("vertex_labels", []), dtype=int)
        g.cell_labels = np.asarray(d.get("cell_labels", []), dtype=int)
        g.attributes = d.get("attributes", {})
        return g


# ============================================================
# 2. 网格数据 I/O (简化 NETCDF 风格)
# ============================================================

def write_grid_to_file(grid: OceanGrid, filename: str):
    """将网格数据写入 numpy .npz 文件 (轻量级替代 NETCDF)."""
    np.savez(filename,
             vertices=grid.vertices,
             edges=grid.edges,
             triangles=grid.triangles,
             tetrahedra=grid.tetrahedra,
             vertex_labels=grid.vertex_labels,
             cell_labels=grid.cell_labels)


def read_grid_from_file(filename: str) -> OceanGrid:
    """从 .npz 文件读取网格数据."""
    data = np.load(filename)
    grid = OceanGrid()
    grid.vertices = data["vertices"]
    grid.edges = data["edges"]
    grid.triangles = data["triangles"]
    grid.tetrahedra = data["tetrahedra"]
    grid.vertex_labels = data["vertex_labels"]
    grid.cell_labels = data["cell_labels"]
    return grid


def create_cylinder_grid(n_r: int = 8, n_theta: int = 16, n_z: int = 4,
                         radius: float = 1.0, height: float = 2.0) -> OceanGrid:
    """
    创建简化圆柱体四面体网格 (模拟海底地形/边界).

    结构:
      - 底部和顶部各一个三角形扇
      - 侧面为三角柱, 细分为四面体
    """
    grid = OceanGrid()
    vertices = []
    # 底部中心
    vertices.append([0.0, 0.0, 0.0])
    # 底部圆周
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / n_theta
        vertices.append([radius * np.cos(theta), radius * np.sin(theta), 0.0])
    # 顶部中心
    vertices.append([0.0, 0.0, height])
    # 顶部圆周
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / n_theta
        vertices.append([radius * np.cos(theta), radius * np.sin(theta), height])

    grid.vertices = np.array(vertices)

    # 三角形面 (底部和顶部)
    bottom_tri = []
    top_tri = []
    for i in range(n_theta):
        bottom_tri.append([0, 1 + i, 1 + (i + 1) % n_theta])
        top_tri.append([n_theta + 1, n_theta + 2 + i, n_theta + 2 + (i + 1) % n_theta])

    grid.triangles = np.array(bottom_tri + top_tri)

    # 四面体 (简化为底部扇区)
    tets = []
    for i in range(n_theta):
        v0 = 0
        v1 = 1 + i
        v2 = 1 + (i + 1) % n_theta
        v3 = n_theta + 2 + i
        tets.append([v0, v1, v2, v3])
    grid.tetrahedra = np.array(tets)

    return grid


# ============================================================
# 3. 稀疏矩阵格式 (from 508_hb_to_mm)
# ============================================================

class SparseMatrix:
    """
    稀疏矩阵 CSC (Compressed Sparse Column) 格式.

    存储:
      data: 非零元素值
      row_ind: 行索引
      col_ptr: 列指针 (每列起始位置)
    """

    def __init__(self, data: np.ndarray, row_ind: np.ndarray, col_ptr: np.ndarray,
                 shape: Tuple[int, int]):
        self.data = np.asarray(data, dtype=float)
        self.row_ind = np.asarray(row_ind, dtype=int)
        self.col_ptr = np.asarray(col_ptr, dtype=int)
        self.shape = shape

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵 (仅用于小矩阵)."""
        A = np.zeros(self.shape)
        for j in range(self.shape[1]):
            for idx in range(self.col_ptr[j], self.col_ptr[j + 1]):
                i = self.row_ind[idx]
                A[i, j] = self.data[idx]
        return A

    @classmethod
    def from_dense(cls, A: np.ndarray, tol: float = 1e-15):
        """从稠密矩阵构造 CSC."""
        A = np.asarray(A)
        m, n = A.shape
        data = []
        row_ind = []
        col_ptr = [0]
        for j in range(n):
            for i in range(m):
                if abs(A[i, j]) > tol:
                    data.append(A[i, j])
                    row_ind.append(i)
            col_ptr.append(len(data))
        return cls(np.array(data), np.array(row_ind), np.array(col_ptr), (m, n))

    def transpose(self):
        """转置 (CSR → CSC)."""
        dense = self.to_dense()
        return SparseMatrix.from_dense(dense.T)


def write_matrix_market(A: SparseMatrix, filename: str, symmetry: str = "general"):
    """
    写入 Matrix Market 坐标格式.

    格式:
      %%MatrixMarket matrix coordinate real general
      M N NNZ
      i j value
      ...
    """
    m, n = A.shape
    nnz = len(A.data)
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real {}\n".format(symmetry))
        f.write("{} {} {}\n".format(m, n, nnz))
        for j in range(n):
            for idx in range(A.col_ptr[j], A.col_ptr[j + 1]):
                i = A.row_ind[idx]
                val = A.data[idx]
                f.write("{} {} {:.16e}\n".format(i + 1, j + 1, val))


def read_matrix_market(filename: str) -> SparseMatrix:
    """读取 Matrix Market 坐标格式."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 跳过注释
    data_lines = [l for l in lines if not l.startswith('%')]
    header = data_lines[0].strip().split()
    m, n, nnz = int(header[0]), int(header[1]), int(header[2])

    rows = []
    cols = []
    vals = []
    for line in data_lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            rows.append(int(parts[0]) - 1)
            cols.append(int(parts[1]) - 1)
            vals.append(float(parts[2]))

    # 转换为 CSC
    A_dense = np.zeros((m, n))
    for i, j, v in zip(rows, cols, vals):
        A_dense[i, j] = v
    return SparseMatrix.from_dense(A_dense)


def write_harwell_boeing(A: SparseMatrix, filename: str,
                         title: str = "SPARSE MATRIX", key: str = "(1I14)"):
    """
    简化版 Harwell-Boeing 格式写入.

    头部包含:
      Line 1: Title (72 chars) + Key (8 chars)
      Line 2: Totcrd, Ptrcrd, Indcrd, Valcrd, Rhscrd
      Line 3: MXTYPE, NROW, NCOL, NNZERO, NELTVL
      Line 4: Ptrfmt, Indfmt, Valfmt, Rhsfmt
    """
    m, n = A.shape
    nnz = len(A.data)
    ptrcrd = (n + 1 + 7) // 8  # 每行8个整数
    indcrd = (nnz + 7) // 8
    valcrd = (nnz + 2) // 3    # 每行3个浮点数
    totcrd = 3 + ptrcrd + indcrd + valcrd

    with open(filename, 'w') as f:
        f.write(f"{title:<72}{key:<8}\n")
        f.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{0:14d}\n")
        f.write(f"{'RUA':<14}{m:14d}{n:14d}{nnz:14d}{0:14d}\n")
        f.write(f"{'(8I10)':<16}{'(8I10)':<16}{'(3E26.18)':<20}{'(3E26.18)':<20}\n")

        # 列指针
        ptr_strs = [f"{v:10d}" for v in A.col_ptr]
        for i in range(0, len(ptr_strs), 8):
            f.write("".join(ptr_strs[i:i + 8]) + "\n")

        # 行索引
        ind_strs = [f"{v:10d}" for v in (A.row_ind + 1)]
        for i in range(0, len(ind_strs), 8):
            f.write("".join(ind_strs[i:i + 8]) + "\n")

        # 数值
        val_strs = [f"{v:26.18e}" for v in A.data]
        for i in range(0, len(val_strs), 3):
            f.write("".join(val_strs[i:i + 3]) + "\n")


def construct_laplacian_csc(Nx: int, Ny: int, dx: float, dy: float) -> SparseMatrix:
    """
    构造二维五点 Laplacian 算子的 CSC 稀疏矩阵.

    离散格式:
      \nabla^2 u ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx^2 + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy^2

    矩阵维度: (Nx*Ny) x (Nx*Ny)
    """
    n = Nx * Ny
    data = []
    row_ind = []
    col_ptr = [0]

    idx = lambda i, j: j * Nx + i
    inv_dx2 = 1.0 / (dx ** 2)
    inv_dy2 = 1.0 / (dy ** 2)

    for j in range(Ny):
        for i in range(Nx):
            diag = -2.0 * (inv_dx2 + inv_dy2)
            entries = [(idx(i, j), diag)]

            if i > 0:
                entries.append((idx(i - 1, j), inv_dx2))
            if i < Nx - 1:
                entries.append((idx(i + 1, j), inv_dx2))
            if j > 0:
                entries.append((idx(i, j - 1), inv_dy2))
            if j < Ny - 1:
                entries.append((idx(i, j + 1), inv_dy2))

            # 排序并存储
            entries.sort(key=lambda x: x[0])
            for r, v in entries:
                data.append(v)
                row_ind.append(r)
            col_ptr.append(len(data))

    return SparseMatrix(np.array(data), np.array(row_ind), np.array(col_ptr), (n, n))


if __name__ == "__main__":
    # 测试网格
    grid = create_cylinder_grid(n_r=4, n_theta=8, n_z=2)
    print("Cylinder grid vertices:", grid.n_vertices())
    print("Tetrahedra:", grid.tetrahedra.shape[0])

    # 测试稀疏矩阵
    A = construct_laplacian_csc(4, 4, 0.1, 0.1)
    print("Laplacian shape:", A.shape, "nnz:", len(A.data))
    Ad = A.to_dense()
    print("Row sum (should be ~0 for interior):", np.sum(Ad, axis=1))

    # 测试 MM 格式
    write_matrix_market(A, "/tmp/test_matrix.mtx")
    A2 = read_matrix_market("/tmp/test_matrix.mtx")
    print("MM roundtrip max diff:", np.max(np.abs(A.to_dense() - A2.to_dense())))
