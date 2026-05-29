r"""
sparse_matrix_utils.py
稀疏矩阵格式转换与操作模块

本模块实现：
1. Matrix Market (MM) 格式解析与构造
2. Sparse Triplet (ST) 格式读写
3. Coordinate (COO) 格式与 Compressed Sparse Row (CSR) 格式互转
4. 稀疏矩阵-向量乘法
5. 对称矩阵的完全展开

核心公式：

A. 稀疏矩阵-向量乘法（CSR 格式）：
    y_i = \sum_{j=idx[i]}^{idx[i+1]-1} data[j] \cdot x[col[j]]

B. COO → CSR 转换：
    1. 统计每行非零元个数：row_counts[row[i]]++
    2. 前缀和得到行指针：idx[i+1] = idx[i] + row_counts[i]
    3. 按行填入：pos[row[i]]++，data[pos] = val[i]，col[pos] = col[i]

C. 对称矩阵展开：
    若 A_{ij} 给定且 i > j，则 A_{ji} = A_{ij}

参考文献：
- Matrix Market: https://math.nist.gov/MatrixMarket/
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
"""

import numpy as np
from typing import Tuple, List, Dict


# ============================================================================
# COO 稀疏矩阵类
# ============================================================================

class SparseMatrixCOO:
    """
    Coordinate (COO) 格式稀疏矩阵。

    存储：
        rows: (nnz,) 行索引（0-based）
        cols: (nnz,) 列索引（0-based）
        data: (nnz,) 非零元素值
        shape: (m, n) 矩阵维度
    """

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []

    def add(self, i: int, j: int, val: float):
        """添加非零元素。"""
        if not (0 <= i < self.m and 0 <= j < self.n):
            raise IndexError(f"SparseMatrixCOO: 索引 ({i}, {j}) 超出维度 ({self.m}, {self.n})")
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(float(val))

    def nnz(self) -> int:
        return len(self.data)

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵（仅用于小矩阵调试）。"""
        A = np.zeros((self.m, self.n))
        for i, j, v in zip(self.rows, self.cols, self.data):
            A[i, j] += v
        return A

    def to_csr(self) -> "SparseMatrixCSR":
        """转换为 CSR 格式。"""
        return coo_to_csr(self)


# ============================================================================
# CSR 稀疏矩阵类
# ============================================================================

class SparseMatrixCSR:
    """
    Compressed Sparse Row (CSR) 格式稀疏矩阵。

    存储：
        data: (nnz,) 非零元素值
        col_idx: (nnz,) 列索引
        row_ptr: (m+1,) 行指针
        shape: (m, n)
    """

    def __init__(self, data: np.ndarray, col_idx: np.ndarray, row_ptr: np.ndarray, m: int, n: int):
        self.data = np.asarray(data, dtype=float)
        self.col_idx = np.asarray(col_idx, dtype=int)
        self.row_ptr = np.asarray(row_ptr, dtype=int)
        self.m = m
        self.n = n

    def nnz(self) -> int:
        return len(self.data)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A @ x。

        算法：
            for i = 0 to m-1:
                y_i = 0
                for j = row_ptr[i] to row_ptr[i+1]-1:
                    y_i += data[j] * x[col_idx[j]]
        """
        x = np.asarray(x, dtype=float)
        if len(x) != self.n:
            raise ValueError("SparseMatrixCSR.matvec: 向量维度不匹配")
        y = np.zeros(self.m)
        for i in range(self.m):
            s = 0.0
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                s += self.data[idx] * x[self.col_idx[idx]]
            y[i] = s
        return y

    def matvec_transpose(self, x: np.ndarray) -> np.ndarray:
        """
        转置矩阵-向量乘法 y = A^T @ x。
        """
        x = np.asarray(x, dtype=float)
        if len(x) != self.m:
            raise ValueError("SparseMatrixCSR.matvec_transpose: 向量维度不匹配")
        y = np.zeros(self.n)
        for i in range(self.m):
            xi = x[i]
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[self.col_idx[idx]] += self.data[idx] * xi
        return y

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵。"""
        A = np.zeros((self.m, self.n))
        for i in range(self.m):
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.col_idx[idx]] = self.data[idx]
        return A


# ============================================================================
# 格式转换
# ============================================================================

def coo_to_csr(coo: SparseMatrixCOO) -> SparseMatrixCSR:
    """
    COO → CSR 转换。

    算法复杂度：O(nnz + m)
    """
    m, n = coo.m, coo.n
    nnz = coo.nnz()
    if nnz == 0:
        return SparseMatrixCSR(np.zeros(0), np.zeros(0, dtype=int), np.zeros(m + 1, dtype=int), m, n)

    rows = np.array(coo.rows, dtype=int)
    cols = np.array(coo.cols, dtype=int)
    data = np.array(coo.data, dtype=float)

    # 按行排序
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    # 构建 row_ptr
    row_ptr = np.zeros(m + 1, dtype=int)
    for r in rows:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    return SparseMatrixCSR(data, cols, row_ptr, m, n)


def csr_to_coo(csr: SparseMatrixCSR) -> SparseMatrixCOO:
    """CSR → COO 转换。"""
    coo = SparseMatrixCOO(csr.m, csr.n)
    for i in range(csr.m):
        for idx in range(csr.row_ptr[i], csr.row_ptr[i + 1]):
            coo.add(i, csr.col_idx[idx], csr.data[idx])
    return coo


def expand_symmetric_coo(coo: SparseMatrixCOO, lower: bool = True) -> SparseMatrixCOO:
    """
    展开对称矩阵的另一半。

    参数：
        coo: 仅含下三角（lower=True）或上三角的 COO 矩阵
        lower: 输入是否为下三角

    返回：
        full: 完整矩阵的 COO 表示
    """
    full = SparseMatrixCOO(coo.m, coo.n)
    for i, j, v in zip(coo.rows, coo.cols, coo.data):
        full.add(i, j, v)
        if i != j:
            full.add(j, i, v)
    return full


# ============================================================================
# Matrix Market 格式 I/O
# ============================================================================

def read_matrix_market(filename: str) -> SparseMatrixCOO:
    """
    读取 Matrix Market 坐标格式文件。

    文件格式（示例）：
        %%MatrixMarket matrix coordinate real general
        % 注释行
        m n nnz
        i j value
        ...
    """
    coo = None
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        header = line.strip().split()
        m, n, nnz_expected = int(header[0]), int(header[1]), int(header[2])
        coo = SparseMatrixCOO(m, n)
        for _ in range(nnz_expected):
            parts = f.readline().strip().split()
            i, j = int(parts[0]) - 1, int(parts[1]) - 1  # 1-based → 0-based
            val = float(parts[2])
            coo.add(i, j, val)
    return coo


def write_matrix_market(filename: str, coo: SparseMatrixCOO, symmetric: bool = False) -> None:
    """
    写入 Matrix Market 坐标格式文件。
    """
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{coo.m} {coo.n} {coo.nnz()}\n")
        for i, j, v in zip(coo.rows, coo.cols, coo.data):
            f.write(f"{i + 1} {j + 1} {v:.16e}\n")


def write_sparse_triplet(filename: str, coo: SparseMatrixCOO, base_zero: bool = True) -> None:
    """
    写入 Sparse Triplet 格式文件。

    格式：
        i j value
    每行一个非零元。
    """
    offset = 0 if base_zero else 1
    with open(filename, 'w') as f:
        for i, j, v in zip(coo.rows, coo.cols, coo.data):
            f.write(f"{i + offset} {j + offset} {v:.16e}\n")


def read_sparse_triplet(filename: str, m: int, n: int, base_zero: bool = True) -> SparseMatrixCOO:
    """读取 Sparse Triplet 格式文件。"""
    coo = SparseMatrixCOO(m, n)
    offset = 0 if base_zero else -1
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            i = int(parts[0]) + offset
            j = int(parts[1]) + offset
            v = float(parts[2])
            coo.add(i, j, v)
    return coo


# ============================================================================
# 实用工具
# ============================================================================

def construct_fem_stiffness_sparse(n: int, h: float, epsilon: float = 1.0) -> SparseMatrixCOO:
    """
    构造一维 Poisson 方程 FEM 刚度矩阵的稀疏表示。

    矩阵形式（n 个节点）：
        A_{ii}     = 2ε/h
        A_{i,i+1}  = A_{i+1,i} = -ε/h

    参数：
        n: 节点数
        h: 网格间距
        epsilon: 介电常数

    返回：
        coo: 稀疏矩阵
    """
    coo = SparseMatrixCOO(n, n)
    diag_val = 2.0 * epsilon / h
    offdiag_val = -epsilon / h
    for i in range(n):
        coo.add(i, i, diag_val)
        if i + 1 < n:
            coo.add(i, i + 1, offdiag_val)
            coo.add(i + 1, i, offdiag_val)
    return coo


def sparse_matvec_timing(csr: SparseMatrixCSR, x: np.ndarray, n_repeat: int = 100) -> float:
    """
    测量稀疏矩阵-向量乘法的平均耗时（秒）。

    返回：
        avg_time: 单次 matvec 平均耗时 [s]
    """
    import time
    start = time.perf_counter()
    for _ in range(n_repeat):
        _ = csr.matvec(x)
    end = time.perf_counter()
    return (end - start) / n_repeat


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 COO 构造
    coo = SparseMatrixCOO(5, 5)
    coo.add(0, 0, 2.0)
    coo.add(0, 1, -1.0)
    coo.add(1, 0, -1.0)
    coo.add(1, 1, 2.0)
    coo.add(2, 2, 3.0)
    assert coo.nnz() == 5

    # 测试 COO → CSR
    csr = coo.to_csr()
    assert csr.nnz() == 5
    assert csr.m == 5 and csr.n == 5

    # 测试 matvec
    x = np.ones(5)
    y = csr.matvec(x)
    assert abs(y[0] - 1.0) < 1e-12
    assert abs(y[1] - 1.0) < 1e-12
    assert abs(y[2] - 3.0) < 1e-12

    # 测试转置乘法
    yt = csr.matvec_transpose(x)
    assert abs(yt[0] - 1.0) < 1e-12

    # 测试对称展开
    sym = SparseMatrixCOO(3, 3)
    sym.add(0, 0, 1.0)
    sym.add(1, 0, 2.0)
    sym.add(2, 1, 3.0)
    full = expand_symmetric_coo(sym)
    dense_full = full.to_dense()
    assert abs(dense_full[0, 1] - 2.0) < 1e-12
    assert abs(dense_full[1, 0] - 2.0) < 1e-12
    assert abs(dense_full[1, 2] - 3.0) < 1e-12
    assert abs(dense_full[2, 1] - 3.0) < 1e-12

    # 测试 FEM 刚度矩阵
    fem_coo = construct_fem_stiffness_sparse(10, 0.1)
    fem_csr = fem_coo.to_csr()
    y_fem = fem_csr.matvec(np.ones(10))
    # 内部节点应满足 A @ 1 = 0（Dirichlet 边界除外）
    assert abs(y_fem[5]) < 1e-12, "FEM 刚度矩阵 × 全1向量 内部节点应为0"

    print("sparse_matrix_utils.py: 所有自测通过")
