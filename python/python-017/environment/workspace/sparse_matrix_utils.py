"""
稀疏矩阵工具模块
融合来源: 1158_st_to_mm (稀疏矩阵格式转换) + 1088_slap_io (SLAP Triad格式I/O)

功能:
- COO/Triad格式稀疏矩阵的构建、读写与格式转换
- 为有限元组装提供稀疏矩阵存储后端
- 索引边界检查与数值鲁棒性处理
"""

import numpy as np
from typing import Tuple, Optional


class SparseMatrixCOO:
    """
    Coordinate (COO) / Triad 格式稀疏矩阵。
    融合 ST 与 SLAP Triad 格式的核心思想，支持零基/一基索引转换。
    """

    def __init__(self, nrow: int, ncol: int, row: np.ndarray = None,
                 col: np.ndarray = None, data: np.ndarray = None):
        if nrow <= 0 or ncol <= 0:
            raise ValueError("矩阵维度必须为正整数")
        self.nrow = nrow
        self.ncol = ncol
        self.row = row if row is not None else np.array([], dtype=int)
        self.col = col if col is not None else np.array([], dtype=int)
        self.data = data if data is not None else np.array([], dtype=float)
        self._validate()

    def _validate(self):
        nnz = len(self.data)
        if len(self.row) != nnz or len(self.col) != nnz:
            raise ValueError("row, col, data 长度不一致")
        if nnz > 0:
            if np.any(self.row < 0) or np.any(self.row >= self.nrow):
                raise ValueError(f"行索引越界 [0, {self.nrow})")
            if np.any(self.col < 0) or np.any(self.col >= self.ncol):
                raise ValueError(f"列索引越界 [0, {self.ncol})")

    def nnz(self) -> int:
        return len(self.data)

    def rebase(self, base_from: int, base_to: int):
        """索引重基，融合 st_to_mm 中 st_rebase 思想。"""
        if self.nnz() == 0:
            return
        self.row = self.row - base_from + base_to
        self.col = self.col - base_from + base_to
        self._validate()

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵（仅用于小维度调试）。"""
        A = np.zeros((self.nrow, self.ncol), dtype=float)
        for i, j, v in zip(self.row, self.col, self.data):
            A[i, j] += v
        return A

    def to_csr(self) -> 'SparseMatrixCSR':
        """COO -> CSR 格式转换。"""
        nnz = self.nnz()
        if nnz == 0:
            return SparseMatrixCSR(self.nrow, self.ncol, np.zeros(self.nrow + 1, dtype=int),
                                   np.array([], dtype=int), np.array([], dtype=float))
        # 按行、列排序
        order = np.lexsort((self.col, self.row))
        row_s = self.row[order]
        col_s = self.col[order]
        data_s = self.data[order]

        indptr = np.zeros(self.nrow + 1, dtype=int)
        for r in row_s:
            indptr[r + 1] += 1
        indptr = np.cumsum(indptr)
        return SparseMatrixCSR(self.nrow, self.ncol, indptr, col_s, data_s)

    def add_entry(self, i: int, j: int, v: float):
        """添加一个非零元，允许重复（求和由 to_dense 处理）。"""
        if not (0 <= i < self.nrow and 0 <= j < self.ncol):
            raise IndexError(f"索引 ({i},{j}) 越界，矩阵大小 ({self.nrow},{self.ncol})")
        if not np.isfinite(v):
            return
        self.row = np.append(self.row, i)
        self.col = np.append(self.col, j)
        self.data = np.append(self.data, v)

    def symmetric_expand(self):
        """对于对称存储的下三角，扩展为全矩阵。"""
        if self.nnz() == 0:
            return
        extra_row = self.col[self.row != self.col]
        extra_col = self.row[self.row != self.col]
        extra_data = self.data[self.row != self.col]
        self.row = np.concatenate([self.row, extra_row])
        self.col = np.concatenate([self.col, extra_col])
        self.data = np.concatenate([self.data, extra_data])

    @staticmethod
    def read_from_triad_file(filename: str) -> 'SparseMatrixCOO':
        """
        从类 SLAP Triad 文本文件读取稀疏矩阵。
        文件格式: 每行 "i j value"，0基索引。
        第一行可选: nrow ncol nnz
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"无法读取文件 {filename}: {e}")

        if not lines:
            raise ValueError("空文件")

        # 尝试解析首行是否为维度信息
        first = lines[0].strip().split()
        if len(first) == 3:
            try:
                nrow, ncol, nnz_expected = int(first[0]), int(first[1]), int(first[2])
                data_lines = lines[1:]
            except ValueError:
                nrow = ncol = nnz_expected = None
                data_lines = lines
        else:
            nrow = ncol = nnz_expected = None
            data_lines = lines

        rows, cols, vals = [], [], []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append(int(parts[0]))
            cols.append(int(parts[1]))
            vals.append(float(parts[2]))

        row = np.array(rows, dtype=int)
        col = np.array(cols, dtype=int)
        data = np.array(vals, dtype=float)

        if nrow is None:
            nrow = int(row.max()) + 1 if len(row) > 0 else 0
            ncol = int(col.max()) + 1 if len(col) > 0 else 0

        return SparseMatrixCOO(nrow, ncol, row, col, data)

    def write_to_triad_file(self, filename: str):
        """写入类 SLAP Triad 格式文件。"""
        with open(filename, 'w') as f:
            f.write(f"{self.nrow} {self.ncol} {self.nnz()}\n")
            for i, j, v in zip(self.row, self.col, self.data):
                f.write(f"{i} {j} {v:.16e}\n")


class SparseMatrixCSR:
    """Compressed Sparse Row (CSR) 格式，用于快速矩阵-向量乘法。"""

    def __init__(self, nrow: int, ncol: int, indptr: np.ndarray,
                 indices: np.ndarray, data: np.ndarray):
        self.nrow = nrow
        self.ncol = ncol
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def dot(self, x: np.ndarray) -> np.ndarray:
        """CSR 矩阵-向量乘法 y = A @ x。"""
        if x.shape[0] != self.ncol:
            raise ValueError("维度不匹配")
        y = np.zeros(self.nrow, dtype=float)
        for i in range(self.nrow):
            for jj in range(self.indptr[i], self.indptr[i + 1]):
                y[i] += self.data[jj] * x[self.indices[jj]]
        return y

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.nrow, self.ncol), dtype=float)
        for i in range(self.nrow):
            for jj in range(self.indptr[i], self.indptr[i + 1]):
                A[i, self.indices[jj]] += self.data[jj]
        return A


def coo_to_dense_solve(coo: SparseMatrixCOO, b: np.ndarray) -> np.ndarray:
    """
    将 COO 矩阵转为稠密后用 NumPy 求解线性方程组 Ax = b。
    适用于中小规模有限元系统（演示/验证用途）。
    """
    A_dense = coo.to_dense()
    # 边界鲁棒性: 若矩阵奇异，加微小正则化
    cond = np.linalg.cond(A_dense)
    if cond > 1e12:
        A_dense += np.eye(A_dense.shape[0]) * 1e-12
    return np.linalg.solve(A_dense, b)
