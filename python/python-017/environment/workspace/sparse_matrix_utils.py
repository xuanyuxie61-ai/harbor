
import numpy as np
from typing import Tuple, Optional


class SparseMatrixCOO:

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
        if self.nnz() == 0:
            return
        self.row = self.row - base_from + base_to
        self.col = self.col - base_from + base_to
        self._validate()

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.nrow, self.ncol), dtype=float)
        for i, j, v in zip(self.row, self.col, self.data):
            A[i, j] += v
        return A

    def to_csr(self) -> 'SparseMatrixCSR':
        nnz = self.nnz()
        if nnz == 0:
            return SparseMatrixCSR(self.nrow, self.ncol, np.zeros(self.nrow + 1, dtype=int),
                                   np.array([], dtype=int), np.array([], dtype=float))

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
        if not (0 <= i < self.nrow and 0 <= j < self.ncol):
            raise IndexError(f"索引 ({i},{j}) 越界，矩阵大小 ({self.nrow},{self.ncol})")
        if not np.isfinite(v):
            return
        self.row = np.append(self.row, i)
        self.col = np.append(self.col, j)
        self.data = np.append(self.data, v)

    def symmetric_expand(self):
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
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"无法读取文件 {filename}: {e}")

        if not lines:
            raise ValueError("空文件")


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
        with open(filename, 'w') as f:
            f.write(f"{self.nrow} {self.ncol} {self.nnz()}\n")
            for i, j, v in zip(self.row, self.col, self.data):
                f.write(f"{i} {j} {v:.16e}\n")


class SparseMatrixCSR:

    def __init__(self, nrow: int, ncol: int, indptr: np.ndarray,
                 indices: np.ndarray, data: np.ndarray):
        self.nrow = nrow
        self.ncol = ncol
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def dot(self, x: np.ndarray) -> np.ndarray:
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
    A_dense = coo.to_dense()

    cond = np.linalg.cond(A_dense)
    if cond > 1e12:
        A_dense += np.eye(A_dense.shape[0]) * 1e-12
    return np.linalg.solve(A_dense, b)
