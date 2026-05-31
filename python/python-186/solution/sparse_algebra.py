"""
sparse_algebra.py
稀疏矩阵代数模块 (CCS格式)

基于种子项目:
- 975_r8ccs: 稀疏矩阵CCS格式库
"""

import numpy as np
from typing import Tuple, List, Optional


class SparseCCS:
    """
    Compressed Column Storage (CCS) 稀疏矩阵。

    数据结构:
        colptr: (n_cols + 1,) 列指针
        rowind: (nnz,) 非零元行索引
        values: (nnz,) 非零元值

    列 j 的非零元位于 indices[colptr[j]:colptr[j+1]]
    """

    def __init__(self, n_rows: int, n_cols: int,
                 colptr: np.ndarray, rowind: np.ndarray, values: np.ndarray):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.colptr = colptr.copy()
        self.rowind = rowind.copy()
        self.values = values.copy()
        self.nnz = len(values)

        self._validate()

    def _validate(self):
        assert self.colptr.shape[0] == self.n_cols + 1
        assert self.colptr[0] == 0
        assert self.colptr[-1] == self.nnz
        assert len(self.rowind) == self.nnz
        assert len(self.values) == self.nnz
        assert np.all(self.rowind >= 0) and np.all(self.rowind < self.n_rows)
        # 检查每列内部行索引升序
        for j in range(self.n_cols):
            start = self.colptr[j]
            end = self.colptr[j + 1]
            if end > start:
                rows = self.rowind[start:end]
                assert np.all(np.diff(rows) > 0), f"Column {j} rows not sorted"

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> 'SparseCCS':
        """从稠密矩阵构造CCS稀疏矩阵"""
        n_rows, n_cols = dense.shape
        colptr = [0]
        rowind = []
        values = []

        for j in range(n_cols):
            col_nonzeros = []
            for i in range(n_rows):
                if abs(dense[i, j]) > 1e-15:
                    col_nonzeros.append((i, dense[i, j]))
            col_nonzeros.sort(key=lambda x: x[0])
            for i, v in col_nonzeros:
                rowind.append(i)
                values.append(v)
            colptr.append(len(values))

        return cls(n_rows, n_cols,
                   np.array(colptr, dtype=np.int32),
                   np.array(rowind, dtype=np.int32),
                   np.array(values, dtype=np.float64))

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵"""
        dense = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        for j in range(self.n_cols):
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                dense[i, j] = self.values[idx]
        return dense

    def get(self, i: int, j: int) -> float:
        """获取元素 A[i,j]"""
        if i < 0 or i >= self.n_rows or j < 0 or j >= self.n_cols:
            raise IndexError(f"Index ({i},{j}) out of bounds")
        start = self.colptr[j]
        end = self.colptr[j + 1]
        pos = np.searchsorted(self.rowind[start:end], i)
        if pos < (end - start) and self.rowind[start + pos] == i:
            return self.values[start + pos]
        return 0.0

    def set(self, i: int, j: int, value: float):
        """设置元素 A[i,j] = value (简化版: 转为稠密再转回)"""
        dense = self.to_dense()
        dense[i, j] = value
        new_sparse = SparseCCS.from_dense(dense)
        self.__dict__.update(new_sparse.__dict__)

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A * x。

        算法:
            y_i = sum_{j: A_{ij} != 0} A_{ij} * x_j
        """
        if x.shape[0] != self.n_cols:
            raise ValueError(f"Dimension mismatch: A is {self.n_rows}x{self.n_cols}, x has {x.shape[0]}")

        y = np.zeros(self.n_rows, dtype=np.float64)
        for j in range(self.n_cols):
            xj = x[j]
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                y[i] += self.values[idx] * xj

        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵转置-向量乘法 y = A^T * x。
        """
        if x.shape[0] != self.n_rows:
            raise ValueError(f"Dimension mismatch")

        y = np.zeros(self.n_cols, dtype=np.float64)
        for j in range(self.n_cols):
            dot = 0.0
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                dot += self.values[idx] * x[i]
            y[j] = dot

        return y

    @classmethod
    def network_laplacian(cls, adj: np.ndarray) -> 'SparseCCS':
        """
        从邻接矩阵构造图拉普拉斯矩阵 L = D - A。

        其中 D 为度对角矩阵，D_{ii} = sum_j A_{ij}。
        """
        n = adj.shape[0]
        degrees = np.sum(adj, axis=1)

        colptr = [0]
        rowind = []
        values = []

        for j in range(n):
            col_nonzeros = []
            for i in range(n):
                val = -adj[i, j]
                if i == j:
                    val = degrees[i] - adj[i, j]
                if abs(val) > 1e-15:
                    col_nonzeros.append((i, val))
            col_nonzeros.sort(key=lambda x: x[0])
            for i, v in col_nonzeros:
                rowind.append(i)
                values.append(v)
            colptr.append(len(values))

        return cls(n, n,
                   np.array(colptr, dtype=np.int32),
                   np.array(rowind, dtype=np.int32),
                   np.array(values, dtype=np.float64))

    def power_iteration_sparse(self, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        对稀疏矩阵执行幂迭代，计算主特征值和特征向量。
        """
        n = self.n_cols
        x = np.random.rand(n)
        x /= np.linalg.norm(x)

        lambda_old = 0.0

        for it in range(max_iter):
            y = self.mv(x)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-15:
                break
            x_new = y / norm_y

            # Rayleigh商
            lambda_new = float(x_new @ self.mv(x_new))

            diff = abs(lambda_new - lambda_old)
            cos_angle = float(x @ x_new)
            sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))

            x = x_new
            lambda_old = lambda_new

            if diff < tol and sin_angle < tol:
                break

        return lambda_old, x
