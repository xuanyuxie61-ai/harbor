"""
sparse_matrix.py
================
稀疏矩阵工具库，用于大规模分子动力学系统中哈密顿量矩阵的
稀疏存储与格式转换。

核心数学内容：
  - General (GE) 格式到 Sparse Triplet (ST) 格式的转换
  - Compressed Sparse Row (CSR) 格式支持
  - 稀疏矩阵-向量乘法，用于求解 Poisson-Boltzmann 方程离散化后的线性系统

种子项目映射：
  - 459_ge_to_st  →  GE 到 ST 格式转换
"""

import numpy as np
from typing import Tuple, List


class SparseMatrix:
    """
    稀疏矩阵容器，支持 GE / ST / CSR 三种格式互转。
    """

    def __init__(self, m: int, n: int):
        if m <= 0 or n <= 0:
            raise ValueError("SparseMatrix: dimensions must be positive.")
        self.m = m
        self.n = n
        self._st_ist: np.ndarray = np.array([], dtype=int)
        self._st_jst: np.ndarray = np.array([], dtype=int)
        self._st_ast: np.ndarray = np.array([], dtype=float)
        self._csr_data: np.ndarray = np.array([], dtype=float)
        self._csr_indices: np.ndarray = np.array([], dtype=int)
        self._csr_indptr: np.ndarray = np.array([], dtype=int)

    # -----------------------------------------------------------------------
    # GE -> ST 转换（种子项目 459_ge_to_st 核心算法）
    # -----------------------------------------------------------------------
    def from_dense(self, Age: np.ndarray, drop_tol: float = 0.0) -> "SparseMatrix":
        """
        将稠密矩阵 Age 转换为 ST 稀疏格式。

        参数边界：
            Age      : shape (m, n) 的二维 ndarray
            drop_tol : 绝对值小于此阈值的元素视为零
        """
        Age = np.asarray(Age, dtype=float)
        if Age.ndim != 2:
            raise ValueError("from_dense: Age must be a 2D array.")
        m, n = Age.shape
        if m != self.m or n != self.n:
            raise ValueError("from_dense: Age shape does not match declared dimensions.")

        mask = np.abs(Age) > drop_tol
        nz_num = int(np.count_nonzero(mask))

        ist = np.zeros(nz_num, dtype=int)
        jst = np.zeros(nz_num, dtype=int)
        Ast = np.zeros(nz_num, dtype=float)

        k = 0
        for j in range(n):
            for i in range(m):
                if mask[i, j]:
                    ist[k] = i
                    jst[k] = j
                    Ast[k] = Age[i, j]
                    k += 1

        self._st_ist = ist
        self._st_jst = jst
        self._st_ast = Ast
        return self

    def to_dense(self) -> np.ndarray:
        """将当前 ST 格式还原为稠密矩阵。"""
        A = np.zeros((self.m, self.n), dtype=float)
        for i, j, v in zip(self._st_ist, self._st_jst, self._st_ast):
            A[i, j] = v
        return A

    # -----------------------------------------------------------------------
    # ST -> CSR 转换（工程扩展，用于高效 SpMV）
    # -----------------------------------------------------------------------
    def to_csr(self) -> "SparseMatrix":
        """
        将 ST 格式转换为 CSR (Compressed Sparse Row) 格式。
        CSR 格式下，稀疏矩阵-向量乘法的时间复杂度为 O(nnz)。
        """
        if self._st_ist.size == 0:
            self._csr_data = np.array([], dtype=float)
            self._csr_indices = np.array([], dtype=int)
            self._csr_indptr = np.zeros(self.m + 1, dtype=int)
            return self

        # 按行优先排序
        order = np.lexsort((self._st_jst, self._st_ist))
        ist = self._st_ist[order]
        jst = self._st_jst[order]
        ast = self._st_ast[order]

        # 合并同一位置的重复项（求和）
        uniq_keys = []
        uniq_vals = []
        prev = (-1, -1)
        cur_sum = 0.0
        for idx in range(ist.size):
            key = (int(ist[idx]), int(jst[idx]))
            val = float(ast[idx])
            if key == prev:
                cur_sum += val
            else:
                if prev != (-1, -1):
                    uniq_keys.append(prev)
                    uniq_vals.append(cur_sum)
                prev = key
                cur_sum = val
        if prev != (-1, -1):
            uniq_keys.append(prev)
            uniq_vals.append(cur_sum)

        if not uniq_keys:
            self._csr_data = np.array([], dtype=float)
            self._csr_indices = np.array([], dtype=int)
            self._csr_indptr = np.zeros(self.m + 1, dtype=int)
            return self

        ist_u = np.array([k[0] for k in uniq_keys], dtype=int)
        jst_u = np.array([k[1] for k in uniq_keys], dtype=int)
        ast_u = np.array(uniq_vals, dtype=float)

        nnz = ist_u.size
        data = ast_u
        indices = jst_u
        indptr = np.zeros(self.m + 1, dtype=int)

        row_counts = np.bincount(ist_u, minlength=self.m)
        indptr[1:] = np.cumsum(row_counts)

        self._csr_data = data
        self._csr_indices = indices
        self._csr_indptr = indptr
        return self

    def spmv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A * x。

        参数边界：
            x 的长度必须等于矩阵列数 n。
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n:
            raise ValueError("spmv: x length must equal matrix column count.")

        if self._csr_indptr.size == 0:
            self.to_csr()

        y = np.zeros(self.m, dtype=float)
        for i in range(self.m):
            row_start = self._csr_indptr[i]
            row_end = self._csr_indptr[i + 1]
            for idx in range(row_start, row_end):
                j = self._csr_indices[idx]
                y[i] += self._csr_data[idx] * x[j]
        return y

    @property
    def nnz(self) -> int:
        """非零元素个数。"""
        return self._st_ist.size

    def get_st(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (ist, jst, Ast) 三元组。"""
        return self._st_ist.copy(), self._st_jst.copy(), self._st_ast.copy()


def spdiags(diags: np.ndarray, offsets: List[int], m: int, n: int) -> SparseMatrix:
    """
    从对角线构造稀疏矩阵（MATLAB spdiags 的简化实现）。

    参数：
        diags   : shape (len(offsets), max(m,n)) 的数组，每行是一条对角线
        offsets : 每条对角线相对于主对角线的偏移量
        m, n    : 矩阵维度
    """
    if diags.ndim != 2:
        raise ValueError("spdiags: diags must be 2D.")
    if len(offsets) != diags.shape[0]:
        raise ValueError("spdiags: number of offsets must match number of diagonals.")

    S = SparseMatrix(m, n)
    ist_list: List[int] = []
    jst_list: List[int] = []
    val_list: List[float] = []

    for d_idx, offset in enumerate(offsets):
        if offset >= 0:
            i_start = 0
            j_start = offset
            length = min(m, n - offset)
        else:
            i_start = -offset
            j_start = 0
            length = min(m + offset, n)

        for k in range(length):
            i = i_start + k
            j = j_start + k
            v = diags[d_idx, k]
            if abs(v) > 0.0:
                ist_list.append(i)
                jst_list.append(j)
                val_list.append(v)

    S._st_ist = np.array(ist_list, dtype=int)
    S._st_jst = np.array(jst_list, dtype=int)
    S._st_ast = np.array(val_list, dtype=float)
    return S
