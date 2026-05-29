"""
sparse_linear_algebra.py
================================================================================
稀疏矩阵线性代数模块（CRS 格式）

本模块融合以下种子项目的核心算法：
  - 978_r8crs : 压缩行存储（CRS）稀疏矩阵的构造、矩阵-向量乘法、
                格式转换、示例矩阵生成

科学背景
--------
在最优控制伴随方程方法中，PDE 的有限元离散产生大规模稀疏线性系统。
对于 N 个自由度的 FEM 离散，质量矩阵 M 与刚度矩阵 A 均为稀疏矩阵，
非零元数量约为 O(N)。使用稠密存储将导致 O(N²) 的内存开销与 O(N³)
的求解复杂度，完全不可行。CRS（Compressed Row Storage）格式仅存储
非零元的列索引与数值，是科学计算中最主流的稀疏矩阵格式之一。

CRS 格式定义
------------
对于一个 m × n 的稀疏矩阵 A，记 nnz 为非零元总数：
  - val[0:nnz]     : 非零元的数值，按行优先顺序排列
  - col[0:nnz]     : 每个非零元所在的列索引
  - rowptr[0:m+1]  : rowptr[i] 表示第 i 行的第一个非零元在 val/col 中的位置
                     rowptr[m] = nnz

核心运算复杂度
--------------
  - SpMV (y = A x)      : O(nnz)
  - SpMTV (y = A^T x)   : O(nnz)
  - 稠密↔稀疏转换        : O(m·n)
"""

import numpy as np


class CRSMatrix:
    """
    压缩行存储（CRS）稀疏矩阵类。
    支持矩阵-向量乘法、转置乘法、与稠密矩阵互转。
    """

    def __init__(self, m, n, nnz=0):
        self.m = m
        self.n = n
        self.nnz = nnz
        self.val = np.zeros(nnz, dtype=float)
        self.col = np.zeros(nnz, dtype=int)
        self.rowptr = np.zeros(m + 1, dtype=int)

    def matvec(self, x):
        """
        稀疏矩阵-向量乘法 y = A @ x。
        对 x 的维度做边界检查。
        """
        if x.shape[0] != self.n:
            raise ValueError(f"matvec: 向量维度 {x.shape[0]} 不匹配矩阵列数 {self.n}")
        y = np.zeros(self.m, dtype=float)
        for i in range(self.m):
            s = 0.0
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                s += self.val[idx] * x[self.col[idx]]
            y[i] = s
        return y

    def matvec_transpose(self, x):
        """
        稀疏转置矩阵-向量乘法 y = A^T @ x。
        """
        if x.shape[0] != self.m:
            raise ValueError(f"matvec_transpose: 向量维度 {x.shape[0]} 不匹配矩阵行数 {self.m}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.m):
            xi = x[i]
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                y[self.col[idx]] += self.val[idx] * xi
        return y

    def to_dense(self):
        """将 CRS 矩阵转换为稠密 numpy 数组。"""
        A = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.m):
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                A[i, self.col[idx]] = self.val[idx]
        return A

    @staticmethod
    def from_dense(dense):
        """从稠密 numpy 数组构造 CRS 矩阵。"""
        m, n = dense.shape
        rows, cols = np.nonzero(np.abs(dense) > 1.0e-15)
        nnz = len(rows)
        val = dense[rows, cols].astype(float)
        col = cols.astype(int)
        rowptr = np.zeros(m + 1, dtype=int)
        for i in range(m):
            rowptr[i + 1] = rowptr[i] + np.count_nonzero(rows == i)
        # 现在需要按行重新排序 val 和 col
        # 由于 np.nonzero 是按行优先返回的，所以 rowptr 可以直接用累积计数
        # 但为了安全，还是重新构建
        rowptr = np.zeros(m + 1, dtype=int)
        for r in rows:
            rowptr[r + 1] += 1
        rowptr = np.cumsum(rowptr)
        obj = CRSMatrix(m, n, nnz)
        obj.val = val
        obj.col = col
        obj.rowptr = rowptr
        return obj

    def copy(self):
        """深拷贝。"""
        obj = CRSMatrix(self.m, self.n, self.nnz)
        obj.val = self.val.copy()
        obj.col = self.col.copy()
        obj.rowptr = self.rowptr.copy()
        return obj


def build_sparse_dif2(n):
    """
    构建一维二阶差分稀疏矩阵（Tridiagonal with 2 on diag, -1 on off-diags）。
    这是经典的有限差分离散矩阵，对应算子 -d²/dx² 的离散。
    融合 978_r8crs 的 r8crs_dif2 思想。
    """
    nnz = 3 * n - 2
    val = np.zeros(nnz, dtype=float)
    col = np.zeros(nnz, dtype=int)
    rowptr = np.zeros(n + 1, dtype=int)
    idx = 0
    for i in range(n):
        rowptr[i] = idx
        if i > 0:
            val[idx] = -1.0
            col[idx] = i - 1
            idx += 1
        val[idx] = 2.0
        col[idx] = i
        idx += 1
        if i < n - 1:
            val[idx] = -1.0
            col[idx] = i + 1
            idx += 1
    rowptr[n] = idx
    A = CRSMatrix(n, n, idx)
    A.val = val
    A.col = col
    A.rowptr = rowptr
    return A


def sparse_solve_cg(A, b, x0=None, tol=1.0e-10, max_iter=1000):
    """
    共轭梯度法（CG）求解对称正定稀疏线性系统 A x = b。
    这是大规模 FEM 线性系统中最主流的迭代求解器之一。

    算法（Krylov 子空间方法）：
    给定初值 x₀，计算 r₀ = b - A x₀，p₀ = r₀
    对于 k = 0, 1, 2, ...:
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k
    """
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A.matvec(x)
    p = r.copy()
    rsold = np.dot(r, r)

    if np.sqrt(rsold) < tol:
        return x

    for _ in range(max_iter):
        Ap = A.matvec(p)
        alpha = rsold / (np.dot(p, Ap) + 1.0e-30)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def sparse_solve_jacobi(A, b, x0=None, tol=1.0e-10, max_iter=2000):
    """
    Jacobi 迭代法求解 A x = b。
    对 A 为严格对角占优时收敛。用于边界处理和数值鲁棒性验证。
    """
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    # 提取对角线
    D = np.zeros(n, dtype=float)
    for i in range(n):
        for idx in range(A.rowptr[i], A.rowptr[i + 1]):
            if A.col[idx] == i:
                D[i] = A.val[idx]
                break
        if abs(D[i]) < 1.0e-15:
            raise ValueError("Jacobi: 对角线元素为零，无法迭代。")

    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=float)
        for i in range(n):
            s = 0.0
            for idx in range(A.rowptr[i], A.rowptr[i + 1]):
                j = A.col[idx]
                if j != i:
                    s += A.val[idx] * x[j]
            x_new[i] = (b[i] - s) / D[i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x
