"""
sparse_linalg.py
稀疏矩阵线性代数运算库
基于 r8st (Sparse Triplet/COO) 与 r8po (SPD packed) 核心算法重构

声学工程应用：
- 有限元离散 Helmholtz 方程产生的大型稀疏对称矩阵求解
- 共轭梯度法 (CG) 求解稀疏线性系统
- Cholesky 分解用于模态分析中的广义特征值问题预处理
"""

import numpy as np


class SparseCOO:
    """
    COO (Coordinate) 格式稀疏矩阵。
    基于 r8st 格式思想。
    """
    def __init__(self, rows, cols, vals, shape):
        self.rows = np.asarray(rows, dtype=int)
        self.cols = np.asarray(cols, dtype=int)
        self.vals = np.asarray(vals, dtype=float)
        self.shape = shape
        self.n = shape[0]
        self.m = shape[1]
        self.nnz = len(vals)

    def mv(self, x):
        """
        稀疏矩阵-向量乘法 y = A @ x。
        基于 r8st_mv 算法。
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.n, dtype=float)
        for i in range(self.nnz):
            y[self.rows[i]] += self.vals[i] * x[self.cols[i]]
        return y

    def mtv(self, x):
        """
        稀疏转置矩阵-向量乘法 y = A.T @ x。
        基于 r8st_mtv 算法。
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for i in range(self.nnz):
            y[self.cols[i]] += self.vals[i] * x[self.rows[i]]
        return y

    def to_dense(self):
        """
        转为稠密矩阵（仅用于小规模调试）。
        """
        A = np.zeros(self.shape, dtype=float)
        for i in range(self.nnz):
            A[self.rows[i], self.cols[i]] += self.vals[i]
        return A

    def residual(self, x, b):
        """
        计算残差 r = b - A @ x。
        基于 r8st_res 算法。
        """
        return b - self.mv(x)


def conjugate_gradient(A_sparse, b, x0=None, tol=1e-10, max_iter=None):
    """
    共轭梯度法求解 A x = b（A 对称正定）。
    基于 r8st_cg 核心算法。

    算法推导:
    给定 SPD 矩阵 A，CG 迭代通过共轭方向最小化二次泛函:
        Φ(x) = 0.5 * x^T A x - b^T x
    等价于最小化能量范数误差: ||x - x*||_A

    迭代格式:
        r_k = b - A x_k
        β_k = (r_k^T r_k) / (r_{k-1}^T r_{k-1})
        p_k = r_k + β_k p_{k-1}
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
    """
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    if max_iter is None:
        max_iter = n

    r = b - A_sparse.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = A_sparse.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def jacobi_iteration(A_sparse, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Jacobi 迭代法求解 A x = b。
    基于 r8st_jac_sl 核心算法。

    迭代格式:
        x_i^{(k+1)} = (b_i - Σ_{j≠i} A_{ij} x_j^{(k)}) / A_{ii}
    """
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    # 提取对角元
    diag = np.zeros(n, dtype=float)
    for i in range(A_sparse.nnz):
        if A_sparse.rows[i] == A_sparse.cols[i]:
            diag[A_sparse.rows[i]] += A_sparse.vals[i]
    diag = np.where(np.abs(diag) < 1e-14, 1.0, diag)

    for _ in range(max_iter):
        x_new = b.copy()
        for i in range(A_sparse.nnz):
            if A_sparse.rows[i] != A_sparse.cols[i]:
                x_new[A_sparse.rows[i]] -= A_sparse.vals[i] * x[A_sparse.cols[i]]
        x_new /= diag
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def r8po_fa(n, a):
    """
    Cholesky 分解：A = R^T * R，R 为上三角矩阵。
    基于 r8po_fa 核心算法，用于 SPD 矩阵。

    在声学模态分析中，质量矩阵 M 是 SPD 的，
    Cholesky 分解可用于广义特征值问题的变换:
        K φ = λ M φ
    令 y = R φ，则化为标准特征值问题:
        R^{-T} K R^{-1} y = λ y
    """
    a = np.asarray(a, dtype=float).copy()
    r = np.zeros((n, n), dtype=float)
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += r[k, j] ** 2
        # 检查正定性
        if a[j, j] - s <= 0.0:
            raise ValueError(f"Matrix is not positive definite at row {j}")
        r[j, j] = np.sqrt(a[j, j] - s)
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += r[k, i] * r[k, j]
            r[j, i] = (a[j, i] - s) / r[j, j]
    return r


def r8po_sl(n, r, b):
    """
    使用 Cholesky 因子 R 求解 A x = b。
    A = R^T R，先解 R^T y = b，再解 R x = y。
    基于 r8po_sl 核心算法（前向/后向替换）。
    """
    b = np.asarray(b, dtype=float).copy()
    x = b.copy()
    # 前向替换: R^T y = b
    for j in range(n):
        x[j] /= r[j, j]
        for i in range(j + 1, n):
            x[i] -= r[j, i] * x[j]
    # 后向替换: R x = y
    for j in range(n - 1, -1, -1):
        x[j] /= r[j, j]
        for i in range(j):
            x[i] -= r[j, i] * x[j]
    return x


def r8po_det(n, r):
    """
    通过 Cholesky 因子计算行列式: det(A) = (Π R_{ii})²
    基于 r8po_det 核心算法。
    """
    det_r = np.prod(np.diag(r))
    return det_r ** 2


def r8po_inverse(n, r):
    """
    通过 Cholesky 因子计算逆矩阵。
    基于 r8po_inverse 核心算法。
    """
    # 先求 R 的逆
    r_inv = np.zeros((n, n), dtype=float)
    for i in range(n):
        r_inv[i, i] = 1.0 / r[i, i]
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i, j):
                s += r[k, j] * r_inv[i, k]
            r_inv[i, j] = -s / r[j, j]
    # A^{-1} = R^{-1} * R^{-T}
    return r_inv @ r_inv.T


def assemble_sparse_from_triplets(rows, cols, vals, n):
    """
    基于 pagerank2 的稀疏矩阵构造思想：
    将 (row, col, val) 三元组列表聚合成稀疏矩阵。
    在 FEM 组装中，多个单元对同一自由度对的刚度贡献需要累加。
    """
    # 使用字典累加相同位置的值
    accum = {}
    for i, j, v in zip(rows, cols, vals):
        key = (int(i), int(j))
        accum[key] = accum.get(key, 0.0) + v
    rows_out = np.array([k[0] for k in accum.keys()], dtype=int)
    cols_out = np.array([k[1] for k in accum.keys()], dtype=int)
    vals_out = np.array(list(accum.values()), dtype=float)
    return SparseCOO(rows_out, cols_out, vals_out, (n, n))
