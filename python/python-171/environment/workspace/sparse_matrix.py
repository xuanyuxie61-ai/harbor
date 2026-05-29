# -*- coding: utf-8 -*-
"""
sparse_matrix.py
================
多种稀疏矩阵存储格式与矩阵-向量乘法。

融合种子项目：
- 149_cg : R8GE, R83, R83S, R83T, R8PBU, R8SD, R8SP 等格式
"""

import numpy as np


# ---------------------------------------------------------------------------
# R8GE : 一般稠密矩阵
# ---------------------------------------------------------------------------

def r8ge_mv(A, x):
    """一般稠密矩阵乘向量：y = A @ x"""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    return A @ x


# ---------------------------------------------------------------------------
# R83 : 三对角矩阵（竖直折叠存储，shape (3, n)）
# ---------------------------------------------------------------------------

def r83_mv(a, x):
    """
    R83 三对角矩阵乘向量。
    a[0, 1:n]   = 上对角线
    a[1, 0:n]   = 主对角线
    a[2, 0:n-1] = 下对角线
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    if a.shape != (3, n):
        raise ValueError(f"R83 matrix shape must be (3, {n}), got {a.shape}")
    b = np.zeros(n, dtype=float)
    if n == 1:
        b[0] = a[1, 0] * x[0]
        return b
    b[0] = a[1, 0] * x[0] + a[0, 1] * x[1]
    for i in range(1, n - 1):
        b[i] = a[2, i - 1] * x[i - 1] + a[1, i] * x[i] + a[0, i + 1] * x[i + 1]
    b[n - 1] = a[2, n - 2] * x[n - 2] + a[1, n - 1] * x[n - 1]
    return b


# ---------------------------------------------------------------------------
# R83S : 三对角标量矩阵（每行相同，shape (3,)）
# ---------------------------------------------------------------------------

def r83s_mv(a, x):
    """
    R83S 三对角标量矩阵乘向量。
    a = [subdiag, diag, superdiag]，每行相同。
    """
    a = np.asarray(a, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    if a.size != 3:
        raise ValueError("R83S matrix must be length-3 vector.")
    b = np.zeros(n, dtype=float)
    for i in range(1, n):
        b[i] += a[0] * x[i - 1]
    for i in range(n):
        b[i] += a[1] * x[i]
    for i in range(n - 1):
        b[i] += a[2] * x[i + 1]
    return b


# ---------------------------------------------------------------------------
# R83T : 三对角矩阵（水平折叠存储，shape (n, 3)）
# ---------------------------------------------------------------------------

def r83t_mv(a, x):
    """
    R83T 三对角矩阵乘向量。
    a[0:n, 0] = 下对角线（a[0,0] 未使用）
    a[0:n, 1] = 主对角线
    a[0:n, 2] = 上对角线（a[n-1,2] 未使用）
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    if a.shape != (n, 3):
        raise ValueError(f"R83T matrix shape must be ({n}, 3), got {a.shape}")
    b = np.zeros(n, dtype=float)
    if n == 1:
        b[0] = a[0, 1] * x[0]
        return b
    b[0] = a[0, 1] * x[0] + a[0, 2] * x[1]
    for i in range(1, n - 1):
        b[i] = a[i, 0] * x[i - 1] + a[i, 1] * x[i] + a[i, 2] * x[i + 1]
    b[n - 1] = a[n - 1, 0] * x[n - 2] + a[n - 1, 1] * x[n - 1]
    return b


# ---------------------------------------------------------------------------
# R8PBU : 对称正定带状矩阵（shape (mu+1, n)）
# ---------------------------------------------------------------------------

def r8pbu_mv(mu, a, x):
    """
    对称带状矩阵乘向量。
    a[mu, :]     = 主对角线
    a[mu-1, 1:]  = 第一条上对角线
    ...
    a[0, mu:]    = 第 mu 条上对角线
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    if a.shape != (mu + 1, n):
        raise ValueError(f"R8PBU shape mismatch: expected ({mu+1}, {n}), got {a.shape}")
    b = np.zeros(n, dtype=float)
    for i in range(n):
        b[i] = a[mu, i] * x[i]
    for k in range(1, mu + 1):
        for j in range(mu + 1 - k, n):
            ieqn = k + j - mu - 1
            val = a[mu - k, j]
            b[ieqn] += val * x[j]
            b[j] += val * x[ieqn]
    return b


# ---------------------------------------------------------------------------
# R8SD : 对称对角存储矩阵（shape (n, ndiag)）
# ---------------------------------------------------------------------------

def r8sd_mv(offset, a, x):
    """
    对称对角存储矩阵乘向量。
    offset[j] >= 0 为第 j 条存储对角线到主对角线的偏移量。
    a[i, j] 对应 A[i, i+offset[j]] 的非零元。
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    ndiag = len(offset)
    if a.shape != (n, ndiag):
        raise ValueError(f"R8SD shape mismatch: expected ({n}, {ndiag}), got {a.shape}")
    b = np.zeros(n, dtype=float)
    for i in range(n):
        for jd in range(ndiag):
            off = offset[jd]
            if off < 0:
                continue
            j = i + off
            if 0 <= j < n:
                b[i] += a[i, jd] * x[j]
                if off != 0:
                    b[j] += a[i, jd] * x[i]
    return b


# ---------------------------------------------------------------------------
# R8SP : 稀疏三元组（COO）格式
# ---------------------------------------------------------------------------

def r8sp_mv(row, col, a_val, x):
    """
    COO 格式稀疏矩阵乘向量。
    row[k], col[k], a_val[k] 为第 k 个非零元。
    """
    x = np.asarray(x, dtype=float).flatten()
    n = x.size
    nz = len(a_val)
    b = np.zeros(n, dtype=float)
    for k in range(nz):
        i = row[k]
        j = col[k]
        if 0 <= i < n and 0 <= j < n:
            b[i] += a_val[k] * x[j]
    return b


# ---------------------------------------------------------------------------
# 矩阵格式转换与工厂函数
# ---------------------------------------------------------------------------

def dif2_r8ge(n):
    """构造 n×n DIF2 矩阵（一维离散 Laplacian）为稠密格式。"""
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    return A


def dif2_r83(n):
    """DIF2 的 R83 格式。"""
    a = np.zeros((3, n), dtype=float)
    a[1, :] = 2.0
    if n > 1:
        a[0, 1:] = -1.0
        a[2, :-1] = -1.0
    return a


def dif2_r83s():
    """DIF2 的 R83S 格式（标量三对角）。"""
    return np.array([-1.0, 2.0, -1.0], dtype=float)


def dif2_r83t(n):
    """DIF2 的 R83T 格式。"""
    a = np.zeros((n, 3), dtype=float)
    if n > 1:
        a[1:, 0] = -1.0
    a[:, 1] = 2.0
    if n > 1:
        a[:-1, 2] = -1.0
    return a


def dif2_r8pbu(n, mu=1):
    """DIF2 的 R8PBU 格式。"""
    a = np.zeros((mu + 1, n), dtype=float)
    if mu >= 1:
        a[mu - 1, 1:] = -1.0
    a[mu, :] = 2.0
    return a


def dif2_r8sd(n, ndiag=2):
    """DIF2 的 R8SD 格式。"""
    if ndiag < 2:
        raise ValueError("ndiag must be >= 2 for DIF2 in R8SD.")
    a = np.zeros((n, ndiag), dtype=float)
    a[:, 0] = 2.0
    a[:-1, 1] = -1.0
    offset = [0, 1] + [0] * (ndiag - 2)
    return a, offset


def dif2_r8sp(n):
    """DIF2 的 COO 格式。"""
    row = []
    col = []
    val = []
    for i in range(n):
        if i > 0:
            row.append(i)
            col.append(i - 1)
            val.append(-1.0)
        row.append(i)
        col.append(i)
        val.append(2.0)
        if i < n - 1:
            row.append(i)
            col.append(i + 1)
            val.append(-1.0)
    return np.array(row, dtype=int), np.array(col, dtype=int), np.array(val, dtype=float)


# ---------------------------------------------------------------------------
# 统一 matvec 接口
# ---------------------------------------------------------------------------

class SparseMatrixOperator:
    """
    封装多种稀疏格式的矩阵-向量乘法，提供统一接口 matvec(x)。
    """

    def __init__(self, fmt, data, n, extra=None):
        """
        fmt: 'ge', 'r83', 'r83s', 'r83t', 'pbu', 'sd', 'sp'
        data: 格式相关的矩阵数据
        n: 矩阵阶数
        extra: 额外参数（如 mu, offset 等）
        """
        self.fmt = fmt
        self.data = data
        self.n = n
        self.extra = extra if extra is not None else {}

    def matvec(self, x):
        x = np.asarray(x, dtype=float).flatten()
        if self.fmt == 'ge':
            return r8ge_mv(self.data, x)
        elif self.fmt == 'r83':
            return r83_mv(self.data, x)
        elif self.fmt == 'r83s':
            return r83s_mv(self.data, x)
        elif self.fmt == 'r83t':
            return r83t_mv(self.data, x)
        elif self.fmt == 'pbu':
            return r8pbu_mv(self.extra['mu'], self.data, x)
        elif self.fmt == 'sd':
            return r8sd_mv(self.extra['offset'], self.data, x)
        elif self.fmt == 'sp':
            return r8sp_mv(self.extra['row'], self.extra['col'], self.data, x)
        else:
            raise ValueError(f"Unknown format: {self.fmt}")

    def to_dense(self):
        """转换为稠密 numpy 数组（仅用于小规模验证）。"""
        A = np.zeros((self.n, self.n), dtype=float)
        for j in range(self.n):
            e = np.zeros(self.n, dtype=float)
            e[j] = 1.0
            A[:, j] = self.matvec(e)
        return A
