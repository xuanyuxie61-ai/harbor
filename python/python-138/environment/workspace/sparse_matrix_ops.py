"""
微反应器有限元模型的稀疏 skyline 矩阵运算 (基于 R8SS 存储格式)
===============================================================
在微反应器多物理场模拟中，有限元或有限体积离散产生的大型稀疏对称矩阵
适合使用 skyline (skyline/symmetric sparse, R8SS) 格式存储，以节省内存
并加速矩阵-向量乘法。

R8SS 存储格式：
    对于 N×N 对称矩阵 A，仅存储每列从第一个非零元到对角线的元素。
    DIAG(j) 记录第 j 列对角元在紧凑数组 A 中的索引。
    第 j 列的非零元为 A(DIAG(j-1)+1 : DIAG(j))，
    对应原矩阵行范围 (j - (DIAG(j)-DIAG(j-1)) + 1) 到 j。

矩阵-向量乘法：
    y = A x

    for j = 1 .. N:
        ilo = j + 1 + DIAGOLD - DIAG(j)
        for i = ilo .. j-1:
            y(i) += A(k) * x(j)
            y(j) += A(k) * x(i)
            k += 1
        y(j) += A(k) * x(j)
        k += 1
        DIAGOLD = DIAG(j)

应用：
    微反应器传热有限元方程 K T = F，其中 K 为对称正定的 skyline 刚度矩阵。
"""

import numpy as np
from typing import Tuple, Optional


class SkylineMatrixOperator:
    """
    对称稀疏 skyline 矩阵操作器。
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("矩阵阶数必须至少为 1")
        self.n = n
        self.diag = None
        self.a = None
        self.na = 0

    def build_from_tridiagonal(
        self, lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray
    ):
        """
        从三对角矩阵构建 skyline 存储。
        三对角矩阵的 skyline 每列高度最多为 2。
        """
        n = self.n
        if len(diagonal) != n:
            raise ValueError("对角线长度不匹配")
        # 计算 diag 索引
        diag_idx = np.zeros(n, dtype=int)
        count = 0
        for j in range(n):
            # 三对角：每列最多 2 个非对角元 + 1 个对角元
            height = 1
            if j > 0 and lower[j] != 0.0:
                height = 2
            count += height
            diag_idx[j] = count - 1
        self.diag = diag_idx
        self.na = count
        self.a = np.zeros(count)
        k = 0
        for j in range(n):
            if j > 0 and lower[j] != 0.0:
                self.a[k] = lower[j]
                k += 1
            self.a[k] = diagonal[j]
            k += 1

    def build_from_dense(self, A: np.ndarray, tol: float = 1.0e-14):
        """
        从稠密对称矩阵构建 skyline 存储。
        """
        if A.shape != (self.n, self.n):
            raise ValueError("矩阵维度不匹配")
        n = self.n
        diag_idx = np.zeros(n, dtype=int)
        count = 0
        for j in range(n):
            # 找该列第一个非零元行号
            rows = np.where(np.abs(A[:, j]) > tol)[0]
            if len(rows) == 0:
                first_row = j
            else:
                first_row = min(rows)
            first_row = min(first_row, j)
            height = j - first_row + 1
            count += height
            diag_idx[j] = count - 1
        self.diag = diag_idx
        self.na = count
        self.a = np.zeros(count)
        k = 0
        for j in range(n):
            first_row = j - (diag_idx[j] - (diag_idx[j - 1] if j > 0 else -1)) + 1
            for i in range(first_row, j + 1):
                self.a[k] = A[i, j]
                k += 1

    def multiply(self, x: np.ndarray) -> np.ndarray:
        """
        计算 y = A x，使用 R8SS 格式。
        """
        if len(x) != self.n:
            raise ValueError("向量维度不匹配")
        if self.a is None or self.diag is None:
            raise ValueError("矩阵未初始化")

        n = self.n
        y = np.zeros(n)
        diagold = -1
        k = 0
        for j in range(n):
            height = self.diag[j] - diagold
            ilo = j + 1 - height + 1
            for i in range(ilo, j):
                y[i] += self.a[k] * x[j]
                y[j] += self.a[k] * x[i]
                k += 1
            y[j] += self.a[k] * x[j]
            k += 1
            diagold = self.diag[j]
        return y

    def to_dense(self) -> np.ndarray:
        """将 skyline 矩阵展开为稠密矩阵（用于验证）。"""
        n = self.n
        A = np.zeros((n, n))
        if self.a is None:
            return A
        diagold = -1
        k = 0
        for j in range(n):
            height = self.diag[j] - diagold
            ilo = j + 1 - height + 1
            for i in range(ilo, j):
                A[i, j] = self.a[k]
                A[j, i] = self.a[k]
                k += 1
            A[j, j] = self.a[k]
            k += 1
            diagold = self.diag[j]
        return A

    def solve_cholesky_skyline(
        self, b: np.ndarray
    ) -> np.ndarray:
        """
        对 skyline 对称正定矩阵执行 Cholesky 分解并求解。
        简化版：先展开为稠密矩阵再求解（用于小规模验证）。
        大规模时可实现 skyline Cholesky。
        """
        A_dense = self.to_dense()
        try:
            L = np.linalg.cholesky(A_dense)
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x
        except np.linalg.LinAlgError:
            # 回退到 LU
            return np.linalg.solve(A_dense, b)

    def condition_number_estimate(self) -> float:
        """估计条件数（使用稠密展开）。"""
        A_dense = self.to_dense()
        s = np.linalg.svd(A_dense, compute_uv=False)
        if s[-1] < 1.0e-15:
            return float("inf")
        return s[0] / s[-1]
