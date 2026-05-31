import math
"""
sparse_linear_algebra.py
================================================================================
高性能计算检查点容错：稀疏线性代数与迭代求解器

融合原项目：
  - 965_r83s (三对角标量矩阵存储与迭代求解器)
  - 025_asa006 (Cholesky 分解)

科学角色：
  1) 在检查点重启后的状态恢复阶段，需要快速求解大型稀疏线性系统；
  2) Cholesky 分解用于协方差矩阵与预处理子；
  3) CG/Jacobi/Gauss-Seidel 用于恢复 PDE 状态的时间步隐式系统。
================================================================================
"""

import numpy as np


# =============================================================================
# R83S : 三对角 Toeplitz 标量矩阵 [sub, diag, super]
# =============================================================================
class R83SMatrix:
    """紧凑存储三对角标量矩阵 A = sub * I_{-1} + diag * I_0 + super * I_{+1}。"""

    def __init__(self, m: int, n: int, a: np.ndarray):
        self.m = m
        self.n = n
        self.a = np.asarray(a, dtype=float)  # shape (3,)
        if self.a.shape != (3,):
            raise ValueError("a must have shape (3,)")

    @staticmethod
    def dif2(n: int):
        """经典 DIF2 矩阵: [-1, 2, -1]。"""
        return R83SMatrix(n, n, np.array([-1.0, 2.0, -1.0]))

    def mv(self, x: np.ndarray) -> np.ndarray:
        """矩阵-向量乘积 y = A @ x。"""
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        if self.n != len(x):
            raise ValueError("Dimension mismatch")
        sub, diag, sup = self.a
        for i in range(self.m):
            if i > 0:
                y[i] += sub * x[i - 1]
            y[i] += diag * x[i]
            if i < self.n - 1:
                y[i] += sup * x[i + 1]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        """转置矩阵-向量乘积 y = A^T @ x。"""
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.n, dtype=float)
        sub, diag, sup = self.a
        for i in range(self.m):
            if i > 0:
                y[i - 1] += sub * x[i]
            y[i] += diag * x[i]
            if i < self.n - 1:
                y[i + 1] += sup * x[i]
        return y

    def residual(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """残差 r = b - A @ x。"""
        return b - self.mv(x)

    def to_dense(self) -> np.ndarray:
        """展开为稠密矩阵。"""
        A = np.zeros((self.m, self.n))
        sub, diag, sup = self.a
        for i in range(min(self.m, self.n)):
            A[i, i] = diag
        for i in range(1, min(self.m, self.n)):
            A[i, i - 1] = sub
            A[i - 1, i] = sup
        return A


# =============================================================================
# 迭代求解器
# =============================================================================
def r83s_cg(n: int, a: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
            tol: float = 1.0e-10, max_iter: int = None):
    """
    共轭梯度法求解对称正定的 R83S 线性系统 A x = b。
    a: shape (3,) 的 [sub, diag, super]。
    """
    if max_iter is None:
        max_iter = n
    A = R83SMatrix(n, n, a)
    b = np.asarray(b, dtype=float)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    r = b - A.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        Ap = A.mv(p)
        alpha = rs_old / (np.dot(p, Ap) + 1.0e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol * (np.linalg.norm(b) + 1.0e-30):
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def r83s_jacobi(n: int, a: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
                tol: float = 1.0e-10, max_iter: int = 10000):
    """Jacobi 迭代法求解 R83S 线性系统。"""
    A = R83SMatrix(n, n, a)
    b = np.asarray(b, dtype=float)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    sub, diag, sup = a
    if abs(diag) < 1.0e-14:
        raise ValueError("Zero diagonal in Jacobi")
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = 0.0
            if i > 0:
                s += sub * x[i - 1]
            if i < n - 1:
                s += sup * x[i + 1]
            x_new[i] = (b[i] - s) / diag
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def r83s_gauss_seidel(n: int, a: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
                      tol: float = 1.0e-10, max_iter: int = 10000):
    """Gauss-Seidel 迭代法求解 R83S 线性系统。"""
    A = R83SMatrix(n, n, a)
    b = np.asarray(b, dtype=float)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    sub, diag, sup = a
    if abs(diag) < 1.0e-14:
        raise ValueError("Zero diagonal in Gauss-Seidel")
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = 0.0
            if i > 0:
                s += sub * x[i - 1]
            if i < n - 1:
                s += sup * x[i + 1]
            x[i] = (b[i] - s) / diag
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


# =============================================================================
# Cholesky 分解（AS 6）
# =============================================================================
def cholesky_decompose(A: np.ndarray, eta: float = 1.0e-14):
    """
    对对称半正定矩阵 A 进行 Cholesky 分解 A = L @ L^T。
    返回 (L, nullty, ifault)。
    nullty: 秩亏数（接近零的对角元个数）。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    L = np.zeros((n, n), dtype=float)
    nullty = 0
    ifault = 0
    for j in range(n):
        L[j, j] = A[j, j]
        for k in range(j):
            L[j, j] -= L[j, k] * L[j, k]
        if L[j, j] <= eta:
            L[j, j] = 0.0
            nullty += 1
        else:
            L[j, j] = math.sqrt(L[j, j])
        for i in range(j + 1, n):
            L[i, j] = A[i, j]
            for k in range(j):
                L[i, j] -= L[i, k] * L[j, k]
            if L[j, j] != 0.0:
                L[i, j] /= L[j, j]
    return L, nullty, ifault


