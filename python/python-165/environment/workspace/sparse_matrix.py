"""
sparse_matrix.py
稀疏矩阵存储与迭代求解器
融合种子项目：r8st (sparse triplet format, CG solver)
"""

import numpy as np
from typing import List, Tuple, Optional


class SparseMatrix:
    """
    稀疏矩阵的 COO (Coordinate) / Triplet 格式存储。

    对于大规模电网导纳矩阵 Y_bus，其非零元数量与节点度数成正比，
    采用稀疏存储可将空间复杂度从 O(n^2) 降至 O(nnz)。
    """

    def __init__(self, n: int):
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []

    def add(self, i: int, j: int, v: float):
        """添加非零元。"""
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise IndexError("index out of bounds")
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(v)

    def to_dense(self) -> np.ndarray:
        """转为稠密矩阵（仅用于小规模验证）。"""
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A·x。
        时间复杂度 O(nnz)。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[i] += v * x[j]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵转置-向量乘法 y = A^T·x。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[j] += v * x[i]
        return y

    def diagonal(self) -> np.ndarray:
        """提取对角线元素。"""
        d = np.zeros(self.n, dtype=np.float64)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            if i == j:
                d[i] += v
        return d

    def residual(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算残差 r = b - A·x。"""
        return b - self.mv(x)


def conjugate_gradient(A: SparseMatrix, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       max_iter: Optional[int] = None) -> np.ndarray:
    """
    共轭梯度法（Conjugate Gradient, CG）求解 Ax = b。

    适用于 A 为对称正定（SPD）矩阵的情形。
    算法推导基于 Krylov 子空间上的能量泛函极小化：
        minimize  φ(x) = 0.5·x^T·A·x - b^T·x
    等价于求解 ∇φ = A·x - b = 0。

    迭代格式：
        r_0 = b - A·x_0,   p_0 = r_0
        α_k = (r_k^T·r_k) / (p_k^T·A·p_k)
        x_{k+1} = x_k + α_k·p_k
        r_{k+1} = r_k - α_k·A·p_k
        β_k = (r_{k+1}^T·r_{k+1}) / (r_k^T·r_k)
        p_{k+1} = r_{k+1} + β_k·p_k

    理论上，n 维 SPD 系统在最多 n 步内精确收敛；
    实际中用于大规模稀疏系统时，通常在远小于 n 的迭代次数内达到所需精度。

    在电网潮流计算中，CG 用于求解牛顿-拉夫逊迭代中的修正方程：
        J·Δx = -F(x)
    其中雅可比矩阵 J 在大规模系统中非常稀疏且近似 SPD，
    采用 CG 避免显式求逆，复杂度从 O(n^3) 降至 O(nnz·√κ)。
    """
    b = np.asarray(b, dtype=np.float64)
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64)
    if max_iter is None:
        max_iter = n

    r = A.residual(x, b)
    p = r.copy()
    rsold = float(np.dot(r, r))

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-15:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(np.dot(r, r))
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x


def jacobi_sparse_solve(A: SparseMatrix, b: np.ndarray,
                        x0: Optional[np.ndarray] = None,
                        tol: float = 1e-10,
                        max_iter: int = 10000) -> np.ndarray:
    """
    Jacobi 迭代法求解 Ax = b。

    将 A 分裂为 A = D - (L+U)，其中 D 为对角部分：
        x^{(k+1)} = D^{-1}·[b - (L+U)·x^{(k)}]

    收敛的充分条件：A 严格对角占优或不可约对角占优。
    在电网导纳矩阵中，对角元通常为节点所连导纳之和，
    满足严格对角占优，故 Jacobi 迭代收敛。
    """
    b = np.asarray(b, dtype=np.float64)
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64)

    d = A.diagonal()
    if np.any(np.abs(d) < 1e-15):
        raise ValueError("Jacobi iteration requires nonzero diagonal elements")

    for _ in range(max_iter):
        r = A.residual(x, b)
        dx = r / d
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x
