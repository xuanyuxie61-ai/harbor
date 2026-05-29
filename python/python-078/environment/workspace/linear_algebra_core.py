"""
linear_algebra_core.py
稀疏三对角线性系统求解器

融合来源:
- 966_r83t: R83T三对角矩阵紧凑存储格式、CG/GS/Jacobi迭代求解器、DIF2测试矩阵

科学背景:
在动脉脉动流的数值模拟中，Womersley方程（轴对称非定常NS方程的径向部分）
经过有限差分离散后，在每个时间步都需要求解形如 Ax=b 的三对角线性系统：

    (1 - α) u_{j-1}^{n+1} + (1 + 2α) u_j^{n+1} - α u_{j+1}^{n+1} = RHS_j

其中 α = νΔt/(Δr)²。该系统对称正定，适合用共轭梯度法（CG）或直接法求解。

本模块提供R83T格式的矩阵-向量乘法、残差计算，以及三种迭代求解器。
"""

import numpy as np
from typing import Tuple, Optional


# ======================================================================
# R83T 三对角矩阵紧凑存储格式
# ======================================================================

class R83TMatrix:
    """
    R83T格式三对角矩阵存储。
    对于一个 M×N 的三对角矩阵，存储为 M×3 的数组：
        data[:, 0] = 次对角线 (sub-diagonal)
        data[:, 1] = 主对角线 (main diagonal)
        data[:, 2] = 超对角线 (super-diagonal)
    """
    def __init__(self, data: np.ndarray, n: int):
        """
        参数:
            data: (n, 3) 数组
            n: 矩阵维度（方阵）
        """
        if data.shape != (n, 3):
            raise ValueError(f"R83T data shape {data.shape} incompatible with n={n}")
        self.data = data.astype(float)
        self.n = n

    @classmethod
    def from_diagonals(cls, sub: np.ndarray, main: np.ndarray, super: np.ndarray):
        """从三条对角线构造R83T矩阵。"""
        n = len(main)
        if len(sub) != n or len(super) != n:
            raise ValueError("Diagonals must have same length")
        data = np.column_stack([sub, main, super])
        return cls(data, n)

    @classmethod
    def dif2(cls, n: int) -> 'R83TMatrix':
        """
        构造经典的DIF2测试矩阵（一维Laplacian离散）。

        矩阵形式:
            A = [ 2  -1   0   ...   0 ]
                [-1   2  -1   ...   0 ]
                [ 0  -1   2   ...   0 ]
                ...
                [ 0   0   0   ...   2 ]

        特征值: λ_i = 4 sin²(iπ / (2n+2)), i=1,...,n
        条件数 ~ 4n²/π²
        """
        sub = np.full(n, -1.0)
        sub[0] = 0.0
        main = np.full(n, 2.0)
        super = np.full(n, -1.0)
        super[-1] = 0.0
        data = np.column_stack([sub, main, super])
        return cls(data, n)

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵（仅用于小维度测试）。"""
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i > 0:
                A[i, i - 1] = self.data[i, 0]
            A[i, i] = self.data[i, 1]
            if i < self.n - 1:
                A[i, i + 1] = self.data[i, 2]
        return A

    def eigenvalue_dif2(self, i: int) -> float:
        """
        DIF2矩阵的理论特征值（仅当本矩阵为DIF2时有效）。
        λ_i = 4 sin²(iπ / (2n+2))
        """
        if i < 1 or i > self.n:
            raise ValueError("Index out of range")
        return 4.0 * np.sin(i * np.pi / (2.0 * self.n + 2.0)) ** 2


def r83t_mv(A: R83TMatrix, x: np.ndarray) -> np.ndarray:
    """
    R83T格式矩阵-向量乘法: y = A @ x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != A.n:
        raise ValueError("Dimension mismatch")
    y = np.zeros(A.n)
    for i in range(A.n):
        if i > 0:
            y[i] += A.data[i, 0] * x[i - 1]
        y[i] += A.data[i, 1] * x[i]
        if i < A.n - 1:
            y[i] += A.data[i, 2] * x[i + 1]
    return y


def r83t_mtv(A: R83TMatrix, x: np.ndarray) -> np.ndarray:
    """
    R83T格式转置矩阵-向量乘法: y = A^T @ x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != A.n:
        raise ValueError("Dimension mismatch")
    y = np.zeros(A.n)
    for i in range(A.n):
        if i > 0:
            y[i - 1] += A.data[i, 0] * x[i]
        y[i] += A.data[i, 1] * x[i]
        if i < A.n - 1:
            y[i + 1] += A.data[i, 2] * x[i]
    return y


def r83t_res(A: R83TMatrix, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算残差: r = b - A @ x
    """
    return b - r83t_mv(A, x)


# ======================================================================
# 迭代求解器
# ======================================================================

def r83t_jacobi_solve(A: R83TMatrix, b: np.ndarray,
                      x0: Optional[np.ndarray] = None,
                      max_iter: int = 10000,
                      tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    """
    Jacobi迭代法求解 Ax = b。

    迭代格式:
        x_i^{(k+1)} = (b_i - Σ_{j≠i} A_{ij} x_j^{(k)}) / A_{ii}

    收敛条件: A严格对角占优或对称正定（对于DIF2类矩阵，Jacobi收敛较慢）。

    返回:
        x: 解向量
        iters: 实际迭代次数
        residual: 最终残差范数
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")

    # 检查主对角线非零
    diag = A.data[:, 1].copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal element detected")

    x_new = np.zeros(n)
    for it in range(max_iter):
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A.data[i, 0] * x[i - 1]
            if i < n - 1:
                sigma += A.data[i, 2] * x[i + 1]
            x_new[i] = (b[i] - sigma) / diag[i]

        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x[:] = x_new[:]

        if diff < tol:
            r = r83t_res(A, x, b)
            return x, it + 1, float(np.linalg.norm(r))

    r = r83t_res(A, x, b)
    return x, max_iter, float(np.linalg.norm(r))


def r83t_gauss_seidel_solve(A: R83TMatrix, b: np.ndarray,
                            x0: Optional[np.ndarray] = None,
                            max_iter: int = 10000,
                            tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    """
    Gauss-Seidel迭代法求解 Ax = b。

    迭代格式（使用最新可用值）:
        x_i^{(k+1)} = (b_i - Σ_{j<i} A_{ij} x_j^{(k+1)} - Σ_{j>i} A_{ij} x_j^{(k)}) / A_{ii}

    对于对称正定矩阵，GS总是收敛，且比Jacobi快约2倍。
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")

    diag = A.data[:, 1].copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Zero diagonal element detected")

    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A.data[i, 0] * x[i - 1]
            if i < n - 1:
                sigma += A.data[i, 2] * x[i + 1]
            x[i] = (b[i] - sigma) / diag[i]

        diff = np.linalg.norm(x - x_old, ord=np.inf)
        if diff < tol:
            r = r83t_res(A, x, b)
            return x, it + 1, float(np.linalg.norm(r))

    r = r83t_res(A, x, b)
    return x, max_iter, float(np.linalg.norm(r))


def r83t_cg_solve(A: R83TMatrix, b: np.ndarray,
                  x0: Optional[np.ndarray] = None,
                  max_iter: int = None,
                  tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    """
    共轭梯度法（Conjugate Gradient）求解对称正定三对角系统 Ax = b。

    算法推导:
    最小化二次泛函: Φ(x) = 0.5 x^T A x - b^T x
    迭代方向 p_k 满足 A-共轭性: p_i^T A p_j = 0 (i≠j)

    迭代格式:
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k

    理论性质: 对于n维系统，最多n步精确收敛（不考虑舍入误差）。
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b) != n:
        raise ValueError("Dimension mismatch")

    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()
    if len(x) != n:
        raise ValueError("Dimension mismatch")

    r = b - r83t_mv(A, x)
    p = r.copy()
    rs_old = float(np.dot(r, r))

    # 边界检查：若b为零向量或已收敛
    if rs_old < tol * tol:
        return x, 0, float(np.sqrt(rs_old))

    for it in range(max_iter):
        Ap = r83t_mv(A, p)
        pAp = float(np.dot(p, Ap))

        if abs(pAp) < 1e-15:
            raise RuntimeError("CG breakdown: p^T A p ≈ 0")

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))

        if np.sqrt(rs_new) < tol:
            return x, it + 1, float(np.sqrt(rs_new))

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, float(np.sqrt(rs_new))


# ======================================================================
# 针对Womersley方程的三对角系统构造
# ======================================================================

def build_womersley_tridiagonal(n_r: int, alpha_w: float,
                                dt: float, dr: float,
                                kinematic_viscosity: float) -> R83TMatrix:
    """
    构造Womersley方程隐式时间步进的三对角系统。

    Womersley方程（轴对称脉动流）:
        ∂u/∂t = -1/ρ ∂p/∂z + ν (∂²u/∂r² + (1/r) ∂u/∂r)

    隐式离散（向后Euler + 中心差分）:
        (u_j^{n+1} - u_j^n)/Δt = RHS^n + ν [ (u_{j-1}^{n+1} - 2u_j^{n+1} + u_{j+1}^{n+1})/Δr²
                                           + (u_{j+1}^{n+1} - u_{j-1}^{n+1})/(2 r_j Δr) ]

    整理为: a_j u_{j-1}^{n+1} + b_j u_j^{n+1} + c_j u_{j+1}^{n+1} = d_j

    参数:
        n_r: 径向网格点数
        alpha_w: Womersley数（用于边界条件校验）
        dt: 时间步长
        dr: 径向网格间距
        kinematic_viscosity: 运动粘度 ν

    返回:
        R83TMatrix: 三对角系数矩阵
    """
    # TODO: 构造Womersley方程隐式时间步进的三对角系数矩阵
    # 提示：
    #   1. 验证 dr, dt, kinematic_viscosity 为正
    #   2. 计算 coeff = nu * dt / dr^2
    #   3. 对 j=0 (轴心，r=0)：对称边界条件
    #   4. 对 j=n_r-1 (壁面)：Dirichlet u=0
    #   5. 对内部点：中心差分离散 (∂²u/∂r² + (1/r)∂u/∂r)
    #   返回 R83TMatrix.from_diagonals(sub, main, super)
    raise NotImplementedError("Hole 1: build_womersley_tridiagonal 待实现")


def thomas_algorithm(A: R83TMatrix, b: np.ndarray) -> np.ndarray:
    """
    Thomas算法（三对角矩阵直接求解，O(n)复杂度）。

    前向消去:
        c'_i = c_i / (b_i - a_i c'_{i-1})
        d'_i = (d_i - a_i d'_{i-1}) / (b_i - a_i c'_{i-1})

    回代:
        x_n = d'_n
        x_i = d'_i - c'_i x_{i+1}

    适用于严格对角占优的三对角系统。
    """
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    n = A.n
    if len(b_vec) != n:
        raise ValueError("Dimension mismatch")

    a = A.data[:, 0].copy()
    bb = A.data[:, 1].copy()
    c = A.data[:, 2].copy()
    d = b_vec.copy()

    # 前向消去
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / (bb[0] + 1e-15)
    dp[0] = d[0] / (bb[0] + 1e-15)

    for i in range(1, n):
        denom = bb[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # 回代
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x
