"""
tridiagonal_solver.py
三对角矩阵求解器模块（R83 格式）
=================================
对应原项目 962_r83 的核心算法，将 MATLAB 实现迁移至 Python。
支持共轭梯度法（CG）、循环约化法（Cyclic Reduction）、Jacobi 迭代与 Gauss-Seidel 迭代。
所有矩阵均以紧凑 3×N 格式存储：
    A = [ subdiag[1:N] ; diag[0:N-1] ; superdiag[0:N-1] ]
其中 subdiag[0] 与 superdiag[N-1] 为占位零。
"""

import numpy as np
from system_utils import EPS, TOL_RANK, MAX_ITER, check_finite


# ---------------------------------------------------------------------------
# R83 基本运算
# ---------------------------------------------------------------------------

def r83_mv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    R83 矩阵向量乘法 y = A * x。

    参数
    ----
    A : np.ndarray, shape (3, n)
        subdiag (A[0]), diag (A[1]), superdiag (A[2])
    x : np.ndarray, shape (n,) or (n, nrhs)
    """
    A = np.asarray(A)
    x = np.asarray(x)
    n = A.shape[1]
    if x.shape[0] != n:
        raise ValueError("Dimension mismatch.")
    y = A[1] * x
    if n > 1:
        y[:-1] += A[2, :-1] * x[1:]
        y[1:] += A[0, 1:] * x[:-1]
    return y


def r83_mtv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    R83 转置矩阵向量乘法 y = A^T * x。
    """
    A = np.asarray(A)
    x = np.asarray(x)
    n = A.shape[1]
    y = A[1] * x
    if n > 1:
        y[:-1] += A[0, 1:] * x[1:]
        y[1:] += A[2, :-1] * x[:-1]
    return y


def r83_dif2(n: int) -> np.ndarray:
    """
    构造经典 1D 离散 Laplacian 三对角矩阵 DIF2：
        diag = 2,  subdiag = superdiag = -1
    对应有限差分格式 -u'' 在均匀网格上的离散化。

    特征值理论
    ----------
    λ_k = 4 * sin²( kπ / (2n+2) ),  k=1,...,n
    条件数 κ ≈ 4(n+1)² / π²，随 n 二次增长，严重病态。
    """
    A = np.zeros((3, n), dtype=float)
    A[1, :] = 2.0
    if n > 1:
        A[0, 1:] = -1.0
        A[2, :-1] = -1.0
    return A


def r83_res(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算残差 r = b - A*x。
    """
    return b - r83_mv(A, x)


# ---------------------------------------------------------------------------
# 共轭梯度法（CG）
# ---------------------------------------------------------------------------

def r83_cg(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
           tol: float = 1e-10, max_iter: int = None) -> np.ndarray:
    """
    针对 SPD 三对角矩阵的共轭梯度法求解 A x = b。

    算法推导
    --------
    对二次泛函  φ(x) = 0.5 x^T A x - b^T x，
    CG 沿 A-共轭方向 {p_k} 逐次最小化 φ，满足
        p_i^T A p_j = 0  (i≠j)
    迭代格式：
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k

    收敛性：对 DIF2，CG 理论上 n 步内精确收敛（满正交化），
    实际因舍入误差需设定容差。
    """
    # TODO: 实现基于 R83 紧凑格式的共轭梯度法（CG）求解器
    # 提示：
    #   1. A 为 R83 紧凑格式，shape=(3, n)，其中
    #      A[1,:] = 对角线元素，A[0,1:] = 下次对角线，A[2,:-1] = 上次对角线
    #   2. 使用 r83_mv 进行矩阵-向量乘法（R83 格式）
    #   3. 严格按照上述 docstring 中的 CG 迭代格式实现
    #   4. 注意与 fem_discretization.extract_tridiagonal 产生的 R83 格式保持一致
    raise NotImplementedError("Hole 2: R83 CG solver 待实现")


# ---------------------------------------------------------------------------
# 循环约化法（Cyclic Reduction）
# ---------------------------------------------------------------------------

def r83_cr_fa(A: np.ndarray) -> np.ndarray:
    """
    对三对角矩阵进行循环约化因子分解。

    数学原理
    --------
    将 n 个方程按奇偶分为两组：
        A_{even}  x_{even} + A_{odd} x_{odd} = b_{even}
    消去偶数（或奇数）指标，得到规模减半的约化系统，其新三对角系数为
        A'_{ii} = A_{ii} - A_{i,i±1} * A_{i±1,i} / A_{i±1,i±1}
    递归进行直至规模为 1，再回代求解。复杂度 O(n log n)。

    返回的 fac 为多层系数数组列表，每层 shape (3, m)。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    fac = [A.copy()]
    m = n
    while m > 1:
        m_prev = m
        m = m // 2
        if m < 1:
            break
        prev = fac[-1]
        nxt = np.zeros((3, m), dtype=float)
        for i in range(m):
            i2 = 2 * i + 1
            # 中心对角元
            diag = prev[1, i2]
            if i2 > 0:
                diag -= prev[0, i2] * prev[2, i2 - 1] / (prev[1, i2 - 1] + EPS)
            if i2 + 1 < m_prev:
                diag -= prev[2, i2] * prev[0, i2 + 1] / (prev[1, i2 + 1] + EPS)
            nxt[1, i] = diag
            if i > 0:
                nxt[0, i] = -prev[0, i2] * prev[0, i2 - 1] / (prev[1, i2 - 1] + EPS)
            if i + 1 < m:
                nxt[2, i] = -prev[2, i2] * prev[2, i2 + 1] / (prev[1, i2 + 1] + EPS)
        fac.append(nxt)
    return fac


def r83_cr_sl(fac: list, b: np.ndarray) -> np.ndarray:
    """
    使用循环约化因子分解求解 A x = b（单右端项）。
    """
    b = np.asarray(b, dtype=float)
    n = fac[0].shape[1]
    x = b.copy()
    # 前向约化
    levels = len(fac)
    rhs = [x]
    for lev in range(1, levels):
        m = fac[lev].shape[1]
        prev_rhs = rhs[-1]
        new_rhs = np.zeros(m, dtype=float)
        for i in range(m):
            i2 = 2 * i + 1
            val = prev_rhs[i2]
            if i2 > 0:
                val -= fac[lev - 1][0, i2] * prev_rhs[i2 - 1] / (fac[lev - 1][1, i2 - 1] + EPS)
            if i2 + 1 < len(prev_rhs):
                val -= fac[lev - 1][2, i2] * prev_rhs[i2 + 1] / (fac[lev - 1][1, i2 + 1] + EPS)
            new_rhs[i] = val
        rhs.append(new_rhs)
    # 最粗层求解
    x_coarse = rhs[-1] / (fac[-1][1, :] + EPS)
    # 回代插值
    sol = [x_coarse]
    for lev in range(levels - 2, -1, -1):
        m = fac[lev].shape[1]
        fine = np.zeros(m, dtype=float)
        coarse = sol[-1]
        for i in range(m):
            if i % 2 == 1:
                fine[i] = coarse[i // 2]
            else:
                fine[i] = rhs[lev][i]
                if i > 0:
                    fine[i] -= fac[lev][0, i] * fine[i - 1]
                if i + 1 < m:
                    fine[i] -= fac[lev][2, i] * fine[i + 1]
                fine[i] /= (fac[lev][1, i] + EPS)
        sol.append(fine)
    return sol[-1]


# ---------------------------------------------------------------------------
# Jacobi 与 Gauss-Seidel 迭代
# ---------------------------------------------------------------------------

def r83_jac_sl(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
               tol: float = 1e-10, max_iter: int = None) -> np.ndarray:
    """
    Jacobi 迭代求解 A x = b。

    迭代格式
    --------
    x_i^{(k+1)} = ( b_i - Σ_{j≠i} A_{ij} x_j^{(k)} ) / A_{ii}

    收敛条件：A 严格对角占优或对称正定（对 DIF2 满足）。
    """
    A = np.asarray(A)
    b = np.asarray(b)
    n = A.shape[1]
    if max_iter is None:
        max_iter = min(MAX_ITER, 10 * n)
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    diag = A[1, :].copy()
    diag = np.where(np.abs(diag) < EPS, EPS, diag)
    for _ in range(max_iter):
        x_new = b.copy()
        if n > 1:
            x_new[:-1] -= A[2, :-1] * x[1:]
            x_new[1:] -= A[0, 1:] * x[:-1]
        x_new /= diag
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x_new):
            return x_new
        x = x_new
    return x


def r83_gs_sl(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
              tol: float = 1e-10, max_iter: int = None) -> np.ndarray:
    """
    Gauss-Seidel 迭代求解 A x = b。

    迭代格式
    --------
    x_i^{(k+1)} = ( b_i - Σ_{j<i} A_{ij} x_j^{(k+1)} - Σ_{j>i} A_{ij} x_j^{(k)} ) / A_{ii}

    利用最新分量更新，收敛速度通常优于 Jacobi（对 SPD 矩阵）。
    """
    A = np.asarray(A)
    b = np.asarray(b)
    n = A.shape[1]
    if max_iter is None:
        max_iter = min(MAX_ITER, 10 * n)
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    diag = A[1, :].copy()
    diag = np.where(np.abs(diag) < EPS, EPS, diag)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += A[0, i] * x[i - 1]
            if i < n - 1:
                sigma += A[2, i] * x[i + 1]
            x[i] = (b[i] - sigma) / diag[i]
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x):
            return x
    return x
