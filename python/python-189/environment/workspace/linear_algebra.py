"""
linear_algebra.py

博士级数值线性代数工具库

基于种子项目:
  - 1048_rref2: 行简化阶梯形 (RREF) 线性系统求解
  - 964_r83p: 周期三对角矩阵 (R83P) 的 LU 分解与求解
  - 1262_toeplitz_cholesky: Toeplitz 矩阵的递推 Cholesky 分解

科学应用:
  1. RREF 用于优势函数 (advantage function) 的最小二乘估计:
     在策略梯度中, 需解 A·w ≈ b 其中 A 为特征矩阵, b 为时序差分回报.
     RREF 提供对亏秩系统的鲁棒求解.
  2. R83P 用于 Fisher 信息矩阵的快速逆运算:
     自然策略梯度中 Fisher 矩阵常具有周期三对角结构,
     利用 R83P 格式可将 O(n^3) 降为 O(n).
  3. Toeplitz Cholesky 用于高斯策略协方差矩阵的采样:
     时间序列策略的协方差矩阵常是 Toeplitz 型,
     其 Cholesky 因子可用于生成相关动作噪声.
"""

import numpy as np
from math import sqrt
from typing import Tuple


# ---------------------------------------------------------------------------
# RREF (Reduced Row Echelon Form) 求解器
# ---------------------------------------------------------------------------

def rref_compute(A: np.ndarray, tol: float = 1.0e-12) -> Tuple[np.ndarray, list]:
    """
    将矩阵 A 化为行简化阶梯形 (RREF).

    数学定义:
        RREF 满足:
        (1) 非零行在零行之上;
        (2) 每行主元 (pivot) 严格在上一行主元右侧;
        (3) 主元为 1, 且主元所在列其余元素为 0.

    参数:
        A:  m×n 实矩阵
        tol: 零判断容差

    返回:
        (ARREF, pivot_cols)  其中 pivot_cols 记录主元列索引
    """
    A = A.astype(float).copy()
    m, n = A.shape
    pivot_cols = []
    row = 0
    for col in range(n):
        # 在 col 列中从当前行向下找绝对值最大元
        pivot_val = 0.0
        pivot_row = -1
        for r in range(row, m):
            if abs(A[r, col]) > pivot_val:
                pivot_val = abs(A[r, col])
                pivot_row = r
        if pivot_val <= tol:
            continue
        # 交换行
        if pivot_row != row:
            A[[pivot_row, row], :] = A[[row, pivot_row], :]
        # 归一化主元行
        A[row, :] = A[row, :] / A[row, col]
        # 消去其他行
        for r in range(m):
            if r != row and abs(A[r, col]) > tol:
                A[r, :] = A[r, :] - A[r, col] * A[row, :]
        pivot_cols.append(col)
        row += 1
        if row >= m:
            break
    return A, pivot_cols


def rref_solve(A: np.ndarray, b: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    """
    使用 RREF 求解线性系统 A·x = b (允许 A 为奇异/亏秩矩阵).

    数学推导:
        构造增广矩阵 [A | b], 化为 RREF 后, 右侧即为解.
        若 rank(A) < rank([A|b]), 系统不相容, 返回最小二乘解.

    参数:
        A: m×n 系数矩阵
        b: m×k 右端项 (可有多列)

    返回:
        x: n×k 解矩阵
    """
    A = np.atleast_2d(A).astype(float)
    b = np.asarray(b, dtype=float)
    m1, n1 = A.shape
    if b.ndim == 1:
        if len(b) == m1:
            b = b.reshape(-1, 1)
        else:
            b = b.reshape(1, -1)
    m2, n2 = b.shape
    if m1 != m2:
        raise ValueError(f"rref_solve: A has {m1} rows but b has {m2} rows")
    AI = np.hstack([A, b])
    AIRREF, pivot_cols = rref_compute(AI, tol=tol)
    x = AIRREF[:n1, n1:n1 + n2]
    return x


def rref_rank(A: np.ndarray, tol: float = 1.0e-12) -> int:
    """通过 RREF 计算矩阵的秩."""
    _, pivot_cols = rref_compute(A, tol=tol)
    return len(pivot_cols)


# ---------------------------------------------------------------------------
# R83P 周期三对角矩阵求解器
# ---------------------------------------------------------------------------

def r83_np_fa(n: int, a: np.ndarray) -> np.ndarray:
    """
    R83 NP (非周期三对角) 矩阵的 LU 分解.

    存储格式: a[0,i] = 上对角线 (i=0..n-2)
             a[1,i] = 主对角线 (i=0..n-1)
             a[2,i] = 下对角线 (i=1..n-1)
    """
    if n < 2:
        raise ValueError("r83_np_fa: n must be at least 2")
    a_lu = a.copy()
    for i in range(1, n):
        a_lu[2, i] = a_lu[2, i] / a_lu[1, i - 1]
        a_lu[1, i] = a_lu[1, i] - a_lu[2, i] * a_lu[0, i - 1]
    return a_lu


def r83_np_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int = 0) -> np.ndarray:
    """
    求解 R83 NP 线性系统 (已 LU 分解).

    参数:
        job: 0 解 A·x=b, 非零 解 A^T·x=b
    """
    x = b.copy().astype(float)
    if job == 0:
        # 前代
        for i in range(1, n):
            x[i] = x[i] - a_lu[2, i] * x[i - 1]
        # 回代
        for i in range(n - 1, -1, -1):
            x[i] = x[i] / a_lu[1, i]
            if i > 0:
                x[i - 1] = x[i - 1] - a_lu[0, i - 1] * x[i]
    else:
        # 解转置
        for i in range(n):
            x[i] = x[i] / a_lu[1, i]
            if i < n - 1:
                x[i + 1] = x[i + 1] - a_lu[0, i] * x[i]
        for i in range(n - 1, 0, -1):
            x[i - 1] = x[i - 1] - a_lu[2, i] * x[i]
    return x


def r83p_fa(n: int, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    R83P (周期三对角) 矩阵的 LU 分解.

    矩阵结构:
        A = [ d1  u1          l1
              l2  d2  u2
                  l3  d3  u3
                      ... ...
              u_n             l_n  d_n ]

    存储格式: a[0,0]=A(n,1), a[0,1..n-1]=上对角线
             a[1,0..n-1]=主对角线
             a[2,0..n-2]=下对角线, a[2,n-1]=A(1,n)

    返回:
        (a_lu, work2, work3, work4)
        work2 = A1^{-1} * A2,  work3 = (A1^{-1} * A3)^T,  work4 = Schur 补
    """
    if n < 3:
        raise ValueError("r83p_fa: n must be at least 3")
    a_lu = a.copy()
    # 提取周期角块
    a2 = np.zeros(n - 1)
    a3 = np.zeros(n - 1)
    a2[0] = a[0, 0]      # A(n,1)
    a2[n - 2] = a[2, n - 1]  # A(1,n)
    a3[0] = a[2, 0]      # A(2,1) 实际上 a3 不需要全部填充
    a3[n - 2] = a[0, n - 2]  # A(n-1,n)

    # 对 A1 (前 n-1 阶非周期三对角) 做 LU
    a_lu_part = r83_np_fa(n - 1, a_lu[:, :n - 1])
    a_lu[:, :n - 1] = a_lu_part

    # work2 = A1^{-1} * A2, 其中 A2[0]=a[0,0], A2[n-2]=a[2,n-1]
    work2 = np.zeros(n - 1)
    work2[0] = a[0, 0]
    work2[n - 2] = a[2, n - 1]
    work2 = r83_np_sl(n - 1, a_lu[:, :n - 1], work2, job=0)

    # work3 = (A1^{-1} * A3)^T, A3[0]=a[2,0], A3[n-2]=a[0,n-2]
    work3 = np.zeros(n - 1)
    work3[0] = a[2, 0]
    work3[n - 2] = a[0, n - 2]
    work3 = r83_np_sl(n - 1, a_lu[:, :n - 1], work3, job=0)

    # Schur 补
    work4 = a[1, n - 1] - a[0, n - 1] * work2[n - 2] - a[2, n - 2] * work3[n - 2]
    if abs(work4) < 1.0e-15:
        raise ValueError("r83p_fa: singular matrix")

    return a_lu, work2, work3, work4


def r83p_sl(n: int, a_lu: np.ndarray, b: np.ndarray,
            job: int, work2: np.ndarray, work3: np.ndarray, work4: float) -> np.ndarray:
    """
    求解 R83P 系统 (已分解).
    """
    x = b.copy().astype(float)
    if job == 0:
        x[:n - 1] = r83_np_sl(n - 1, a_lu[:, :n - 1], x[:n - 1], job=0)
        x[n - 1] = x[n - 1] - a_lu[0, 0] * x[0] - a_lu[2, n - 2] * x[n - 2]
        x[n - 1] = x[n - 1] / work4
        x[:n - 1] = x[:n - 1] - work2 * x[n - 1]
    else:
        x[:n - 1] = r83_np_sl(n - 1, a_lu[:, :n - 1], x[:n - 1], job=1)
        x[n - 1] = x[n - 1] - a_lu[2, n - 1] * x[0] - a_lu[0, n - 1] * x[n - 2]
        x[n - 1] = x[n - 1] / work4
        x[:n - 1] = x[:n - 1] - work3 * x[n - 1]
    return x


def r83p_solve(n: int, a: np.ndarray, b: np.ndarray, job: int = 0) -> np.ndarray:
    """
    一步求解 R83P 系统 A·x = b.
    """
    a_lu, work2, work3, work4 = r83p_fa(n, a)
    return r83p_sl(n, a_lu, b, job, work2, work3, work4)


# ---------------------------------------------------------------------------
# Toeplitz Cholesky 分解
# ---------------------------------------------------------------------------

def toeplitz_cholesky_lower(n: int, a: np.ndarray) -> np.ndarray:
    """
    计算半正定 Toeplitz 矩阵的下三角 Cholesky 因子 L (A = L·L^T).

    数学背景:
        Toeplitz 矩阵 T 满足 T_{i,j} = t_{|i-j|}, 即沿对角线常数.
        在策略梯度中, 时间序列相关噪声的协方差矩阵常是 Toeplitz 型.

    算法:  Stewart (1997) 的递推算法, O(n^2) 复杂度.

    参数:
        n: 矩阵阶数
        a: n×n Toeplitz 矩阵 (只需对称正定)

    返回:
        L: n×n 下三角矩阵
    """
    if n < 1:
        raise ValueError("toeplitz_cholesky_lower: n must be positive")
    a = np.atleast_2d(a).astype(float)
    if a.shape != (n, n):
        raise ValueError("toeplitz_cholesky_lower: a shape mismatch")
    # 检查正定性 (对角元)
    for i in range(n):
        if a[i, i] <= 0:
            raise ValueError(f"toeplitz_cholesky_lower: non-positive diagonal at {i}")

    # Stewart (1997) 的广义 Schur 算法
    # 对生成元进行缩放以确保 A = L * L^T
    scale = np.sqrt(a[0, 0])
    if scale < 1.0e-12:
        raise ValueError("toeplitz_cholesky_lower: zero or near-zero diagonal")
    g = np.zeros((2, n))
    g[0, :] = a[:, 0] / scale
    g[1, 0] = 0.0
    g[1, 1:n] = a[1:n, 0] / scale

    L = np.zeros((n, n))
    L[:, 0] = g[0, :]
    g[0, 1:n] = g[0, 0:n - 1]
    g[0, 0] = 0.0

    for i in range(1, n):
        rho = -g[1, i] / g[0, i]
        denom = sqrt((1.0 - rho) * (1.0 + rho))
        if abs(denom) < 1.0e-15:
            raise ValueError("toeplitz_cholesky_lower: breakdown at step {}".format(i))
        A_mat = np.array([[1.0, rho], [rho, 1.0]])
        g[:, i:n] = (A_mat @ g[:, i:n]) / denom
        L[i:n, i] = g[0, i:n]
        if i + 1 < n:
            g[0, i + 1:n] = g[0, i:n - 1]
        g[0, i] = 0.0
    return L


def sample_from_toeplitz_covariance(n: int, first_col: np.ndarray) -> np.ndarray:
    """
    从由 Toeplitz 协方差矩阵描述的高斯分布中采样.

    参数:
        n: 维度
        first_col: 协方差矩阵第一列 [c0, c1, ..., c_{n-1}]

    返回:
        n 维样本向量
    """
    first_col = np.asarray(first_col, dtype=float)
    if len(first_col) < n:
        raise ValueError("sample_from_toeplitz_covariance: first_col too short")
    first_col = first_col[:n]
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = first_col[abs(i - j)]
    # 检查正定性并微调
    eigvals = np.linalg.eigvalsh(T)
    if np.min(eigvals) <= 0:
        T = T + (-np.min(eigvals) + 1.0e-10) * np.eye(n)
    L = toeplitz_cholesky_lower(n, T)
    z = np.random.randn(n)
    return L @ z
