"""
================================================================================
线性代数引擎模块 (linear_algebra_engine.py)
================================================================================
融合项目:
  - 207_condition (condition_hager): 矩阵L1条件数估计
  - 736_matman: 初等行变换与LU分解

在可压缩CFD中，每个时间步需求解大型稀疏线性系统（压力泊松方程、隐式扩散）。
本模块提供：
  1. Hager条件数估计（监测刚度矩阵病态程度）
  2. 部分选主元LU分解（直接求解器）
  3. 矩阵预处理与正则化
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def condition_hager(n: int, a: np.ndarray) -> float:
    """
    Hager条件数估计 (Hager 1984, SIAM J. Sci. Stat. Comput.)

    估计矩阵 A 的 L1 条件数：

        κ₁(A) = ||A||₁ · ||A⁻¹||₁

    在CFD中用于监测离散算子的病态程度。当 κ₁ >> 1e12 时，
    网格质量差或时间步过大，需调整参数。

    算法通过迭代向量 b 最大化 ||A⁻¹ b||₁ / ||b||₁，
    避免了显式求逆的高昂代价。

    参数:
        n: 矩阵维度
        a: 方阵 (n x n)

    返回:
        cond: L1条件数估计值
    """
    if n <= 0:
        return 1.0

    i1 = -1
    c1 = 0.0
    b = np.ones((n, 1)) / n

    max_iter = 10
    it = 0

    while it < max_iter:
        it += 1
        try:
            b = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            # 矩阵奇异，添加正则化
            a_reg = a + 1e-10 * np.eye(n)
            b = np.linalg.solve(a_reg, b)

        c2 = np.sum(np.abs(b))
        b = np.sign(b)

        # 处理零元素
        b[b == 0.0] = 1.0

        try:
            b = np.linalg.solve(a.T, b)
        except np.linalg.LinAlgError:
            a_reg = a.T + 1e-10 * np.eye(n)
            b = np.linalg.solve(a_reg, b)

        i2 = int(np.argmax(np.abs(b)))

        if i1 >= 0:
            if i1 == i2 or c2 <= c1:
                break

        i1 = i2
        c1 = c2
        b = np.zeros((n, 1))
        b[i1] = 1.0

    norm_a = np.linalg.norm(a, 1)
    cond = float(c2 * norm_a)

    # 边界处理：防止极端值
    if cond < 1.0:
        cond = 1.0
    if not np.isfinite(cond):
        cond = 1e16

    return cond


def lu_decomposition_with_pivot(a: np.ndarray) -> tuple:
    """
    部分选主元Doolittle LU分解

    将矩阵 A 分解为 P A = L U，其中：
      - L 为单位下三角矩阵
      - U 为上三角矩阵
      - P 为置换矩阵

    在CFD中用于直接求解小型稠密子系统（如谱元内部的块系统）。

    参数:
        a: 输入方阵 (n x n)

    返回:
        L, U, P: 分解结果
        success: 是否成功
    """
    n = a.shape[0]
    if a.shape[0] != a.shape[1]:
        return None, None, None, False

    a_copy = a.astype(float).copy()
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n):
        # 部分选主元
        max_idx = np.argmax(np.abs(a_copy[k:, k])) + k
        if abs(a_copy[max_idx, k]) < 1e-15:
            # 奇异矩阵处理：添加微小扰动
            a_copy[k, k] += 1e-12
            max_idx = k

        if max_idx != k:
            a_copy[[k, max_idx], :] = a_copy[[max_idx, k], :]
            P[[k, max_idx], :] = P[[max_idx, k], :]

        for i in range(k + 1, n):
            a_copy[i, k] /= a_copy[k, k]
            for j in range(k + 1, n):
                a_copy[i, j] -= a_copy[i, k] * a_copy[k, j]

    # 提取L和U
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = 1.0
                U[j, i] = a_copy[j, i]
            elif j < i:
                L[i, j] = a_copy[i, j]
                U[j, i] = a_copy[j, i]
            else:
                U[j, i] = a_copy[j, i]

    return L, U, P, True


def solve_lu(L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """利用LU分解求解线性方程组 P A x = P b"""
    n = L.shape[0]
    pb = P @ b

    # 前向替换解 L y = P b
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - np.dot(L[i, :i], y[:i])

    # 回代解 U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def elementary_row_scale(A: np.ndarray, row: int, s: float) -> np.ndarray:
    """初等行变换：第 row 行乘以标量 s"""
    if abs(s) < 1e-15:
        raise ValueError("Scale factor must be non-zero")
    if row < 0 or row >= A.shape[0]:
        raise ValueError("Row index out of bounds")
    A = A.copy()
    A[row, :] *= s
    return A


def elementary_row_swap(A: np.ndarray, row1: int, row2: int) -> np.ndarray:
    """初等行变换：交换 row1 和 row2"""
    if row1 == row2:
        return A.copy()
    A = A.copy()
    A[[row1, row2], :] = A[[row2, row1], :]
    return A


def elementary_row_axpy(A: np.ndarray, target: int, source: int, s: float) -> np.ndarray:
    """初等行变换：target ← target + s · source"""
    if target == source:
        raise ValueError("Target and source rows must differ")
    A = A.copy()
    A[target, :] += s * A[source, :]
    return A


def jacobi_preconditioner(A: np.ndarray) -> np.ndarray:
    """
    Jacobi预处理矩阵：M = diag(A)⁻¹

    用于预处理共轭梯度法（PCG）求解压力泊松方程：

        M⁻¹ A x = M⁻¹ b

    当 A 为对称正定 M-矩阵时，Jacobi预处理可保证收敛。
    """
    diag = np.diag(A).copy()
    diag[np.abs(diag) < 1e-14] = 1e-14
    M_inv = np.diag(1.0 / diag)
    return M_inv


def apply_jacobi_precond(A: np.ndarray, b: np.ndarray, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Jacobi迭代求解 Ax = b"""
    n = len(b)
    x = np.zeros(n)
    diag = np.diag(A).copy()
    diag[np.abs(diag) < 1e-14] = 1e-14

    for _ in range(max_iter):
        x_new = (b - (A @ x - diag * x)) / diag
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x
