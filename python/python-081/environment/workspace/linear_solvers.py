"""
linear_solvers.py
博士级大变形非线性有限元分析 — 线性代数求解器模块

融合原项目:
  - 981_r8ge: 通用实矩阵存储、PLU分解、共轭梯度法
  - 984_r8lt: 下三角矩阵前向/后向替换

核心数学:
  全局切线刚度矩阵 K_T 的稀疏/稠密求解，包括:
  1. PLU 分解 (带部分主元):  K_T = P L U
  2. 共轭梯度法 (CG):  适用于对称正定近似
  3. 下三角前向替换:  用于直接求解器回代

公式:
  PLU 分解:
    P A = L U
    求解 A x = b:
      1) P A x = P b  =>  L U x = P b
      2) L y = P b   (前向替换)
      3) U x = y     (后向替换)

  共轭梯度法 (CG):
    给定对称正定矩阵 A，求解 A x = b
    初始化: r_0 = b - A x_0,  p_0 = r_0
    迭代 k = 0, 1, ..., n-1:
      α_k = (r_k^T r_k) / (p_k^T A p_k)
      x_{k+1} = x_k + α_k p_k
      r_{k+1} = r_k - α_k A p_k
      β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
      p_{k+1} = r_{k+1} + β_k p_k
"""

import numpy as np


class LinearSolverError(Exception):
    pass


def plu_decomposition(A):
    """
    PLU 分解（带部分主元选择）

    输入:
        A: (m, n) ndarray
    输出:
        P, L, U: 满足 P @ A = L @ U

    数学:
        对 j = 1, ..., min(m-1, n):
          选主元: pivot_row = argmax_{i>=j} |U(i,j)|
          交换第 j 行与 pivot_row 行（在 U, L, P 中同步交换）
          对 i = j+1, ..., m:
            L(i,j) = U(i,j) / U(j,j)
            U(i,j:n) = U(i,j:n) - L(i,j) * U(j,j:n)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    L = np.eye(m)
    P = np.eye(m)
    U = A.copy()

    for j in range(min(m - 1, n)):
        pivot_value = 0.0
        pivot_row = -1
        for i in range(j, m):
            if abs(U[i, j]) > pivot_value:
                pivot_value = abs(U[i, j])
                pivot_row = i

        if pivot_row == -1 or pivot_value < 1e-15:
            continue

        # 交换行
        U[[j, pivot_row], :] = U[[pivot_row, j], :]
        if j > 0:
            L[[j, pivot_row], :j] = L[[pivot_row, j], :j]
        P[[j, pivot_row], :] = P[[pivot_row, j], :]

        for i in range(j + 1, m):
            if abs(U[i, j]) > 1e-15:
                L[i, j] = U[i, j] / U[j, j]
                U[i, j] = 0.0
                U[i, j + 1:n] -= L[i, j] * U[j, j + 1:n]

    return P, L, U


def solve_plu(A, b):
    """
    使用 PLU 分解求解 A x = b
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    if m != n:
        raise LinearSolverError("PLU solve requires square matrix")
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != m:
        raise LinearSolverError("Dimension mismatch in PLU solve")

    P, L, U = plu_decomposition(A)
    # L y = P b
    y = forward_substitution(L, P @ b)
    # U x = y
    x = backward_substitution(U, y)
    return x.flatten()


def forward_substitution(L, b):
    """
    下三角矩阵前向替换求解 L x = b

    源自原项目 984_r8lt (r8lt_sl)
    数学:
      x_1 = b_1 / L_{11}
      x_i = (b_i - sum_{j=1}^{i-1} L_{ij} x_j) / L_{ii},  i = 2, ..., n
    """
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = b.copy()

    for j in range(n):
        if abs(L[j, j]) < 1e-15:
            raise LinearSolverError("Zero diagonal in forward substitution")
        x[j] = x[j] / L[j, j]
        if j + 1 < n:
            x[j + 1:n] -= L[j + 1:n, j].reshape(-1, 1) * x[j]
    return x


def backward_substitution(U, b):
    """
    上三角矩阵后向替换求解 U x = b
    """
    U = np.array(U, dtype=float)
    b = np.array(b, dtype=float)
    n = U.shape[0]
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = b.copy()

    for j in range(n - 1, -1, -1):
        if abs(U[j, j]) < 1e-15:
            raise LinearSolverError("Zero diagonal in backward substitution")
        x[j] = x[j] / U[j, j]
        if j > 0:
            x[0:j] -= U[0:j, j].reshape(-1, 1) * x[j]
    return x


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    共轭梯度法 (CG) 求解对称正定系统 A x = b

    源自原项目 981_r8ge (r8ge_cg)

    输入:
        A: (n, n) 对称正定矩阵
        b: (n,) 右端项
        x0: 初始猜测
        tol: 残差容差
        max_iter: 最大迭代次数，默认为 n
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    # 边界检查
    if A.shape[0] != A.shape[1]:
        raise LinearSolverError("CG requires square matrix")
    if b.shape[0] != n:
        raise LinearSolverError("Dimension mismatch in CG")

    # 检查正定性启发式
    diag_min = np.min(np.diag(A))
    if diag_min <= 0:
        # 尝试对称化处理
        A = 0.5 * (A + A.T)

    ap = A @ x
    r = b - ap
    p = r.copy()

    for it in range(max_iter):
        ap = A @ p
        pap = float(np.dot(p, ap))
        pr = float(np.dot(p, r))

        if abs(pap) < 1e-15:
            break

        alpha = pr / pap
        x = x + alpha * p
        r = r - alpha * ap

        # 收敛检查
        if np.linalg.norm(r) < tol:
            break

        rap = float(np.dot(r, ap))
        beta = -rap / pap
        p = r + beta * p

    return x


def matrix_vector_product(A, x):
    """
    通用矩阵-向量乘积 y = A x
    源自原项目 981_r8ge (r8ge_mv)
    """
    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)
    return A @ x


def matrix_matrix_product(A, B):
    """
    通用矩阵-矩阵乘积 C = A B
    源自原项目 981_r8ge (r8ge_mm)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    return A @ B
