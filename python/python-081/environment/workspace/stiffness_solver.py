"""
刚度矩阵线性求解模块
====================
基于种子项目:
  - 981_r8ge: 通用稠密矩阵线性代数(PLU分解、共轭梯度法)
  - 984_r8lt: 下三角矩阵运算(前代法、Cholesky分解)

科学背景:
  非线性有限元的Newton-Raphson迭代每一步都需求解线性系统:
      K_T · Δu = -R
  其中 K_T 为切线刚度矩阵(对称正定，当载荷在稳定分支时)。
  本模块提供多种求解策略：
  1. PLU分解（带部分主元的高斯消去）——通用稠密矩阵
  2. Cholesky分解 ——对称正定矩阵的高效分解
  3. 预处理共轭梯度法(PCG) ——大规模稀疏系统的迭代求解

关键公式:
  - PLU分解: P A = L U
  - Cholesky分解: A = L L^T (L为下三角)
  - 前向替换求解 L y = b:  y_i = (b_i - Σ_{j<i} L_{ij} y_j) / L_{ii}
  - 后向替换求解 U x = y
  - CG迭代: 对 SPD 矩阵，在Krylov子空间上最小化能量范数误差
"""

import numpy as np
from typing import Tuple, Optional


# ========================================================================
# PLU 分解 (基于 r8ge 思想)
# ========================================================================

def plu_decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    带部分主元的PLU分解: P A = L U

    参数:
        A: (n, n) 方阵

    返回:
        P: (n, n) 置换矩阵
        L: (n, n) 单位下三角矩阵
        U: (n, n) 上三角矩阵
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("PLU分解要求方阵")
    U = A.copy().astype(np.float64)
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        # 部分主元
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[max_row, k]) < 1e-15:
            raise ValueError("PLU分解: 矩阵奇异或接近奇异")
        if max_row != k:
            U[[k, max_row], :] = U[[max_row, k], :]
            P[[k, max_row], :] = P[[max_row, k], :]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def solve_plu(P: np.ndarray, L: np.ndarray, U: np.ndarray,
              b: np.ndarray) -> np.ndarray:
    """
    利用PLU分解求解 Ax = b。
    步骤: Ly = Pb, Ux = y
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    pb = P @ b
    # 前向替换
    for i in range(n):
        s = pb[i] - np.dot(L[i, :i], y[:i])
        y[i] = s / L[i, i]
    # 后向替换
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise ValueError("solve_plu: U对角元接近零")
        x[i] = s / U[i, i]
    return x


# ========================================================================
# Cholesky 分解 (基于 r8lt 思想)
# ========================================================================

def cholesky_decompose(A: np.ndarray) -> np.ndarray:
    """
    对对称正定矩阵 A 进行Cholesky分解: A = L L^T
    L为下三角矩阵。

    算法:
      L_{ii} = sqrt(A_{ii} - Σ_{k<i} L_{ik}^2)
      L_{ji} = (A_{ji} - Σ_{k<i} L_{jk} L_{ik}) / L_{ii}   (j > i)

    参数:
        A: (n, n) 对称正定矩阵

    返回:
        L: (n, n) 下三角矩阵
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diag_sum = np.dot(L[i, :i], L[i, :i])
        val = A[i, i] - diag_sum
        if val <= 1e-15:
            # 边界处理: 添加微小扰动保证正定性
            val = 1e-12
        L[i, i] = np.sqrt(val)
        for j in range(i + 1, n):
            off_sum = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - off_sum) / L[i, i]
    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    利用Cholesky分解求解 Ax = b，其中 A = L L^T。
    步骤: 1) 前向替换 L y = b
          2) 后向替换 L^T x = y
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i] - np.dot(L[i, :i], y[:i])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Cholesky求解: L对角元接近零")
        y[i] = s / L[i, i]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i] - np.dot(L[i + 1:, i], x[i + 1:])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Cholesky求解: L对角元接近零")
        x[i] = s / L[i, i]
    return x


# ========================================================================
# 共轭梯度法 (基于 r8ge_cg 思想)
# ========================================================================

def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                        x0: Optional[np.ndarray] = None,
                        tol: float = 1e-10, max_iter: Optional[int] = None) -> np.ndarray:
    """
    共轭梯度法求解 Ax = b，要求 A 对称正定。

    算法:
      r_0 = b - A x_0
      p_0 = r_0
      for k = 0, 1, 2, ...:
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k

    参数:
        A: (n, n) SPD矩阵
        b: (n,) 右端向量
        x0: 初始猜测
        tol: 相对残差容差
        max_iter: 最大迭代次数，默认为 n

    返回:
        x: 近似解
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-14:
        norm_b = 1.0

    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-15:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new) / norm_b < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def apply_dirichlet_to_system(K: np.ndarray, R: np.ndarray,
                               bc_dofs: np.ndarray,
                               bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对全局刚度矩阵和残差向量施加Dirichlet边界条件（置1法）。

    参数:
        K: (n_dof, n_dof) 全局刚度矩阵
        R: (n_dof,) 残差向量
        bc_dofs: 受约束的自由度索引数组
        bc_values: 对应的位移值数组

    返回:
        K_mod, R_mod: 修改后的矩阵和向量
    """
    K_mod = K.copy()
    R_mod = R.copy()
    for dof, val in zip(bc_dofs, bc_values):
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0
        R_mod[dof] = val
    return K_mod, R_mod
