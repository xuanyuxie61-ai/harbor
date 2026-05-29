"""
numerical_linear_algebra.py
===========================
基于 034_asa082、844_pagerank 改造的数值线性代数工具模块。

在 CFD 模拟中，线性代数操作无处不在：网格变换 Jacobian 的行列式检查、
稀疏线性系统的迭代求解、稳态浓度场的幂法收敛。本模块提供：
1. 正交矩阵行列式计算（detq）
2. 幂法求主特征向量（PageRank 式收敛判定）
3. 矩阵条件数估计与稳定性分析

核心公式
--------
1. 正交矩阵行列式（Gower AS 82）：
       若 A 为正交矩阵（A^T A = I），则 det(A) = ±1。
       本算法通过逐步 Householder 化验证正交性并精确计算符号。

2. 幂法（Power Iteration）：
       x_{k+1} = T x_k / ||T x_k||
       其中 T 为 Google/转移矩阵（或 CFD 迭代矩阵）。
       收敛判据：||x_{k+1} - x_k||_∞ < tol。
       主特征值 λ ≈ (x^T T x) / (x^T x)。

3. 稳态物质浓度场的线性系统：
       A c = b
       其中 A = I - Δt D^{-1} K，K 为对流-扩散算子离散矩阵，
       D 为对角质量矩阵。PageRank 的阻尼因子 α 类比于
       隐式时间步进的松弛因子。

4. 矩阵条件数（谱条件数）：
       κ_2(A) = σ_max(A) / σ_min(A)
       大条件数预示数值不稳定性。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Determinant of orthogonal matrix (from 034_asa082)
# ---------------------------------------------------------------------------

def detq_orthogonal(a):
    """
    计算正交矩阵的行列式（Gower AS 82 算法）。

    Parameters
    ----------
    a : ndarray, shape (n, n)
        正交矩阵（行或列存储）。

    Returns
    -------
    d : float
        行列式值（应为 ±1）。
    ifault : int
        0 = 成功，1 = 错误。
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    tol = 0.0001
    ifault = 0
    d = 0.0

    if n <= 0:
        ifault = 1
        return d, ifault

    a2 = a.flatten()
    d = 1.0
    r = 0

    for k in range(1, n + 1):
        q = r
        x = a2[r]
        y = np.sign(x)
        if abs(y) < 1e-15:
            y = 1.0 if x >= 0 else -1.0
        d = d * y
        denom = x + y
        if abs(denom) < 1e-15:
            denom = 1e-15 if denom >= 0 else -1e-15
        y = -1.0 / denom
        x = abs(x) - 1.0

        if tol < abs(x):
            if 0.0 < x:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault
            for i in range(k, n):
                q = q + n
                x_val = a2[q] * y
                p = r
                s = q
                for j in range(k, n):
                    p = p + 1
                    s = s + 1
                    a2[s] = a2[s] + x_val * a2[p]
        r = r + n + 1

    return d, ifault


def check_mesh_transformation_orthogonality(Jac):
    """
    检查 CFD 网格变换 Jacobian 矩阵的正交性。
    在结构化网格中，局部坐标变换矩阵应为近似正交。

    Parameters
    ----------
    Jac : ndarray, shape (n, n)
        Jacobian 矩阵（物理坐标 ↔ 计算坐标）。

    Returns
    -------
    is_orthogonal : bool
    det_value : float
    orthogonality_error : float
        ||J^T J - I||_F。
    """
    Jac = np.asarray(Jac, dtype=float)
    n = Jac.shape[0]
    I = np.eye(n)
    JTJ = Jac.T @ Jac
    err = np.linalg.norm(JTJ - I, ord='fro')
    d, fault = detq_orthogonal(Jac)
    is_ortho = (fault == 0) and (err < 0.1)
    return is_ortho, d, err


# ---------------------------------------------------------------------------
# Power method / PageRank (from 844_pagerank)
# ---------------------------------------------------------------------------

def power_iteration_eigenvector(A, max_iter=200, tol=1e-10, damping=0.85,
                                verbose=False):
    """
    幂法求矩阵 A 的模最大特征值与对应特征向量。
    受 PageRank 启发，引入阻尼因子处理 reducible 矩阵。

    Parameters
    ----------
    A : ndarray, shape (n, n)
        方阵（转移矩阵或迭代矩阵）。
    max_iter : int
    tol : float
    damping : float
        Google 阻尼因子（类比于松弛参数）。
    verbose : bool

    Returns
    -------
    eigenvalue : float
    eigenvector : ndarray
    iterations : int
    converged : bool
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    # 归一化列随机矩阵（若输入为邻接矩阵）
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    T = A / col_sums

    # Google 矩阵：G = damping * T + (1-damping)/n * 1 1^T
    G = damping * T + (1.0 - damping) / n * np.ones((n, n))

    x = np.ones(n) / n
    for it in range(1, max_iter + 1):
        x_new = G @ x
        x_new = x_new / np.linalg.norm(x_new, ord=1)
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            if verbose:
                print(f"[Power] Converged in {it} iterations, diff={diff:.3e}")
            eigenvalue = float((x.T @ G @ x) / (x.T @ x))
            return eigenvalue, x, it, True

    if verbose:
        print(f"[Power] Max iter reached, diff={diff:.3e}")
    eigenvalue = float((x.T @ G @ x) / (x.T @ x))
    return eigenvalue, x, max_iter, False


def steady_state_concentration_solver(K, b, alpha_relax=0.8, max_iter=500,
                                      tol=1e-8):
    """
    用幂法思想求解稳态对流-扩散浓度场：
        (D + K) c = b
    其中 D 为隐式时间步进的对角矩阵。
    改写为不动点形式：c = G c + f，再用幂法收敛。

    Parameters
    ----------
    K : ndarray, shape (n, n)
        对流-扩散算子离散矩阵（可能非对称）。
    b : ndarray, shape (n,)
        源项。
    alpha_relax : float
        松弛因子（类比 PageRank 阻尼）。
    max_iter : int
    tol : float

    Returns
    -------
    c : ndarray
    residual : float
    iterations : int
    converged : bool
    """
    K = np.asarray(K, dtype=float)
    b = np.asarray(b, dtype=float)
    n = K.shape[0]

    # 构造 Jacobi 型迭代矩阵
    diag = np.diag(K)
    diag = np.where(np.abs(diag) < 1e-12, 1e-10, diag)
    D_inv = np.diag(1.0 / diag)

    # 不动点映射：c_{new} = (1-alpha) c + alpha (D^{-1} (b - (K-D) c))
    M = np.eye(n) - alpha_relax * D_inv @ K
    f = alpha_relax * D_inv @ b

    c = np.zeros(n)
    for it in range(1, max_iter + 1):
        c_new = M @ c + f
        diff = np.linalg.norm(c_new - c, ord=np.inf)
        c = c_new
        if diff < tol:
            res = np.linalg.norm(K @ c - b, ord=np.inf)
            return c, res, it, True

    res = np.linalg.norm(K @ c - b, ord=np.inf)
    return c, res, max_iter, False


# ---------------------------------------------------------------------------
# Condition number and stability
# ---------------------------------------------------------------------------

def estimate_condition_number(A):
    """
    估计矩阵的谱条件数 κ_2(A)。
    """
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
