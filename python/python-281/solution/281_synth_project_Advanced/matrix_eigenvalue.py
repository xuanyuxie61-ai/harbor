"""
matrix_eigenvalue.py
====================
矩阵分解与特征值分析模块。
融合项目: 938_qr_solve (QR 分解与线性求解),
         1405_web_matrix (幂法与 PageRank 型迭代)

核心方法:
  1. QR 分解 (Householder 反射)
  2. QR 迭代求特征值
  3. 幂法求主特征值/特征向量
  4. 隐式 QR 位移 (加速收敛)
  5. 线性系统求解 (通过 QR)

应用:
  - 隐式时间步的线性系统求解
  - 放大矩阵特征值计算 (稳定性分析)
  - 转移矩阵的 PageRank 型稳态分析
"""

import math
import numpy as np


def householder_reflection(x):
    """
    计算 Householder 反射向量。

    给定向量 x, 求 v 使得 H = I - 2*v*v^T / (v^T*v)
    满足 H*x = α*e₁, 其中 α = ±||x||

    Parameters
    ----------
    x : ndarray, shape (n,)
        输入向量

    Returns
    -------
    v : ndarray, shape (n,)
        Householder 向量
    beta : float
        2 / (v^T * v)
    """
    n = len(x)
    sigma = np.dot(x, x)
    v = np.zeros(n)
    v[0] = 1.0

    if sigma < 1e-30:
        return v, 0.0

    norm_x = math.sqrt(sigma)

    if x[0] > 0:
        v[0] = x[0] + norm_x
    else:
        v[0] = x[0] - norm_x

    # 其余分量
    v[1:] = x[1:]

    # beta = 2 / (v^T * v)
    vtv = np.dot(v, v)
    if vtv < 1e-30:
        return np.zeros(n), 0.0

    beta = 2.0 / vtv
    return v, beta


def qr_factorization(A):
    """
    QR 分解: A = Q*R。
    融合项目 938: dqrdc.m 的 QR 分解。

    使用 Householder 反射, 复杂度 O(m*n²)。

    Parameters
    ----------
    A : ndarray, shape (m, n)
        输入矩阵 (m ≥ n)

    Returns
    -------
    Q : ndarray, shape (m, m)
        正交矩阵
    R : ndarray, shape (m, n)
        上三角矩阵
    """
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)

    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        v, beta = householder_reflection(x)

        if beta < 1e-30:
            continue

        # 更新 R: R[k:, k:] -= beta * v * (v^T * R[k:, k:])
        vR = v @ R[k:, k:]
        R[k:, k:] -= beta * np.outer(v, vR)

        # 更新 Q: Q[:, k:] -= beta * (Q[:, k:] * v) * v^T
        Qv = Q[:, k:] @ v
        Q[:, k:] -= beta * np.outer(Qv, v)

    return Q, R


def qr_solve_linear(A, b):
    """
    使用 QR 分解求解线性系统 A*x = b。
    融合项目 938: qr_factor_solve.m

    对于超定系统 (m > n), 返回最小二乘解。
    对于方阵系统, 返回精确解。

    Parameters
    ----------
    A : ndarray, shape (m, n)
        系数矩阵
    b : ndarray, shape (m,)
        右端向量

    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    residual : float
        残差范数 ||Ax - b||
    """
    Q, R = qr_factorization(A)
    m, n = A.shape

    # Q^T * b
    Qtb = Q.T @ b

    # 回代求解 R * x = Q^T * b (前 n 个分量)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-14:
            x[i] = 0.0  # 奇异
        else:
            x[i] = (Qtb[i] - np.dot(R[i, i+1:n], x[i+1:n])) / R[i, i]

    residual = float(np.linalg.norm(A @ x - b))
    return x, residual


def qr_eigenvalue_iteration(A, max_iter=200, tol=1e-12, use_shift=True):
    """
    QR 迭代法求矩阵的所有特征值。
    融合项目 938 + 1405 的矩阵谱分析。

    算法: A_0 = A
    for k = 0, 1, ...
      A_k - μ_k*I = Q_k * R_k  (QR 分解)
      A_{k+1} = R_k * Q_k + μ_k*I

    收敛时, A_k → 上三角 (Schur 形式), 对角线 = 特征值。

    位移 μ_k 加速收敛:
    - Wilkinson 位移: μ = A_k[n,n] (最后一个对角元)

    Parameters
    ----------
    A : ndarray, shape (n, n)
        方阵
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差
    use_shift : bool
        是否使用位移加速

    Returns
    -------
    eigenvalues : ndarray
        特征值 (可能为复数)
    Q_total : ndarray
        累积正交变换矩阵 (特征向量)
    n_iterations : int
        实际迭代次数
    """
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.eye(0), 0

    A_k = A.astype(float).copy()
    Q_total = np.eye(n)

    for iteration in range(max_iter):
        # 检查收敛: 次对角线元素是否足够小
        off_diag = 0.0
        for i in range(n - 1):
            off_diag += abs(A_k[i+1, i])

        if off_diag < tol * (np.linalg.norm(A_k, 'fro') + 1e-14):
            break

        # 位移
        if use_shift:
            mu = A_k[n-1, n-1]
        else:
            mu = 0.0

        # QR 分解: A_k - μ*I = Q*R
        A_shifted = A_k - mu * np.eye(n)
        Q, R = qr_factorization(A_shifted)

        # 更新: A_{k+1} = R*Q + μ*I
        A_k = R @ Q + mu * np.eye(n)

        # 累积 Q
        Q_total = Q_total @ Q

    # 提取特征值 (对角线, 需要拷贝因为要修改)
    eigenvalues = np.diag(A_k).copy().astype(complex)

    # 检查 2x2 块 (复数特征值)
    for i in range(n - 1):
        if abs(A_k[i+1, i]) > tol * 10:
            # 2x2 块特征值
            a, b = A_k[i, i], A_k[i, i+1]
            c, d = A_k[i+1, i], A_k[i+1, i+1]
            trace = a + d
            det = a * d - b * c
            disc = trace * trace - 4 * det
            if disc < 0:
                eigenvalues[i] = complex(trace/2, math.sqrt(-disc)/2)
                eigenvalues[i+1] = complex(trace/2, -math.sqrt(-disc)/2)

    return eigenvalues, Q_total, iteration + 1


def hessenberg_reduction(A):
    """
    将矩阵化为上 Hessenberg 形式: H = Q^T * A * Q。

    Hessenberg 矩阵在主对角线以下只有第一次对角线非零。
    这是 QR 迭代的预处理步骤, 将 O(n³) 降到 O(n²) 每步。

    Parameters
    ----------
    A : ndarray, shape (n, n)
        输入矩阵

    Returns
    -------
    H : ndarray
        上 Hessenberg 矩阵
    Q : ndarray
        正交变换矩阵
    """
    n = A.shape[0]
    H = A.astype(float).copy()
    Q = np.eye(n)

    for k in range(n - 2):
        x = H[k+1:, k].copy()
        v, beta = householder_reflection(x)

        if beta < 1e-30:
            continue

        # H[k+1:, k:] -= beta * v * (v^T * H[k+1:, k:])
        vH = v @ H[k+1:, k:]
        H[k+1:, k:] -= beta * np.outer(v, vH)

        # H[:, k+1:] -= beta * (H[:, k+1:] * v) * v^T
        Hv = H[:, k+1:] @ v
        H[:, k+1:] -= beta * np.outer(Hv, v)

        # Q[:, k+1:] -= beta * (Q[:, k+1:] * v) * v^T
        Qv = Q[:, k+1:] @ v
        Q[:, k+1:] -= beta * np.outer(Qv, v)

    return H, Q


def markov_transition_analysis(T_matrix, max_iter=500, tol=1e-12):
    """
    Markov 转移矩阵的稳态分析。
    融合项目 1405: power_rank (PageRank 幂法迭代)。

    求稳态分布: π = T * π, Σ π_i = 1

    使用幂法: x_{k+1} = T * x_k / ||T * x_k||

    Parameters
    ----------
    T_matrix : ndarray, shape (n, n)
        列随机转移矩阵 (每列和为1)
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    stationary : ndarray
        稳态分布
    n_iterations : int
        迭代次数
    spectral_gap : float
        谱间隙 1 - |λ₂| (收敛速度)
    """
    n = T_matrix.shape[0]

    # 幂法
    x = np.ones(n) / n
    for k in range(max_iter):
        x_new = T_matrix @ x
        norm = np.sum(np.abs(x_new))
        if norm < 1e-30:
            break
        x_new /= norm

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    # 确保归一化
    total = np.sum(x)
    if total > 0:
        x /= total

    # 谱间隙估计 (通过 QR 特征值)
    eigenvalues, _, _ = qr_eigenvalue_iteration(T_matrix, max_iter=100)
    abs_eigs = np.sort(np.abs(eigenvalues))[::-1]
    spectral_gap = float(1.0 - abs_eigs[1]) if len(abs_eigs) > 1 else 1.0

    return x, k + 1, max(spectral_gap, 0.0)


def condition_number_estimate(A):
    """
    估计矩阵条件数 κ(A) = ||A|| * ||A^{-1}||。

    通过 QR 分解的对角线元素估计:
    κ(A) ≈ |R_max| / |R_min|

    Parameters
    ----------
    A : ndarray
        方阵

    Returns
    -------
    float
        条件数估计
    """
    _, R = qr_factorization(A)
    diag = np.abs(np.diag(R))
    nonzero = diag[diag > 1e-14]

    if len(nonzero) == 0:
        return float('inf')

    return float(np.max(nonzero) / np.min(nonzero))


def solve_implicit_diffusion_step(A_matrix, c_old, boundary_rhs=None):
    """
    求解隐式扩散步: A * c^{n+1} = c^n + boundary_terms。

    使用 QR 分解求解。

    Parameters
    ----------
    A_matrix : ndarray, shape (N, N)
        隐式矩阵 (I - θ*Δt*L)
    c_old : ndarray
        旧时刻浓度
    boundary_rhs : ndarray, optional
        边界条件贡献

    Returns
    -------
    c_new : ndarray
        新时刻浓度
    residual : float
        残差
    """
    rhs = c_old.copy()
    if boundary_rhs is not None:
        rhs += boundary_rhs

    c_new, residual = qr_solve_linear(A_matrix, rhs)
    return c_new, residual
