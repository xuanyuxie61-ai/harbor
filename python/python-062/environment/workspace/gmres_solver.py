"""
gmres_solver.py
================================================================================
GMRES 迭代求解器模块 —— 基于种子项目 473_gmres

在不可压 LES 中，压力泊松方程的求解是计算瓶颈：
    ∇²p = -ρ ∂²(u_i u_j)/∂x_i∂x_j

本模块提供 Generalized Minimum Residual (GMRES) 算法，
配合 Arnoldi 迭代与 Givens 旋转，用于求解大型稀疏线性系统 A x = b。

核心物理公式
--------------------------------------------------------------------------------
压力泊松方程（源于不可压 Navier-Stokes 的连续性约束 ∂u_i/∂x_i = 0）：
    ∇²p = ρ ∂u_i/∂x_j · ∂u_j/∂x_i

在离散形式下，这形成一个大型对称正定（或近对称）线性系统。
GMRES 通过最小化残差在 Krylov 子空间中的范数来求解：
    x_k = argmin_{x ∈ x0 + K_k} ||b - A x||_2

其中 K_k = span{r0, A r0, A² r0, ..., A^{k-1} r0} 为 Krylov 子空间。
"""

import numpy as np


def givens_rotation(v1, v2):
    """
    计算 Givens 旋转参数 (cs, sn) 使得：
        [ cs  sn ] [ v1 ]   [ r ]
        [-sn  cs ] [ v2 ] = [ 0 ]
    """
    if abs(v1) < 1e-15:
        cs = 0.0
        sn = 1.0
    else:
        t = np.sqrt(v1**2 + v2**2)
        cs = abs(v1) / t
        sn = cs * v2 / v1
    return cs, sn


def apply_givens_rotation(h, cs, sn, k):
    """
    对 H 的第 k 列应用累积的 Givens 旋转。

    参数
    ----------
    h : np.ndarray, shape (k+1,)
        H 矩阵的第 k 列
    cs, sn : np.ndarray
        历史旋转参数
    k : int
        当前迭代步

    返回
    -------
    h : np.ndarray
        更新后的列
    cs_k, sn_k : float
        新旋转参数
    """
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp

    cs_k, sn_k = givens_rotation(h[k], h[k + 1])
    h[k] = cs_k * h[k] + sn_k * h[k + 1]
    h[k + 1] = 0.0
    return h, cs_k, sn_k


def arnoldi_iteration(A_func, Q, k):
    """
    Arnoldi 迭代生成正交基与上 Hessenberg 矩阵元素。

    参数
    ----------
    A_func : callable
        矩阵-向量乘法函数 A_func(v)
    Q : np.ndarray, shape (n, k+1)
        Krylov 正交基
    k : int
        当前迭代步（0-based）

    返回
    -------
    h : np.ndarray, shape (k+2,)
        Hessenberg 矩阵元素
    q : np.ndarray, shape (n,)
        新的正交基向量
    """
    n = Q.shape[0]
    q = A_func(Q[:, k])
    h = np.zeros(k + 2, dtype=np.float64)

    for i in range(k + 1):
        h[i] = np.dot(q, Q[:, i])
        q = q - h[i] * Q[:, i]

    h[k + 1] = np.linalg.norm(q)
    if h[k + 1] < 1e-15:
        q = np.zeros(n, dtype=np.float64)
    else:
        q = q / h[k + 1]

    return h, q


def gmres_solve(A_func, b, x0=None, max_iter=50, tol=1e-8, restart=None):
    """
    GMRES 求解器（重启型）。

    参数
    ----------
    A_func : callable
        矩阵-向量乘法
    b : np.ndarray, shape (n,)
        右端项
    x0 : np.ndarray, optional
        初始猜测
    max_iter : int
        每轮最大迭代次数
    tol : float
        残差容差
    restart : int, optional
        重启周期，默认为 max_iter

    返回
    -------
    x : np.ndarray
        解
    residuals : list
        残差历史
    converged : bool
    """
    n = len(b)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return np.zeros(n), [0.0], True

    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    if restart is None:
        restart = max_iter

    x = np.copy(x0)
    residuals = []

    for outer in range(max_iter // restart + 1):
        r = b - A_func(x)
        r_norm = np.linalg.norm(r)
        error = r_norm / b_norm
        residuals.append(error)

        if error <= tol:
            return x, residuals, True

        m = min(restart, max_iter - outer * restart)

        Q = np.zeros((n, m + 1), dtype=np.float64)
        Q[:, 0] = r / r_norm

        H = np.zeros((m + 1, m), dtype=np.float64)
        cs = np.zeros(m, dtype=np.float64)
        sn = np.zeros(m, dtype=np.float64)
        e1 = np.zeros(m + 1, dtype=np.float64)
        e1[0] = r_norm
        beta = np.copy(e1)

        for k in range(m):
            h_col, q_new = arnoldi_iteration(A_func, Q, k)
            H[:k + 2, k] = h_col
            Q[:, k + 1] = q_new

            h_col[:k + 2], cs[k], sn[k] = apply_givens_rotation(
                h_col[:k + 2].copy(), cs, sn, k)
            H[:k + 2, k] = h_col[:k + 2]

            beta[k + 1] = -sn[k] * beta[k]
            beta[k] = cs[k] * beta[k]
            error = abs(beta[k + 1]) / b_norm
            residuals.append(error)

            if error <= tol:
                # 求解上三角系统
                y = np.linalg.solve(H[:k + 1, :k + 1], beta[:k + 1])
                x = x + Q[:, :k + 1] @ y
                return x, residuals, True

        # 未收敛，更新解并重启
        y = np.linalg.solve(H[:m, :m], beta[:m])
        x = x + Q[:, :m] @ y

    return x, residuals, False


def build_poisson_matrix(nx, ny, nz, dx, dy, dz):
    """
    构建标准 7 点离散 Laplacian 矩阵（用于压力泊松方程）。

    参数
    ----------
    nx, ny, nz : int
        网格点数
    dx, dy, dz : float
        网格间距

    返回
    -------
    A_dense : np.ndarray, shape (N, N)
        稠密矩阵（仅用于小规模测试）
    """
    N = nx * ny * nz
    A = np.zeros((N, N), dtype=np.float64)

    def idx(i, j, k):
        return i + nx * (j + ny * k)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = idx(i, j, k)
                coeff = 0.0

                if i > 0:
                    A[n, idx(i - 1, j, k)] = 1.0 / dx**2
                    coeff += 1.0 / dx**2
                if i < nx - 1:
                    A[n, idx(i + 1, j, k)] = 1.0 / dx**2
                    coeff += 1.0 / dx**2
                if j > 0:
                    A[n, idx(i, j - 1, k)] = 1.0 / dy**2
                    coeff += 1.0 / dy**2
                if j < ny - 1:
                    A[n, idx(i, j + 1, k)] = 1.0 / dy**2
                    coeff += 1.0 / dy**2
                if k > 0:
                    A[n, idx(i, j, k - 1)] = 1.0 / dz**2
                    coeff += 1.0 / dz**2
                if k < nz - 1:
                    A[n, idx(i, j, k + 1)] = 1.0 / dz**2
                    coeff += 1.0 / dz**2

                A[n, n] = -coeff

    # 正则化：固定中心点压力为 0，使矩阵非奇异
    # 对于 Neumann 边界的 Poisson 方程，需去除零空间
    center = N // 2
    A[center, :] = 0.0
    A[center, center] = 1.0

    return A
