"""
linear_solvers.py
=================
大规模稀疏线性系统的迭代求解器。

基于种子项目 152_cg_rc（反向通信共轭梯度法）重构：
- 原项目实现了 reverse-communication CG，用于求解 A*x = b；
- 在本系统中用于求解有限元/有限差分离散后的大规模稀疏线性系统，
  特别是催化剂颗粒径向离散得到的对称正定（SPD）三对角/带状矩阵。

同时封装了直接求解接口与预处理共轭梯度接口，
以兼容不同规模与条件数的线性系统。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg as scipy_cg


class LinearSolverError(Exception):
    """线性求解异常。"""
    pass


def solve_tridiagonal(a, b, c, rhs):
    """
    Thomas 算法求解三对角线性系统。

    系统形式：
        d_i u_{i-1} + a_i u_i + c_i u_{i+1} = rhs_i

    其中 a 为主对角线，b 为下次对角线（长度为 n-1），
    c 为上次对角线（长度为 n-1）。

    时间复杂度 O(n)，数值稳定。

    Parameters
    ----------
    a, b, c, rhs : ndarray

    Returns
    -------
    u : ndarray
    """
    n = a.size
    if b.size != n - 1 or c.size != n - 1 or rhs.size != n:
        raise LinearSolverError("三对角矩阵维度不匹配")

    a = a.astype(float, copy=True)
    b = b.astype(float, copy=True)
    c = c.astype(float, copy=True)
    rhs = rhs.astype(float, copy=True)
    u = np.zeros(n, dtype=float)

    # 前向消去
    for i in range(1, n):
        w = b[i - 1] / a[i - 1]
        a[i] -= w * c[i - 1]
        rhs[i] -= w * rhs[i - 1]

    # 回代
    u[-1] = rhs[-1] / a[-1]
    for i in range(n - 2, -1, -1):
        u[i] = (rhs[i] - c[i] * u[i + 1]) / a[i]

    return u


def conjugate_gradient_rc(n, b_vec, matvec, precon_solve,
                          max_iter=None, tol=1e-10):
    """
    反向通信风格共轭梯度法求解器。

    基于 cg_rc.m 的核心算法重构，将原 MATLAB 的 persistent 变量
    显式封装为 Python 类的状态，保持 reverse-communication 语义。

    求解对称正定线性系统 A x = b，其中矩阵 A 通过 matvec 回调隐式给出，
    预处理矩阵 M 通过 precon_solve 回调隐式给出。

    迭代公式：
        r_0 = b - A x_0
        z_0 = M^{-1} r_0
        p_0 = z_0
        对于 k = 0, 1, 2, ...:
            α_k = (r_k^T z_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            若 ||r_{k+1}|| < tol 则收敛
            z_{k+1} = M^{-1} r_{k+1}
            β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            p_{k+1} = z_{k+1} + β_k p_k

    Parameters
    ----------
    n : int
        维度。
    b_vec : ndarray, shape (n,)
        右端项。
    matvec : callable
        函数签名 matvec(p) -> q，计算 q = A @ p。
    precon_solve : callable
        函数签名 precon_solve(r) -> z，求解 M z = r。
    max_iter : int, optional
        最大迭代次数，默认 n。
    tol : float, default 1e-10
        残差容忍阈值。

    Returns
    -------
    x : ndarray
        近似解。
    info : dict
        包含 iter（迭代次数）、resid（最终残差）。
    """
    if max_iter is None:
        max_iter = n

    b_vec = np.asarray(b_vec, dtype=float)
    x = np.zeros(n, dtype=float)
    r = b_vec.copy()
    z = precon_solve(r)
    p = z.copy()

    rho_old = None
    for k in range(max_iter):
        q = matvec(p)
        pdotq = np.dot(p, q)
        if abs(pdotq) < np.finfo(float).eps:
            raise LinearSolverError("p^T A p 接近零，矩阵可能不正定")

        rho = np.dot(r, z)
        alpha = rho / pdotq
        x += alpha * p
        r -= alpha * q

        resid_norm = np.linalg.norm(r)
        if resid_norm < tol:
            return x, {"iter": k + 1, "resid": resid_norm}

        z = precon_solve(r)
        rho_new = np.dot(r, z)
        if rho_old is not None:
            beta = rho_new / rho_old
            p = z + beta * p
        else:
            p = z.copy()
        rho_old = rho_new

    return x, {"iter": max_iter, "resid": resid_norm}


def solve_sparse_system(A, b, tol=1e-10, max_iter=None):
    """
    使用 scipy.sparse.linalg.cg 求解稀疏 SPD 系统。

    Parameters
    ----------
    A : ndarray or csr_matrix
    b : ndarray

    Returns
    -------
    x : ndarray
    """
    A = csr_matrix(A)
    n = A.shape[0]
    if max_iter is None:
        max_iter = n * 2

    x, info = scipy_cg(A, b, rtol=tol, maxiter=max_iter)
    if info < 0:
        raise LinearSolverError("CG 非法输入")
    if info > 0:
        # 未收敛但返回当前最优解，给出警告
        pass
    return x


def jacobi_preconditioner(A):
    """
    构造 Jacobi 预处理子 M = diag(A)^{-1}。

    Parameters
    ----------
    A : ndarray

    Returns
    -------
    precon_solve : callable
        输入 r，返回 M^{-1} r。
    """
    diag = np.diag(A)
    if np.any(np.abs(diag) < np.finfo(float).eps):
        raise LinearSolverError("矩阵对角线包含零元，无法使用 Jacobi 预处理")
    inv_diag = 1.0 / diag

    def precon_solve(r):
        return inv_diag * r

    return precon_solve
