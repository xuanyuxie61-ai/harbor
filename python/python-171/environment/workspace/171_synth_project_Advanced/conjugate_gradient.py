# -*- coding: utf-8 -*-
"""
conjugate_gradient.py
=====================
多种稀疏存储格式下的共轭梯度法（CG）及预处理 CG（PCG）。

融合种子项目：
- 149_cg : R8GE/R83/R83S/R83T/R8PBU/R8SD/R8SP 格式 CG
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# 标准 CG（适用于任意 matvec 接口）
# ---------------------------------------------------------------------------

def conjugate_gradient(matvec, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    """
    标准共轭梯度法求解 Ax = b，A 为 SPD 矩阵。

    算法：
        r_0 = b - A x_0
        p_0 = r_0
        for k = 0, 1, 2, ...
            α_k = (r_k^T r_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + β_k p_k

    参数
    ----
    matvec : callable
        矩阵-向量乘法函数 matvec(v) -> A v。
    b : ndarray
        右端项。
    x0 : ndarray, optional
        初始猜测。
    max_iter : int, optional
        最大迭代次数，默认为 n。
    tol : float
        相对残差容差 ||r|| / ||b|| < tol。
    verbose : bool

    返回
    ----
    x : ndarray
        近似解。
    info : dict
        包含迭代次数、残差历史、收敛标志。
    """
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    p = r.copy()
    rs_old = float(r @ r)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(rs_old) / norm_b]

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            if verbose:
                print(f"CG break at iteration {k}: pAp too small.")
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        residual_history.append(math.sqrt(rs_new) / norm_b)

        if math.sqrt(rs_new) / norm_b < tol:
            if verbose:
                print(f"CG converged at iteration {k+1}.")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info


# ---------------------------------------------------------------------------
# 预处理 CG（PCG）
# ---------------------------------------------------------------------------

def preconditioned_cg(matvec, preconditioner, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    """
    预处理共轭梯度法（PCG）。

    算法：
        r_0 = b - A x_0
        z_0 = M^{-1} r_0
        p_0 = z_0
        for k = 0, 1, 2, ...
            α_k = (r_k^T z_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            z_{k+1} = M^{-1} r_{k+1}
            β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            p_{k+1} = z_{k+1} + β_k p_k

    preconditioner: callable, preconditioner(r) -> M^{-1} r。
    """
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = float(r @ z)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(float(r @ r)) / norm_b]

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            if verbose:
                print(f"PCG break at iteration {k}: pAp too small.")
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner(r)
        rz_new = float(r @ z)
        residual_history.append(math.sqrt(float(r @ r)) / norm_b)

        if math.sqrt(float(r @ r)) / norm_b < tol:
            if verbose:
                print(f"PCG converged at iteration {k+1}.")
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info


# ---------------------------------------------------------------------------
# 灵活 CG（FCG）- 允许预处理子每步变化
# ---------------------------------------------------------------------------

def flexible_cg(matvec, preconditioner, b, x0=None, max_iter=None, tol=1e-10, verbose=False):
    """
    灵活 CG（Flexible CG），允许预处理子在每步迭代中变化。
    适用于非线性预处理子或不完全固定预处理子。

    参考：Notay, "Flexible Conjugate Gradients", SIAM J. Sci. Comput. 2000。
    """
    b = np.asarray(b, dtype=float).flatten()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).flatten().copy()
    if max_iter is None:
        max_iter = n

    r = b - matvec(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = float(r @ z)
    norm_b = math.sqrt(float(b @ b))
    if norm_b < 1e-30:
        norm_b = 1.0

    residual_history = [math.sqrt(float(r @ r)) / norm_b]
    z_list = [z.copy()]
    Az_list = []

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = preconditioner(r)
        z_list.append(z.copy())
        Az_list.append(Ap.copy())

        residual_history.append(math.sqrt(float(r @ r)) / norm_b)
        if math.sqrt(float(r @ r)) / norm_b < tol:
            break

        # FCG 更新（Gram-Schmidt 正交化）
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {
        'iterations': len(residual_history) - 1,
        'residual_history': residual_history,
        'final_residual': residual_history[-1],
        'converged': residual_history[-1] < tol
    }
    return x, info


# ---------------------------------------------------------------------------
# 重启 CG（适合大规模问题，周期性重新启动以减少舍入误差累积）
# ---------------------------------------------------------------------------

def restarted_cg(matvec, b, restart=50, max_iter=1000, tol=1e-10, verbose=False):
    """
    重启 CG：每 restart 步后以当前解为初始猜测重新启动标准 CG。
    """
    b = np.asarray(b, dtype=float).flatten()
    x = np.zeros(b.size, dtype=float)
    total_iter = 0
    full_history = []

    while total_iter < max_iter:
        x, info = conjugate_gradient(matvec, b, x0=x, max_iter=restart, tol=tol, verbose=False)
        total_iter += info['iterations']
        full_history.extend(info['residual_history'])
        if info['converged']:
            break
        if info['iterations'] == 0:
            break

    info = {
        'iterations': total_iter,
        'residual_history': full_history,
        'final_residual': full_history[-1] if full_history else 1.0,
        'converged': full_history[-1] < tol if full_history else False
    }
    return x, info


# ---------------------------------------------------------------------------
# 多格式封装接口
# ---------------------------------------------------------------------------

def solve_with_format(fmt, data, b, extra=None, preconditioner=None, tol=1e-10, max_iter=None):
    """
    统一接口：给定稀疏格式和数据，求解 Ax = b。

    fmt: 'ge', 'r83', 'r83s', 'r83t', 'pbu', 'sd', 'sp'
    data: 矩阵数据
    b: 右端项
    extra: 额外参数字典
    preconditioner: 预处理函数或 None
    """
    from sparse_matrix import SparseMatrixOperator

    if extra is None:
        extra = {}
    n = len(b)
    op = SparseMatrixOperator(fmt, data, n, extra)
    matvec = op.matvec

    if preconditioner is None:
        x, info = conjugate_gradient(matvec, b, max_iter=max_iter, tol=tol)
    else:
        x, info = preconditioned_cg(matvec, preconditioner, b, max_iter=max_iter, tol=tol)
    return x, info
