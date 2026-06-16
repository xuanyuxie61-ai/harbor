# -*- coding: utf-8 -*-
"""
retrieval_solver.py
======================================================================
光谱反演求解器 —— Gauss 消元 + Levenberg-Marquardt

物理背景:
    系外行星大气光谱反演是一个非线性最小二乘问题:
        min_{x} || F(x) - y_obs ||_2^2 + alpha R(x)              (1)
    其中:
        x : 大气参数向量 (T_eq, log g, C/O, Fe/H, log Kzz, ...)
        F : 正演算子 (辐射传输模型)
        y_obs : 观测光谱数据
        alpha : 正则化系数 (Tikhonov 正则化)
        R(x) : 正则化项

    每次 Gauss-Newton 迭代需求解线性系统:
        (J^T J + alpha I) delta x = J^T (y_obs - F(x))         (2)
    其中 J 为 Jacobian 矩阵 dF/dx。

    本模块采用 Gauss 消元法 (来自 337_eros) 求解线性系统,
    并实现 PLU 分解、行列式计算 (用于检测病态性)。

    Jacobian 矩阵通过有限差分近似:
        J_{ij} = (F_i(x + e_j h_j) - F_i(x)) / h_j            (3)

数学公式:
    Levenberg-Marquardt 更新:
        x_{k+1} = x_k - (J^T J + lambda diag(J^T J))^{-1} J^T r_k
    其中 r_k = y_obs - F(x_k)。

    收敛判据:
        ||delta x|| / ||x|| < tol    或    ||r_k|| < tol

依赖: numpy, atmospheric_model, radiative_transfer, opacity_engine
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable


# ============================================================
# Gauss 消元求解器 (移植自 337_eros)
# ============================================================
def gauss_elimination_solve(
    A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    带部分主元选取的 Gauss 消元法求解 A x = b。
    移植自 337_eros 的 gauss() 函数。

    边界处理:
        - 主元为零时抛出异常
        - 接近奇异矩阵给出警告
    """
    m, n = A.shape
    if m != n:
        raise ValueError(f"矩阵必须为方阵, 当前 {m}x{n}")
    if len(b) != m:
        raise ValueError(f"右端项长度 {len(b)} 与矩阵维度 {m} 不匹配")

    Ab = np.hstack([A.astype(np.float64, copy=True),
                    b.astype(np.float64, copy=True).reshape(-1, 1)])

    for j in range(n):
        # 部分主元 (来自 337_eros 的 pivot)
        p = j + int(np.argmax(np.abs(Ab[j:n, j])))
        if abs(Ab[p, j]) < 1.0e-30:
            raise ValueError(f"Gauss 消元: 第 {j} 列主元为零 (矩阵奇异)")
        if p != j:
            Ab[[j, p]] = Ab[[p, j]]

        # 缩放 (来自 337_eros 的 scale)
        Ab[j] = Ab[j] / Ab[j, j]

        # 消去 (来自 337_eros 的 elim)
        for i in range(n):
            if i != j:
                Ab[i] = Ab[i] - Ab[i, j] * Ab[j]

    return Ab[:, n]


def gauss_plu_decomposition(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PLU 分解 (移植自 337_eros 的 gauss_plu)。
    A = P^T L U
    """
    m, n = A.shape
    P = np.eye(m, dtype=np.float64)
    L = np.eye(m, dtype=np.float64)
    U = A.astype(np.float64, copy=True)

    for j in range(min(m, n) - 1):
        v = np.max(np.abs(U[j:m, j]))
        if v < 1.0e-30:
            continue
        p = j + int(np.argmax(np.abs(U[j:m, j])))

        if p != j:
            P[[j, p]] = P[[p, j]]
            U[[j, p]] = U[[p, j]]
            L[j, :j], L[p, :j] = L[p, :j].copy(), L[j, :j].copy()

        for i in range(j + 1, m):
            s = U[i, j] / U[j, j]
            U[i] = U[i] - s * U[j]
            L[i, j] = s

    return P, L, U


def gauss_determinant(A: np.ndarray) -> float:
    """
    行列式计算 (移植自 337_eros 的 gauss_det)。
    det(A) = prod(diag(U)) * (-1)^{n_swaps}
    """
    _, _, U = gauss_plu_decomposition(A)
    return float(np.prod(np.diag(U)))


def condition_number_estimate(A: np.ndarray) -> float:
    """
    矩阵条件数估计。
    cond(A) = ||A|| * ||A^{-1}||
    使用 1-范数。
    """
    try:
        A_inv = gauss_elimination_solve(A, np.eye(len(A)))
        return float(np.max(np.sum(np.abs(A), axis=0)) *
                     np.max(np.sum(np.abs(A_inv), axis=0)))
    except ValueError:
        return np.inf


# ============================================================
# Jacobian 计算
# ============================================================
def compute_jacobian(
    forward_model: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    h_rel: float = 1.0e-4,
) -> np.ndarray:
    """
    通过中心差分计算 Jacobian 矩阵。
    J_{ij} = (F_i(x + h_j e_j) - F_i(x - h_j e_j)) / (2 h_j)
    """
    f0 = forward_model(x0)
    n_obs = len(f0)
    n_params = len(x0)
    J = np.zeros((n_obs, n_params), dtype=np.float64)

    for j in range(n_params):
        h_j = max(abs(x0[j]) * h_rel, 1.0e-8)
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += h_j
        x_minus[j] -= h_j

        f_plus = forward_model(x_plus)
        f_minus = forward_model(x_minus)

        J[:, j] = (f_plus - f_minus) / (2.0 * h_j)

    return J


# ============================================================
# Levenberg-Marquardt 反演
# ============================================================
def retrieve_atmospheric_parameters(
    forward_model: Callable[[np.ndarray], np.ndarray],
    y_obs: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 30,
    tol: float = 1.0e-6,
    lambda_init: float = 1.0,
    alpha_reg: float = 1.0e-4,
) -> Dict:
    """
    Levenberg-Marquardt 反演算法。

    输入:
        forward_model : 正演模型 F(x)
        y_obs         : 观测光谱
        x0            : 初始猜测
        max_iter      : 最大迭代次数
        tol           : 收敛容差
        lambda_init   : 初始 LM 参数
        alpha_reg     : Tikhonov 正则化系数

    输出:
        result['x']         : 反演结果
        result['history']   : 残差历史
        result['jacobian']  : 最终 Jacobian
        result['converged'] : 是否收敛
    """
    x = x0.copy()
    lam = lambda_init
    history = []

    for it in range(max_iter):
        f_pred = forward_model(x)
        r = y_obs - f_pred
        residual_norm = np.linalg.norm(r)
        history.append(residual_norm)

        if residual_norm < tol:
            return {
                "x": x,
                "history": history,
                "converged": True,
                "n_iter": it,
                "final_residual": residual_norm,
            }

        J = compute_jacobian(forward_model, x)
        JtJ = J.T @ J
        Jtr = J.T @ r

        # Tikhonov 正则化
        JtJ_reg = JtJ + alpha_reg * np.eye(len(x))

        # LM 修正
        diag_JtJ = np.diag(np.diag(JtJ)) + 1.0e-10 * np.eye(len(x))
        A = JtJ_reg + lam * diag_JtJ

        try:
            delta_x = gauss_elimination_solve(A, Jtr)
        except ValueError:
            lam = lam * 10.0
            if lam > 1.0e10:
                break
            continue

        x_new = x + delta_x
        f_new = forward_model(x_new)
        r_new = y_obs - f_new
        new_residual = np.linalg.norm(r_new)

        if new_residual < residual_norm:
            x = x_new
            lam = max(lam / 3.0, 1.0e-8)
        else:
            lam = min(lam * 5.0, 1.0e10)

        rel_change = np.linalg.norm(delta_x) / max(np.linalg.norm(x), 1.0e-30)
        if rel_change < tol * 0.1:
            break

    f_final = forward_model(x)
    return {
        "x": x,
        "history": history,
        "converged": False,
        "n_iter": max_iter,
        "final_residual": np.linalg.norm(y_obs - f_final),
    }
