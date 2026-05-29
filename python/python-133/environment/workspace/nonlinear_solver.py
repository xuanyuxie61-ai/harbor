"""
nonlinear_solver.py
====================
非线性稀疏方程组求解器与 Jacobian 正交性检验

基于种子项目 871_plasma_matrix 与 034_asa082 融合重构。

科学背景：
---------
聚合反应器中的非理想行为（如凝胶效应、相分离）导致
反应-扩散方程呈现强非线性。本模块实现基于 Newton-Krylov
方法的非线性求解器，并引入 plasma_matrix 的稀疏矩阵
组装技术处理二维反应器截面上的非线性泊松型方程：

    -∇·(D(u) ∇u) + R(u) = f

其中扩散系数 D(u) 可能依赖于局部转化率（如凝胶效应导致
的扩散控制终止：D(u) = D_0 (1-u)^{2.5}）。

非线性系统的 Newton 迭代：
    J(u^k) δu = -F(u^k)
    u^{k+1} = u^k + α δu

其中 Jacobian J = ∂F/∂u 的稀疏组装借鉴 plasma_matrix.m
的五点差分模板。

正交性检验（基于 detq.m）：
    在 Newton 迭代中，若 Jacobian 条件数过大，可通过
    QR 分解检验列向量的正交性。正交矩阵行列式 |det(Q)| = 1，
    若偏离过大则提示数值不稳定。
"""

import numpy as np
from typing import Callable, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator


def detq_orthogonal(a: np.ndarray) -> Tuple[float, int]:
    """
    计算（近似）正交矩阵的行列式，并检验正交性。
    基于 detq.m 的算法（ASA 82）。

    对于正交矩阵，|det| = 1。若结果显著偏离 1，则矩阵不正交。

    算法核心：
      利用 Householder 变换将矩阵化为上三角形式，
      行列式为对角线元素的乘积（符号由反射决定）。
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    tol = 1.0e-4
    ifault = 0

    if n <= 0:
        return 0.0, 1

    # 将矩阵展平为一维存储（按列优先兼容 MATLAB）
    a2 = a.flatten(order='F').copy()
    d = 1.0
    r_idx = 0

    for k in range(1, n + 1):
        q = r_idx
        x = a2[q]
        y = np.sign(x)
        d *= y
        y = -1.0 / (x + y)
        x = abs(x) - 1.0

        if tol < abs(x):
            if 0.0 < x:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault

            for i in range(k, n):
                q += n
                x_val = a2[q] * y
                p = r_idx
                s = q
                for j in range(k, n):
                    p += 1
                    s += 1
                    a2[s] += x_val * a2[p]

        r_idx += n + 1

    return d, ifault


def assemble_sparse_jacobian_2d(n: int,
                                 u: np.ndarray,
                                 diff_func: Callable[[np.ndarray], np.ndarray],
                                 reaction_deriv: Callable[[np.ndarray], np.ndarray],
                                 xleft: float = -1.0,
                                 xright: float = 1.0) -> csr_matrix:
    """
    组装二维非线性反应-扩散方程的稀疏 Jacobian 矩阵。
    基于 plasma_matrix.m 的五点差分模板。

    方程：
        -∇·(D(u) ∇u) + R'(u) u = f

    在 n×n 均匀网格上离散，共 numnodes = n² 个未知数。
    每个内部节点 (i,j) 的 stencil：
        (i,j-1)   (i,j)   (i,j+1)
        (i-1,j)           (i+1,j)

    非线性扩散系数 D(u) 在面心取值。
    """
    numnodes = n * n
    x = np.linspace(xleft, xright, n)
    h = x[1] - x[0]
    h2 = h * h

    D = diff_func(u)
    Rprime = reaction_deriv(u)

    row_idx = []
    col_idx = []
    data = []

    def add_entry(r, c, val):
        row_idx.append(r)
        col_idx.append(c)
        data.append(float(val))

    k = 0
    for i in range(n):
        for j in range(n):
            # 扩散系数在相邻面的均值
            D_e = D[k] if j == n - 1 else 0.5 * (D[k] + D[k + 1])
            D_w = D[k] if j == 0 else 0.5 * (D[k] + D[k - 1])
            D_n = D[k] if i == n - 1 else 0.5 * (D[k] + D[k + n])
            D_s = D[k] if i == 0 else 0.5 * (D[k] + D[k - n])

            # 中心系数（考虑边界修正）
            center = 0.0
            if j > 0:
                center += D_w / h2
                add_entry(k, k - 1, -D_w / h2)
            else:
                # Neumann 边界：镜像
                center += D_w / h2

            if j < n - 1:
                center += D_e / h2
                add_entry(k, k + 1, -D_e / h2)
            else:
                center += D_e / h2

            if i > 0:
                center += D_s / h2
                add_entry(k, k - n, -D_s / h2)
            else:
                center += D_s / h2

            if i < n - 1:
                center += D_n / h2
                add_entry(k, k + n, -D_n / h2)
            else:
                center += D_n / h2

            center += Rprime[k]
            add_entry(k, k, center)
            k += 1

    J = csr_matrix((data, (row_idx, col_idx)), shape=(numnodes, numnodes))
    return J


def nonlinear_residual_2d(n: int,
                          u: np.ndarray,
                          diff_func: Callable[[np.ndarray], np.ndarray],
                          reaction_func: Callable[[np.ndarray], np.ndarray],
                          source: np.ndarray,
                          xleft: float = -1.0,
                          xright: float = 1.0) -> np.ndarray:
    """
    计算二维非线性反应-扩散方程的残差向量 F(u)。

    F(u) = -∇·(D(u)∇u) + R(u) - f
    """
    numnodes = n * n
    x = np.linspace(xleft, xright, n)
    h = x[1] - x[0]
    h2 = h * h

    D = diff_func(u)
    R = reaction_func(u)

    F = np.zeros(numnodes)
    k = 0
    for i in range(n):
        for j in range(n):
            D_e = D[k] if j == n - 1 else 0.5 * (D[k] + D[k + 1])
            D_w = D[k] if j == 0 else 0.5 * (D[k] + D[k - 1])
            D_n = D[k] if i == n - 1 else 0.5 * (D[k] + D[k + n])
            D_s = D[k] if i == 0 else 0.5 * (D[k] + D[k - n])

            val = 0.0
            u_k = u[k]
            if j > 0:
                val -= D_w * (u[k - 1] - u_k) / h2
            else:
                val -= D_w * (-u_k) / h2  # Neumann

            if j < n - 1:
                val -= D_e * (u[k + 1] - u_k) / h2
            else:
                val -= D_e * (-u_k) / h2

            if i > 0:
                val -= D_s * (u[k - n] - u_k) / h2
            else:
                val -= D_s * (-u_k) / h2

            if i < n - 1:
                val -= D_n * (u[k + n] - u_k) / h2
            else:
                val -= D_n * (-u_k) / h2

            F[k] = val + R[k] - source[k]
            k += 1

    return F


def newton_krylov_solve(n: int,
                        u0: np.ndarray,
                        diff_func: Callable[[np.ndarray], np.ndarray],
                        reaction_func: Callable[[np.ndarray], np.ndarray],
                        reaction_deriv: Callable[[np.ndarray], np.ndarray],
                        source: np.ndarray,
                        xleft: float = -1.0,
                        xright: float = 1.0,
                        tol: float = 1.0e-8,
                        max_iter: int = 50,
                        alpha: float = 1.0) -> Tuple[np.ndarray, int, float]:
    """
    Newton-Krylov 方法求解二维非线性反应-扩散方程。

    每步 Newton 迭代：
        1. 组装稀疏 Jacobian J
        2. 计算残差 F
        3. 解 J δu = -F（使用 GMRES 或直接稀疏求解）
        4. 更新 u += α δu
        5. 正交性检验（可选）
    """
    u = u0.copy()
    numnodes = n * n

    for it in range(max_iter):
        F = nonlinear_residual_2d(n, u, diff_func, reaction_func, source, xleft, xright)
        res_norm = np.linalg.norm(F)

        if res_norm < tol:
            return u, it, res_norm

        J = assemble_sparse_jacobian_2d(n, u, diff_func, reaction_deriv, xleft, xright)

        # 若规模小，直接求解；若大，用 GMRES
        if numnodes <= 400:
            try:
                delta = spsolve(J, -F)
            except Exception:
                delta, info = gmres(J, -F, rtol=tol, maxiter=200)
                if info != 0:
                    # 失败时回退到阻尼最小二乘
                    delta = -F * 0.01
        else:
            delta, info = gmres(J, -F, rtol=tol, maxiter=500)
            if info != 0:
                delta = -F * 0.01

        # 线搜索阻尼
        alpha_local = alpha
        for _ in range(5):
            u_trial = u + alpha_local * delta
            F_trial = nonlinear_residual_2d(n, u_trial, diff_func, reaction_func, source, xleft, xright)
            if np.linalg.norm(F_trial) < res_norm:
                break
            alpha_local *= 0.5

        u = u + alpha_local * delta

        # 边界保护
        u = np.maximum(u, 0.0)
        u = np.minimum(u, 1.0e3)

    return u, max_iter, np.linalg.norm(nonlinear_residual_2d(n, u, diff_func, reaction_func, source, xleft, xright))


def gel_effect_diffusion(conversion: np.ndarray,
                         D0: float = 1.0e-4,
                         beta: float = 2.5,
                         c_crit: float = 0.8) -> np.ndarray:
    """
    凝胶效应（Trommsdorff 效应）导致的扩散系数下降模型：

        D(c) = D_0 * (1 - c)^{β}    for c < c_crit
        D(c) = D_0 * (1 - c_crit)^{β} * exp(-10(c-c_crit))   for c ≥ c_crit

    其中 c 为局部转化率。
    """
    c = np.asarray(conversion)
    D = np.zeros_like(c)
    mask1 = c < c_crit
    mask2 = ~mask1
    D[mask1] = D0 * ((1.0 - c[mask1]) ** beta)
    D[mask2] = D0 * ((1.0 - c_crit) ** beta) * np.exp(-10.0 * (c[mask2] - c_crit))
    D = np.maximum(D, 1.0e-8 * D0)
    return D


def nonlinear_source_reaction(conversion: np.ndarray,
                              k0: float = 1.0,
                              activation: float = 10.0) -> np.ndarray:
    """
    非线性反应源项：自催化效应

        R(c) = k_0 * c * (1-c) * (1 + activation * c²)
    """
    c = np.asarray(conversion)
    c = np.clip(c, 0.0, 1.0)
    return k0 * c * (1.0 - c) * (1.0 + activation * c ** 2)


def nonlinear_source_derivative(conversion: np.ndarray,
                                k0: float = 1.0,
                                activation: float = 10.0) -> np.ndarray:
    """
    反应源项对转化率 c 的导数：

        dR/dc = k_0 * [(1-2c)(1+α c²) + c(1-c)(2α c)]
              = k_0 * [1 - 2c + α c² - 2α c³ + 2α c² - 2α c³]
              = k_0 * [1 - 2c + 3α c² - 4α c³]
    """
    c = np.asarray(conversion)
    c = np.clip(c, 0.0, 1.0)
    return k0 * (1.0 - 2.0 * c + 3.0 * activation * c ** 2 - 4.0 * activation * c ** 3)
