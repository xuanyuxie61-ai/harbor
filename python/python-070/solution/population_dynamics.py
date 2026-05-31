"""
population_dynamics.py
鱼类种群时空动态有限元求解模块

整合算法：
1. 1D 二次有限元法（基于 fem1d_bvp_quadratic）
2. 非线性有限元法（基于 fem1d_nonlinear）

核心科学问题：
求解年龄结构或空间结构鱼类种群的稳态分布，控制方程为：
    -d/dx( a(x) du/dx ) + c(x) u = f(x)        (线性 BVP)
    -d/dx( p(x) du/dx ) + q(x)u + u*du/dx = f(x)  (非线性 BVP)

在渔业模型中：
- u(x): 年龄 x 处的种群密度 或 空间位置 x 处的生物量密度
- a(x)/p(x): 扩散/老化系数
- c(x)/q(x): 自然死亡率相关项
- f(x): 补充源项

有限元离散：
将域 [0,L] 划分为 N_e 个单元，每个单元 3 个节点（二次元）
使用 3 点 Gauss-Legendre 求积计算刚度矩阵和载荷向量
"""

import numpy as np
from utils import NumericalConfig, gauss_legendre_3point, solve_tridiagonal


def fem1d_bvp_quadratic(n, a_func, c_func, f_func, x_nodes):
    """
    使用分段二次元求解 1D 两点边值问题

    方程：
        -d/dx( a(x) du/dx ) + c(x) u = f(x),  x ∈ [x_0, x_{n-1}]
        u(x_0) = 0,  u(x_{n-1}) = 0

    有限元离散：
    单元 e 使用节点 L=2e-1, M=2e, R=2e+1（局部编号 0,1,2）
    形函数（Lagrange 二次）：
        φ_0(ξ) = ξ(ξ-1)/2
        φ_1(ξ) = 1-ξ^2
        φ_2(ξ) = ξ(ξ+1)/2
    其中 ξ ∈ [-1,1] 为参考坐标

    Parameters
    ----------
    n : int
        节点数，必须为奇数且 >= 3
    a_func : callable
        扩散系数函数 a(x)
    c_func : callable
        反应系数函数 c(x)
    f_func : callable
        源项函数 f(x)
    x_nodes : ndarray
        节点坐标数组，长度 n

    Returns
    -------
    u : ndarray, shape (n,)
        有限元解在节点处的值
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if n % 2 != 1:
        raise ValueError("n must be odd for quadratic elements")
    if len(x_nodes) != n:
        raise ValueError("x_nodes length must equal n")

    abscissa, weight = gauss_legendre_3point()

    A_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)

    e_num = (n - 1) // 2

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2

        xl = x_nodes[l]
        xm = x_nodes[m]
        xr = x_nodes[r]

        for q in range(3):
            # 映射到实际坐标
            xq = 0.5 * ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr)
            wq = weight[q] * 0.5 * (xr - xl)

            axq = a_func(xq)
            cxq = c_func(xq)
            fxq = f_func(xq)

            # 二次元形函数及其导数
            vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
            vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
            vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

            vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                  ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
            vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                  ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
            vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                  ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))

            # 组装刚度矩阵和载荷向量
            A_mat[l, l] += wq * (vlp * axq * vlp + vl * cxq * vl)
            A_mat[l, m] += wq * (vlp * axq * vmp + vl * cxq * vm)
            A_mat[l, r] += wq * (vlp * axq * vrp + vl * cxq * vr)
            b_vec[l] += wq * (vl * fxq)

            A_mat[m, l] += wq * (vmp * axq * vlp + vm * cxq * vl)
            A_mat[m, m] += wq * (vmp * axq * vmp + vm * cxq * vm)
            A_mat[m, r] += wq * (vmp * axq * vrp + vm * cxq * vr)
            b_vec[m] += wq * (vm * fxq)

            A_mat[r, l] += wq * (vrp * axq * vlp + vr * cxq * vl)
            A_mat[r, m] += wq * (vrp * axq * vmp + vr * cxq * vm)
            A_mat[r, r] += wq * (vrp * axq * vrp + vr * cxq * vr)
            b_vec[r] += wq * (vr * fxq)

    # Dirichlet 边界条件：u(0)=0, u(L)=0
    A_mat[0, :] = 0.0
    A_mat[0, 0] = 1.0
    b_vec[0] = 0.0

    A_mat[n - 1, :] = 0.0
    A_mat[n - 1, n - 1] = 1.0
    b_vec[n - 1] = 0.0

    # 求解线性系统
    u = np.linalg.solve(A_mat, b_vec)
    return u


def fem1d_nonlinear_picard_newton(n, x_nodes, p_func, q_func, f_func,
                                   nonlinear_coeff=1.0, max_iter=50, tol=1e-10):
    """
    非线性 1D FEM 求解器：Picard 迭代 + Newton 迭代

    方程：
        -d/dx( p(x) du/dx ) + q(x) u + nonlinear_coeff * u * du/dx = f(x)
        u(0) = 0, u(L) = u_L

    Picard 迭代（前 5 步）：将非线性项 u*du/dx 用旧值替代
    Newton 迭代（后续步）：求解Jacobian系统

    Parameters
    ----------
    n : int
        节点数（奇数）
    x_nodes : ndarray
        节点坐标
    p_func : callable
        扩散系数
    q_func : callable
        反应系数
    f_func : callable
        源项
    nonlinear_coeff : float
        非线性项系数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值

    Returns
    -------
    u : ndarray
        收敛解
    iterations : int
        实际迭代次数
    residual_norm : float
        最终残差范数
    """
    if n < 3 or n % 2 != 1:
        raise ValueError("n must be odd and >= 3")

    abscissa, weight = gauss_legendre_3point()
    e_num = (n - 1) // 2

    # 初始猜测：线性解
    u = np.zeros(n, dtype=float)
    u_old = u.copy()

    for it in range(max_iter):
        A_mat = np.zeros((n, n), dtype=float)
        b_vec = np.zeros(n, dtype=float)

        for e in range(e_num):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2

            xl = x_nodes[l]
            xm = x_nodes[m]
            xr = x_nodes[r]

            for q_idx in range(3):
                xq = 0.5 * ((1.0 - abscissa[q_idx]) * xl + (1.0 + abscissa[q_idx]) * xr)
                wq = weight[q_idx] * 0.5 * (xr - xl)

                pxq = p_func(xq)
                qxq = q_func(xq)
                fxq = f_func(xq)

                vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
                vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
                vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

                vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                      ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
                vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                      ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
                vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                      ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))

                # 非线性项处理
                if it < 5:
                    # Picard：使用旧值 u_old 计算 u * du/dx
                    u_old_q = u_old[l] * vl + u_old[m] * vm + u_old[r] * vr
                    du_old_q = u_old[l] * vlp + u_old[m] * vmp + u_old[r] * vrp
                    # 将非线性项移到右端
                    b_vec[l] += wq * vl * (fxq - nonlinear_coeff * u_old_q * du_old_q)
                    b_vec[m] += wq * vm * (fxq - nonlinear_coeff * u_old_q * du_old_q)
                    b_vec[r] += wq * vr * (fxq - nonlinear_coeff * u_old_q * du_old_q)

                    # 线性部分正常组装
                    for i_idx, vi, vip in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                        for j_idx, vj, vjp in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                            A_mat[i_idx, j_idx] += wq * (vip * pxq * vjp + vi * qxq * vj)
                else:
                    # Newton：线性化 u*du/dx ≈ u_old*du/dx + u*du_old/dx
                    u_old_q = u_old[l] * vl + u_old[m] * vm + u_old[r] * vr
                    du_old_q = u_old[l] * vlp + u_old[m] * vmp + u_old[r] * vrp

                    for i_idx, vi, vip in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                        for j_idx, vj, vjp in [(l, vl, vlp), (m, vm, vmp), (r, vr, vrp)]:
                            # 线性扩散和反应
                            A_mat[i_idx, j_idx] += wq * (vip * pxq * vjp + vi * qxq * vj)
                            # Newton Jacobian: d/du[u*du/dx] = du_old/dx * v_j + u_old * dv_j/dx
                            A_mat[i_idx, j_idx] += wq * nonlinear_coeff * vi * (du_old_q * vj + u_old_q * vjp)

                        b_vec[i_idx] += wq * vi * (fxq + nonlinear_coeff * u_old_q * du_old_q)

        # Dirichlet BC
        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = 0.0
        A_mat[n - 1, :] = 0.0
        A_mat[n - 1, n - 1] = 1.0
        b_vec[n - 1] = 0.0

        u = np.linalg.solve(A_mat, b_vec)

        residual = np.linalg.norm(u - u_old)
        u_old = u.copy()

        if residual < tol:
            return u, it + 1, residual

    return u, max_iter, residual


def solve_age_structured_steady_state(L_age, n_nodes, mortality_func,
                                      recruitment_rate, diffusion_age=0.1):
    """
    求解年龄结构种群稳态分布

    控制方程（McKendrick-von Foerster 稳态形式）：
        -D_a d^2N/da^2 + dN/da + M(a) N = R \delta(a-a_0)

    简化为 BVP 形式：
        -D_a d^2N/da^2 + (1 + M(a)) N = R(a)

    Parameters
    ----------
    L_age : float
        最大年龄
    n_nodes : int
        节点数（奇数）
    mortality_func : callable
        年龄相关自然死亡率 M(a)
    recruitment_rate : float
        边界补充率（a=0 处的入流）
    diffusion_age : float
        年龄扩散系数（模拟个体间年龄差异）

    Returns
    -------
    a_nodes : ndarray
        年龄节点
    N : ndarray
        年龄结构种群密度
    """
    if n_nodes % 2 == 0:
        n_nodes += 1

    a_nodes = np.linspace(0.0, L_age, n_nodes)

    def a_func(a):
        return diffusion_age

    def c_func(a):
        return 1.0 + mortality_func(a)

    def f_func(a):
        # 补充集中在低龄段
        if a < L_age * 0.05:
            return recruitment_rate / (L_age * 0.05)
        return 0.0

    N = fem1d_bvp_quadratic(n_nodes, a_func, c_func, f_func, a_nodes)
    return a_nodes, N


def l2_error_quadratic(u_exact_func, u_fem, x_nodes):
    """
    计算 FEM 解的 L2 误差

    ||u - u_h||_{L2} = sqrt( \int (u - u_h)^2 dx )
    """
    n = len(x_nodes)
    if n % 2 != 1 or n < 3:
        raise ValueError("n must be odd and >= 3")

    abscissa, weight = gauss_legendre_3point()
    e_num = (n - 1) // 2
    error_sq = 0.0

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2
        xl, xm, xr = x_nodes[l], x_nodes[m], x_nodes[r]

        for q in range(3):
            xq = 0.5 * ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr)
            wq = weight[q] * 0.5 * (xr - xl)

            vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
            vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
            vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

            u_h = u_fem[l] * vl + u_fem[m] * vm + u_fem[r] * vr
            u_ex = u_exact_func(xq)
            error_sq += wq * (u_ex - u_h) ** 2

    return np.sqrt(error_sq)
