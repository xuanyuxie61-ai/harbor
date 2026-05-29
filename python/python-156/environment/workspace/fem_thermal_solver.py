"""
fem_thermal_solver.py
=====================
基于有限元法（FEM, piecewise linear basis）的1D稳态火焰面温度场求解器。

核心算法源自 fem1d_heat_steady (Project 392)，并改造用于求解火焰面方程：

    - d/dZ [ κ(Z) dT/dZ ] = S_T(Z)

其中 κ(Z) = ρ(Z) χ(Z) / 2 为等效扩散系数，
S_T(Z) = Q * ω̇_F / (ρ c_p) 为反应热源项。

有限元弱形式推导：
-----------------
乘以测试函数 V(I, Z) 并在 [0, 1] 上积分：

    ∫₀¹ κ(Z) T'(Z) V_I'(Z) dZ = ∫₀¹ S_T(Z) V_I(Z) dZ

对于线性基函数，V_I 仅在单元 [Z_{I-1}, Z_I] 和 [Z_I, Z_{I+1}] 上非零，
因此刚度矩阵为三对角矩阵。

数值积分采用 2 点 Gauss-Legendre 求积：
    ξ₁ = -1/√3,  ξ₂ = +1/√3,  w₁ = w₂ = 1.0

离散化后得到线性方程组 A T = b，其中：
    A_{I,I-1} = ∫ κ V_{I-1}' V_I' dZ
    A_{I,I}   = ∫ κ V_I' V_I' dZ
    A_{I,I+1} = ∫ κ V_{I+1}' V_I' dZ
    b_I       = ∫ S_T V_I dZ

边界条件：
    T(0) = T_ox,  T(1) = T_fuel
"""

import numpy as np
from flamelet_core import (
    scalar_dissipation_rate,
    density_mixture,
    reaction_rate_one_step,
    flamelet_boundary_conditions,
    HEAT_RELEASE,
)


def kappa_func(Z, T, chi_st):
    """
    计算等效扩散系数 κ(Z) = ρ(Z) χ(Z) / 2。
    """
    chi = scalar_dissipation_rate(Z, chi_st)
    rho = density_mixture(Z, T)
    return np.maximum(rho * chi / 2.0, 1.0e-12)


def source_func(Z, T, Y_F, Y_O, chi_st):
    """
    计算温度源项 S_T(Z)。

    采用高斯型热源近似（基于 Arrhenius 反应速率的参数化）：
        S_T(Z) = S_max * exp[ -(Z - Z_st)² / (2 σ²) ] * (1 - T/T_ad)_+

    其中 S_max 由典型 Arrhenius 反应速率估算：
        S_max ≈ Q * A * ρ² * Y_F,st * Y_O,st * exp(-E_a / R_u T_st) / (ρ c_p)

    温度自限因子 (1 - T/T_ad)_+ 确保温度不超过绝热火焰温度。
    """
    # TODO: Hole 2 - 实现温度方程源项 S_T(Z)
    # 需要基于 Arrhenius 反应速率构造高斯型热源近似：
    #   S_T(Z) = S_max * exp[-(Z - Z_st)^2 / (2 * sigma^2)] * (1 - T/T_ad)_+
    # 其中 S_max 由典型条件下的反应速率估算：
    #   S_max = Q * omega_peak / (rho_ref * cp)
    # 温度自限因子 (1 - T/T_ad)_+ 确保温度不超过绝热火焰温度
    # 返回温度源项 source，单位 K/s
    raise NotImplementedError("Hole 2: 请实现 source_func 函数")


def solve_fem_thermal(n, Z_nodes, T_init, Y_F_init, Y_O_init, chi_st,
                      tol=1.0e-10, max_iter=100):
    """
    使用有限元法求解稳态火焰面温度分布 T(Z)。

    Parameters
    ----------
    n : int
        节点数，必须 >= 3。
    Z_nodes : ndarray, shape (n,)
        混合分数空间离散节点，单调递增，Z[0]=0, Z[-1]=1。
    T_init : ndarray, shape (n,)
        温度初始猜测。
    Y_F_init, Y_O_init : ndarray, shape (n,)
        燃料/氧化剂质量分数初始分布。
    chi_st : float
        化学计量点标量耗散率。
    tol : float
        非线性迭代容差。
    max_iter : int
        最大非线性迭代次数。

    Returns
    -------
    T : ndarray, shape (n,)
        收敛后的温度分布。
    iter_count : int
        实际迭代次数。
    """
    if n < 3:
        raise ValueError("节点数 n 必须 >= 3")
    if Z_nodes[0] != 0.0 or Z_nodes[-1] != 1.0:
        raise ValueError("Z_nodes 必须满足 Z[0]=0, Z[-1]=1")
    if not np.all(np.diff(Z_nodes) > 0):
        raise ValueError("Z_nodes 必须严格单调递增")

    bc = flamelet_boundary_conditions()
    T = np.array(T_init, dtype=float)
    Y_F = np.array(Y_F_init, dtype=float)
    Y_O = np.array(Y_O_init, dtype=float)

    # 2-point Gauss-Legendre quadrature
    abscissa = np.array([-0.5773502691896258, 0.5773502691896258])
    weight = np.array([1.0, 1.0])
    quad_num = 2

    for iteration in range(max_iter):
        T_old = T.copy()
        Amat = np.zeros((n, n))
        bvec = np.zeros(n)

        # 左边界条件
        Amat[0, 0] = 1.0
        bvec[0] = bc['T_left']

        # 内部节点
        for i in range(1, n - 1):
            xl = Z_nodes[i - 1]
            xm = Z_nodes[i]
            xr = Z_nodes[i + 1]

            al = 0.0
            am = 0.0
            ar = 0.0
            bm = 0.0

            for q in range(quad_num):
                # 左侧单元 [xl, xm]
                xq = ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xm) / 2.0
                wq = weight[q] * (xm - xl) / 2.0

                # 线性基函数导数
                vlp = -1.0 / (xm - xl)
                vmp = 1.0 / (xm - xl)

                # 使用上一迭代的温度估计 κ 和源项
                Tq = np.interp(xq, Z_nodes, T_old)
                Y_Fq = np.interp(xq, Z_nodes, Y_F)
                Y_Oq = np.interp(xq, Z_nodes, Y_O)

                kxq = kappa_func(xq, Tq, chi_st)
                fxq = source_func(xq, Tq, Y_Fq, Y_Oq, chi_st)

                vl = (xm - xq) / (xm - xl)
                vm = (xq - xl) / (xm - xl)

                al += wq * kxq * vlp * vmp
                am += wq * kxq * vmp * vmp
                bm += wq * fxq * vm

                # 右侧单元 [xm, xr]
                xq = ((1.0 - abscissa[q]) * xm + (1.0 + abscissa[q]) * xr) / 2.0
                wq = weight[q] * (xr - xm) / 2.0

                vmp = -1.0 / (xr - xm)
                vrp = 1.0 / (xr - xm)

                Tq = np.interp(xq, Z_nodes, T_old)
                Y_Fq = np.interp(xq, Z_nodes, Y_F)
                Y_Oq = np.interp(xq, Z_nodes, Y_O)

                kxq = kappa_func(xq, Tq, chi_st)
                fxq = source_func(xq, Tq, Y_Fq, Y_Oq, chi_st)

                vm = (xr - xq) / (xr - xm)

                am += wq * kxq * vmp * vmp
                ar += wq * kxq * vrp * vmp
                bm += wq * fxq * vm

            Amat[i, i - 1] = al
            Amat[i, i] = am
            Amat[i, i + 1] = ar
            bvec[i] = bm

        # 右边界条件
        Amat[n - 1, n - 1] = 1.0
        bvec[n - 1] = bc['T_right']

        # 求解线性系统
        T_new = np.linalg.solve(Amat, bvec)

        # 松弛迭代（防止热失控）
        relaxation = 0.3
        T = relaxation * T_new + (1.0 - relaxation) * T_old

        # 边界处理：温度不能低于入口温度，不能过高
        T = np.clip(T, bc['T_left'], 3000.0)

        max_change = np.max(np.abs(T - T_old))
        if max_change < tol:
            return T, iteration + 1

    return T, max_iter
