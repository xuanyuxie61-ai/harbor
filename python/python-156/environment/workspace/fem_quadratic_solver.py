"""
fem_quadratic_solver.py
=======================
基于二次有限元（piecewise quadratic elements）的1D火焰面组分输运方程求解器。

核心算法源自 fem1d_bvp_quadratic (Project 387)，并改造用于求解：

    - d/dZ [ D(Z) dY_k/dZ ] = ω̇_k / ρ

其中 D(Z) 为等效扩散系数，ω̇_k 为组分 k 的化学反应源项，
Y_k(Z) 为组分 k 的质量分数分布。

二次有限元形函数（在参考单元 [-1, 1] 上）：
    N₁(ξ) = ξ(ξ - 1) / 2
    N₂(ξ) = (1 - ξ)(1 + ξ)
    N₃(ξ) = ξ(ξ + 1) / 2

数值积分采用 3 点 Gauss-Legendre 求积：
    ξ₁ = -√(3/5),  ξ₂ = 0,  ξ₃ = +√(3/5)
    w₁ = w₃ = 5/9,  w₂ = 8/9

局部单元节点编号：
    L = 2e - 1,  M = 2e,  R = 2e + 1

刚度矩阵元素（在单元 e 上）：
    A_{ij}^{(e)} = ∫ [ D(Z) N_i'(Z) N_j'(Z) + c(Z) N_i(Z) N_j(Z) ] dZ
    b_i^{(e)} = ∫ f(Z) N_i(Z) dZ

其中 c(Z) 为可选的线性化反应项系数，f(Z) 为源项。

边界条件：
    Y_F(0) = 0,  Y_F(1) = Y_{F,0}
    Y_O(0) = Y_{O,0},  Y_O(1) = 0
"""

import numpy as np
from flamelet_core import (
    scalar_dissipation_rate,
    density_mixture,
    reaction_rate_one_step,
    flamelet_boundary_conditions,
    thermal_diffusivity_ref,
)


def solve_fem_quadratic_species(n, Z_nodes, species_type, T_field,
                                chi_st, tol=1.0e-10, max_iter=50):
    """
    使用二次有限元求解稳态火焰面组分质量分数分布 Y_k(Z)。

    Parameters
    ----------
    n : int
        节点数，必须为奇数且 >= 3。
    Z_nodes : ndarray, shape (n,)
        混合分数空间节点，单调递增。
    species_type : str
        'fuel' 或 'oxidizer'。
    T_field : ndarray, shape (n,)
        已收敛的温度场。
    chi_st : float
        化学计量标量耗散率。
    tol : float
        迭代容差。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    Y : ndarray, shape (n,)
        组分质量分数分布。
    iter_count : int
        迭代次数。
    """
    if n < 3 or n % 2 == 0:
        raise ValueError("二次有限元要求节点数 n 为奇数且 >= 3")
    if not np.all(np.diff(Z_nodes) > 0):
        raise ValueError("Z_nodes 必须严格单调递增")

    bc = flamelet_boundary_conditions()
    e_num = (n - 1) // 2

    # 3-point Gauss-Legendre quadrature
    abscissa = np.array([
        -0.7745966692414834,
        0.0,
        0.7745966692414834
    ])
    weight = np.array([
        0.5555555555555556,
        0.8888888888888889,
        0.5555555555555556
    ])
    quad_num = 3

    # 初始化
    if species_type == 'fuel':
        Y = np.linspace(bc['Y_F_left'], bc['Y_F_right'], n)
        left_bc = bc['Y_F_left']
        right_bc = bc['Y_F_right']
    elif species_type == 'oxidizer':
        Y = np.linspace(bc['Y_O_left'], bc['Y_O_right'], n)
        left_bc = bc['Y_O_left']
        right_bc = bc['Y_O_right']
    else:
        raise ValueError("species_type 必须是 'fuel' 或 'oxidizer'")

    D_ref = thermal_diffusivity_ref()

    for iteration in range(max_iter):
        Y_old = Y.copy()
        A = np.zeros((n, n))
        b = np.zeros(n)

        for e in range(e_num):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2

            xl = Z_nodes[l]
            xm = Z_nodes[m]
            xr = Z_nodes[r]

            for q in range(quad_num):
                xq = ((1.0 - abscissa[q]) * xl + (1.0 + abscissa[q]) * xr) / 2.0
                wq = weight[q] * (xr - xl) / 2.0

                # 二次 Lagrange 形函数
                vl = ((xq - xm) / (xl - xm)) * ((xq - xr) / (xl - xr))
                vm = ((xq - xl) / (xm - xl)) * ((xq - xr) / (xm - xr))
                vr = ((xq - xl) / (xr - xl)) * ((xq - xm) / (xr - xm))

                vlp = (1.0 / (xl - xm)) * ((xq - xr) / (xl - xr)) + \
                      ((xq - xm) / (xl - xm)) * (1.0 / (xl - xr))
                vmp = (1.0 / (xm - xl)) * ((xq - xr) / (xm - xr)) + \
                      ((xq - xl) / (xm - xl)) * (1.0 / (xm - xr))
                vrp = (1.0 / (xr - xl)) * ((xq - xm) / (xr - xm)) + \
                      ((xq - xl) / (xr - xl)) * (1.0 / (xr - xm))

                # 物理量插值
                Tq = np.interp(xq, Z_nodes, T_field)
                Yq = np.interp(xq, Z_nodes, Y_old)

                # 计算扩散系数 D(Z) = α * ρ_ox / ρ(Z)
                rho = density_mixture(xq, Tq)
                rho_ox = density_mixture(0.0, bc['T_left'])
                D_coeff = D_ref * rho_ox / rho
                D_coeff = max(D_coeff, 1.0e-12)

                # TODO: Hole 3 - 实现组分输运方程的反应源项线性化
                # 需要调用 reaction_rate_one_step 计算燃料消耗速率 omega_f，
                # 并根据组分类型（fuel 或 oxidizer）构造对应的源项 f_source。
                # 源项的一般形式为：f = -omega_f / rho（简化处理）
                # 注意参数传递需要与 reaction_rate_one_step 的接口保持一致
                raise NotImplementedError("Hole 3: 请实现反应源项线性化处理")

                # 组装局部刚度矩阵和载荷向量
                A[l, l] += wq * (vlp * D_coeff * vlp + vl * 0.0 * vl)
                A[l, m] += wq * (vlp * D_coeff * vmp + vl * 0.0 * vm)
                A[l, r] += wq * (vlp * D_coeff * vrp + vl * 0.0 * vr)
                b[l] += wq * (vl * f_source)

                A[m, l] += wq * (vmp * D_coeff * vlp + vm * 0.0 * vl)
                A[m, m] += wq * (vmp * D_coeff * vmp + vm * 0.0 * vm)
                A[m, r] += wq * (vmp * D_coeff * vrp + vm * 0.0 * vr)
                b[m] += wq * (vm * f_source)

                A[r, l] += wq * (vrp * D_coeff * vlp + vr * 0.0 * vl)
                A[r, m] += wq * (vrp * D_coeff * vmp + vr * 0.0 * vm)
                A[r, r] += wq * (vrp * D_coeff * vrp + vr * 0.0 * vr)
                b[r] += wq * (vr * f_source)

        # 边界条件
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = left_bc

        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = 1.0
        b[n - 1] = right_bc

        # 求解
        Y = np.linalg.solve(A, b)

        # 边界处理
        Y = np.clip(Y, 0.0, 1.0)

        max_change = np.max(np.abs(Y - Y_old))
        if max_change < tol:
            return Y, iteration + 1

    return Y, max_iter
