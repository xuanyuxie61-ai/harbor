"""
fd_scalar_solver.py
===================
基于有限差分法（FDM）的1D标量耗散率方程求解器。

核心算法源自 fd1d_bvp (Project 358)，并改造用于求解混合分数空间中的
标量耗散率修正方程：

    - d/dZ [ D_eff(Z) dχ/dZ ] + c(Z) χ = f(Z)

其中 D_eff(Z) 为有效扩散系数，c(Z) 为衰减系数，f(Z) 为源项。

在火焰面模型中，标量耗散率 χ 的控制方程可写为：

    dχ/dt + U_flamelet dχ/dZ = D_eff d²χ/dZ² - C_χ ω χ / k_turb + S_χ

对于稳态情况，简化为：

    - D_eff d²χ/dZ² + (C_χ ω / k_turb) χ = S_χ(Z)

有限差分离散（非均匀网格）：
----------------------------
在节点 i 处，使用中心差分：

    d²χ/dZ² ≈ 2/(Z_{i+1}-Z_{i-1}) * [
        (χ_{i+1} - χ_i)/(Z_{i+1} - Z_i) - (χ_i - χ_{i-1})/(Z_i - Z_{i-1})
    ]

    dχ/dZ ≈ (χ_{i+1} - χ_{i-1})/(Z_{i+1} - Z_{i-1})

离散后形成三对角线性系统 A χ = rhs。

边界条件：
    χ(0) = χ_ox,  χ(1) = χ_fuel
"""

import numpy as np
from flamelet_core import scalar_dissipation_rate, thermal_diffusivity_ref


def solve_fd_scalar_dissipation(n, Z_nodes, chi_st, C_chi=2.0, omega_turb=100.0,
                                k_turb=10.0, tol=1.0e-10, max_iter=50):
    """
    使用有限差分法求解标量耗散率修正方程。

    Parameters
    ----------
    n : int
        节点数，必须 >= 3。
    Z_nodes : ndarray, shape (n,)
        混合分数空间节点，严格递增。
    chi_st : float
        化学计量点标量耗散率。
    C_chi : float
        标量耗散率模型常数，典型值 2.0。
    omega_turb : float
        湍流频率，单位 s⁻¹。
    k_turb : float
        湍流动能，单位 m²/s²。
    tol : float
        迭代容差。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    chi : ndarray, shape (n,)
        收敛的标量耗散率分布。
    iter_count : int
        迭代次数。
    """
    if n < 3:
        raise ValueError("节点数 n 必须 >= 3")
    if not np.all(np.diff(Z_nodes) > 0):
        raise ValueError("Z_nodes 必须严格单调递增")

    D_eff = thermal_diffusivity_ref()

    # 初始猜测：使用解析的标量耗散率分布
    chi = scalar_dissipation_rate(Z_nodes, chi_st)

    # 边界值
    chi_ox = float(chi[0])
    chi_fuel = float(chi[-1])

    for iteration in range(max_iter):
        chi_old = chi.copy()
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        # 左边界
        A[0, 0] = 1.0
        rhs[0] = chi_ox

        # 内部节点
        for i in range(1, n - 1):
            xm = Z_nodes[i]
            dxl = Z_nodes[i] - Z_nodes[i - 1]
            dxr = Z_nodes[i + 1] - Z_nodes[i]
            dx_total = Z_nodes[i + 1] - Z_nodes[i - 1]

            # 有效扩散系数（使用上一迭代值）
            Dm = D_eff * (1.0 + 0.1 * chi_old[i] / max(chi_st, 1.0e-6))
            Dm = max(Dm, 1.0e-12)

            # 衰减系数
            cm = C_chi * omega_turb / max(k_turb, 1.0e-6)

            # 源项：标量耗散率的生成项（简化模型）
            fm = 0.5 * chi_st * np.exp(-((xm - 0.5) / 0.3) ** 2)

            # 二阶导数系数
            coeff_l = -2.0 * Dm / (dxl * dx_total)
            coeff_r = -2.0 * Dm / (dxr * dx_total)
            coeff_c = 2.0 * Dm / (dxl * dxr) + cm

            A[i, i - 1] = coeff_l
            A[i, i] = coeff_c
            A[i, i + 1] = coeff_r
            rhs[i] = fm

        # 右边界
        A[n - 1, n - 1] = 1.0
        rhs[n - 1] = chi_fuel

        # 求解
        chi = np.linalg.solve(A, rhs)

        # 边界处理
        chi = np.maximum(chi, 0.0)

        max_change = np.max(np.abs(chi - chi_old))
        if max_change < tol:
            return chi, iteration + 1

    return chi, max_iter
