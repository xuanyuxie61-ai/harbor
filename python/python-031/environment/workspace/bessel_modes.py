# -*- coding: utf-8 -*-
"""
bessel_modes.py
柱坐标本征模式与贝塞尔函数零点

本模块利用贝塞尔函数零点(080_besselj_zero)计算柱对称核pasta相
(spaghetti/anti-spaghetti)的振动模式与库仑势展开.

核心物理公式:
1. 贝塞尔函数 J_n(x):
   J_n(x) = sum_{k=0}^infty (-1)^k (x/2)^{n+2k} / (k! * Gamma(n+k+1))
   
2. 零点计算 (Newton迭代):
   x_{k+1} = x_k - J_n(x_k) / J_n'(x_k)
   初始猜测:
   n<=20: x0 = 2.82141 + 1.15859*n
   n>20:  x0 = n + 1.85576*n^{1/3} + 1.03315/n^{1/3}
   
3. 柱坐标泊松方程的Green函数:
   Phi(r,phi,z) = sum_{m,n} A_{mn} J_m(alpha_{mn} r/R) * cos(m*phi) * exp(i*k*z)
   
4. 核pasta振动频率:
   omega_{mn}^2 = (sigma/rho) * (alpha_{mn}/R)^3 * tanh(alpha_{mn} h/R)
   
   其中:
   sigma: 表面张力
   rho: 质量密度
   R: 柱半径
   h: 特征高度
   
5. 柱对称库仑势:
   Phi(r) = (2*e*rho_p/R) * sum_n [J_0(alpha_{0n} r/R) / (alpha_{0n}^3 * J_1(alpha_{0n}))]
"""

import numpy as np
from scipy.special import jv, jvp


def besselj_zero(n, nt):
    """
    计算贝塞尔函数J_n的前nt个零点 (来自080_besselj_zero).
    
    使用Newton迭代法:
    x_{k+1} = x_k - J_n(x_k) / J_n'(x_k)
    
    输入:
        n: 贝塞尔函数阶数
        nt: 零点个数
    输出:
        rj0: 零点数组 (nt,)
    """
    rj0 = np.zeros(nt)

    if n <= 20:
        x = 2.82141 + 1.15859 * n
    else:
        x = n + 1.85576 * (n ** 0.33333) + 1.03315 / (n ** 0.33333)

    l = 0
    max_iter = 1000
    for _ in range(max_iter):
        x0 = x
        bjn = jv(n, x)
        djn = jvp(n, x, 1)

        if abs(djn) < 1e-15:
            x = x + 0.1
            continue

        x = x - bjn / djn

        if abs(x - x0) > 1.0e-9:
            continue

        if x > 0:
            rj0[l] = x
            l += 1

            if l < nt:
                # 下一个零点的初始猜测
                x = x + np.pi + (0.0972 + 0.0679 * n - 0.000354 * n**2) / l
        else:
            x = x0 + np.pi

        if l >= nt:
            break

    return rj0


def jyndd(n, x):
    """
    计算J_n, J_n', Y_n, Y_n' (来自080_besselj_zero的jyndd).
    
    输入:
        n: 阶数
        x: 自变量
    输出:
        bjn: J_n(x)
        djn: J_n'(x)
        byn: Y_n(x)
        dyn: Y_n'(x)
    """
    from scipy.special import yv, yvp
    bjn = jv(n, x)
    djn = jvp(n, x, 1)
    byn = yv(n, x)
    dyn = yvp(n, x, 1)
    return bjn, djn, byn, dyn


def cylinder_coulomb_potential(r, R_cyl, rho_p, n_modes=20):
    """
    计算柱对称库仑势的贝塞尔展开.
    
    公式:
    Phi(r) = 4*pi*e*rho_p * sum_{n=1}^infty 
             J_0(alpha_n * r / R) / (alpha_n^3 * J_1(alpha_n))
    
    其中 alpha_n 是 J_0 的第n个零点.
    
    输入:
        r: 径向坐标 (数组或标量)
        R_cyl: 柱半径
        rho_p: 质子密度
        n_modes: 模式数
    输出:
        Phi: 电势
    """
    r = np.asarray(r)
    alpha = besselj_zero(0, n_modes)

    Phi = np.zeros_like(r, dtype=float)
    for n in range(n_modes):
        al = alpha[n]
        if al <= 0.0:
            continue
        J1_al = jv(1, al)
        if abs(J1_al) < 1e-15:
            continue
        coeff = 1.0 / (al**3 * J1_al)
        arg = al * r / R_cyl
        # 限制在柱内
        mask = r <= R_cyl
        Phi[mask] += coeff * jv(0, arg[mask])

    Phi = Phi * 4.0 * np.pi * 1.43996448 * rho_p * R_cyl**2
    return Phi


def cylinder_vibration_frequencies(R_cyl, surface_tension, mass_density, n_modes=10):
    """
    计算核pasta柱相的振动频率.
    
    公式 (Rayleigh模式):
    omega_{mn}^2 = (sigma / (rho * R_cyl^3)) * alpha_{mn} * (alpha_{mn}^2 - m^2)
    
    输入:
        R_cyl: 柱半径 (fm)
        surface_tension: 表面张力 (MeV/fm^2)
        mass_density: 质量密度 (MeV/fm^3)
        n_modes: 模式数
    输出:
        freqs: 频率数组 (Hz, 自然单位)
    """
    if R_cyl <= 0.0 or surface_tension <= 0.0 or mass_density <= 0.0:
        return np.array([])

    freqs = []
    for m in range(n_modes):
        alpha = besselj_zero(m, n_modes)
        for n in range(len(alpha)):
            al = alpha[n]
            if al <= m:
                continue
            omega_sq = (surface_tension / (mass_density * R_cyl**3)) * al * (al**2 - m**2)
            if omega_sq > 0:
                freqs.append(np.sqrt(omega_sq))

    return np.array(sorted(freqs))


def spherical_coulomb_potential(r, R_sphere, rho_p):
    """
    球对称库仑势 (精确解).
    
    r < R: Phi(r) = (2*pi*e*rho_p) * (R^2 - r^2/3)
    r > R: Phi(r) = (4*pi*e*rho_p*R^3) / (3*r)
    
    输入:
        r: 径向坐标
        R_sphere: 球半径
        rho_p: 质子密度
    输出:
        Phi: 电势
    """
    r = np.asarray(r)
    e2 = 1.43996448  # MeV·fm
    Phi = np.zeros_like(r, dtype=float)

    mask_in = r <= R_sphere
    mask_out = r > R_sphere

    Phi[mask_in] = 2.0 * np.pi * e2 * rho_p * (R_sphere**2 - r[mask_in]**2 / 3.0)
    Phi[mask_out] = (4.0 * np.pi * e2 * rho_p * R_sphere**3) / (3.0 * r[mask_out])

    return Phi


def sheet_coulomb_potential(z, t_sheet, rho_p):
    """
    片状相的库仑势.
    
    公式:
    |z| < t/2: Phi(z) = 2*pi*e*rho_p * (t^2/4 - z^2)
    |z| > t/2: Phi(z) = pi*e*rho_p*t * (t/2 - |z|)
    
    输入:
        z: 垂直于片面的坐标
        t_sheet: 片厚度
        rho_p: 质子密度
    输出:
        Phi: 电势
    """
    z = np.asarray(z)
    e2 = 1.43996448
    Phi = np.zeros_like(z, dtype=float)

    mask_in = np.abs(z) <= t_sheet / 2.0
    mask_out = np.abs(z) > t_sheet / 2.0

    Phi[mask_in] = 2.0 * np.pi * e2 * rho_p * (t_sheet**2 / 4.0 - z[mask_in]**2)
    Phi[mask_out] = np.pi * e2 * rho_p * t_sheet * (t_sheet / 2.0 - np.abs(z[mask_out]))

    return Phi


def pasta_deformation_energy(phase_id, R, amplitude, mode_m, surface_tension):
    """
    计算pasta相表面形变能.
    
    公式 (Rayleigh形变):
    deltaE = pi * sigma * R^2 * |epsilon|^2 * (m^2 + m - 2)
    
    输入:
        phase_id: 相类型 (1=球, 2=柱)
        R: 特征半径
        amplitude: 形变幅度 epsilon
        mode_m: 角向模式数
        surface_tension: 表面张力
    输出:
        deltaE: 形变能 (MeV)
    """
    if R <= 0.0 or surface_tension <= 0.0:
        return 0.0

    eps_sq = amplitude**2
    m = mode_m

    if phase_id == 1:  # 球
        # 球谐形变
        deltaE = 4.0 * np.pi * surface_tension * R**2 * eps_sq * (m - 1) * (m + 2) / 2.0
    elif phase_id == 2:  # 柱
        # 柱形变 (Rayleigh不稳定性)
        deltaE = np.pi * surface_tension * R**2 * eps_sq * (m**2 + m - 2.0)
    else:
        deltaE = 0.0

    return deltaE


if __name__ == '__main__':
    # 自测试
    zeros = besselj_zero(0, 5)
    print(f"J_0 zeros: {zeros}")
    Phi = cylinder_coulomb_potential(np.array([0.0, 0.5, 1.0]), 1.0, 0.01)
    print(f"Cylinder Coulomb potential: {Phi}")
