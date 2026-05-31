"""
potential_field.py
==================
电势场解析计算模块（基于特殊函数展开）

基于种子项目:
  - 462_gegenbauer_polynomial: Gegenbauer 正交多项式
  - 1082_sinc: Sinc 函数及积分

科学背景:
  在均匀球状或柱对称介质中，点电流源产生的电势可用
  特殊函数展开表示。对于耳蜗近似柱坐标下的电势:

  柱坐标 Poisson 方程:
      (1/r) ∂/∂r(r σ ∂V/∂r) + (1/r²) ∂²V/∂θ² + ∂²V/∂z² = -I_e δ(r-r_e)/σ

  在轴对称近似下，径向部分满足修正 Bessel 方程，
  角度部分可用 Gegenbauer 多项式展开。

  对于球坐标下的单电极近似，电势可用 Legendre/Gegenbauer 展开:
      V(r,θ) = Σ_{n=0}^∞ A_n C_n^{(α)}(cos θ) (r^n 或 r^{-n-1})

  其中 C_n^{(α)} 为 Gegenbauer 多项式，满足:
      (1-x²) y'' - (2α+1) x y' + n(n+2α) y = 0
"""

import numpy as np
from scipy.special import gamma as gamma_func


def gegenbauer_polynomial_value(m, alpha, x):
    """
    计算 Gegenbauer 多项式 C_n^{(α)}(x)，n=0..m。

    基于种子 462_gegenbauer_polynomial 的递推关系:
        C_0^{(α)}(x) = 1
        C_1^{(α)}(x) = 2α x
        C_n^{(α)}(x) = [ (2n-2+2α) x C_{n-1}^{(α)}(x)
                         + (-n+2-2α) C_{n-2}^{(α)}(x) ] / n

    正交性:
        ∫_{-1}^{1} (1-x²)^{α-1/2} C_m^{(α)}(x) C_n^{(α)}(x) dx
        = π 2^{1-2α} Γ(m+2α) / [m! (m+α) (Γ(α))²]  δ_{mn}

    Parameters
    ----------
    m : int
        最高阶数
    alpha : float
        Gegenbauer 参数，必须 > -0.5
    x : float or ndarray
        求值点

    Returns
    -------
    C : ndarray, shape (m+1, len(x))
        C[n, :] = C_n^{(α)}(x)
    """
    if alpha <= -0.5:
        raise ValueError("alpha 必须大于 -0.5")

    x = np.atleast_1d(x)
    n_points = len(x)
    C = np.zeros((m + 1, n_points))

    if m >= 0:
        C[0, :] = 1.0
    if m >= 1:
        C[1, :] = 2.0 * alpha * x

    for n in range(2, m + 1):
        C[n, :] = (
            (2.0 * n - 2.0 + 2.0 * alpha) * x * C[n - 1, :]
            + (-n + 2.0 - 2.0 * alpha) * C[n - 2, :]
        ) / n

    return C


def gegenbauer_norm_squared(n, alpha):
    """
    计算 Gegenbauer 多项式的归一化常数平方:
        h_n = π 2^{1-2α} Γ(n+2α) / [n! (n+α) (Γ(α))²]
    """
    from scipy.special import gamma, factorial
    h_n = (np.pi * 2.0**(1.0 - 2.0 * alpha) * gamma(n + 2.0 * alpha)
           / (factorial(n) * (n + alpha) * gamma(alpha)**2))
    return h_n


def sincn(x):
    """
    归一化 sinc 函数:
        sinc(x) = sin(π x) / (π x),  sinc(0) = 1

    基于种子 1082_sinc 的实现。

    Sinc 函数在采样定理和带限插值中有核心作用:
        f(x) = Σ_n f(nΔ) sinc((x - nΔ)/Δ)
    """
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    nz = np.abs(x) > 1e-15
    result[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    return result


def sinc_interpolation_1d(x_samples, f_samples, x_query):
    """
    Whittaker-Shannon sinc 插值。

    Parameters
    ----------
    x_samples : ndarray
        等距采样点
    f_samples : ndarray
        采样值
    x_query : ndarray
        查询点

    Returns
    -------
    f_query : ndarray
        插值结果
    """
    x_samples = np.asarray(x_samples, dtype=float)
    f_samples = np.asarray(f_samples, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    dx = np.mean(np.diff(x_samples))
    if dx <= 0:
        raise ValueError("采样点必须严格递增")

    f_query = np.zeros_like(x_query)
    for i, xs in enumerate(x_samples):
        f_query += f_samples[i] * sincn((x_query - xs) / dx)

    return f_query


def analytical_potential_spherical(r, theta, I_source, sigma, R_cochlea,
                                    n_terms=20, alpha=0.5):
    """
    球坐标下耳蜗近似模型的解析电势。

    假设耳蜗近似为半径 R 的球壳，电极位于 θ=0 的球面上，
    发放电流 I。电势满足:
        ∇²V = 0  (除电极外)

    使用 Gegenbauer 多项式展开:
        V(r,θ) = (I / (4πσR)) Σ_{n=0}^∞ P_n(cos θ) (r/R)^n   (r < R)

    这里用 Gegenbauer 多项式 C_n^{(1/2)}(cos θ) = P_n(cos θ) (Legendre)。

    Parameters
    ----------
    r : float or ndarray
        径向坐标 (mm)
    theta : float or ndarray
        极角 (rad)
    I_source : float
        电流源强度 (A)
    sigma : float
        电导率 (S/m)
    R_cochlea : float
        球壳半径 (mm)
    n_terms : int
        展开项数
    alpha : float
        Gegenbauer 参数，0.5 对应 Legendre

    Returns
    -------
    V : ndarray
        电势 (V)
    """
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    if len(r) != len(theta):
        raise ValueError("r 和 theta 长度必须相同")

    x = np.cos(theta)
    C = gegenbauer_polynomial_value(n_terms, alpha, x)

    V = np.zeros_like(r)
    prefactor = I_source / (4.0 * np.pi * sigma * R_cochlea * 1e-3)  # mm->m

    for n in range(n_terms + 1):
        ratio = np.where(r < R_cochlea, r / R_cochlea, R_cochlea / r)
        V += prefactor * C[n, :] * ratio**n / (n + 1.0)

    return V


def cylindrical_potential_line_source(rho, z, z_e, I_e, sigma):
    """
    无限长圆柱介质中线电流源的电势。

    线源沿 z 轴方向，电极位于 z=z_e。在柱坐标下:
        V(ρ,z) = (I_e / (4πσ)) * 1/√(ρ² + (z-z_e)²)

    这是点源在三维空间中的基本解，在耳蜗纵向近似中常用。

    Parameters
    ----------
    rho : ndarray
        到电极的横向距离 (mm)
    z : ndarray
        纵向坐标 (mm)
    z_e : float
        电极纵向位置 (mm)
    I_e : float
        电流 (A)
    sigma : float
        电导率 (S/m)

    Returns
    -------
    V : ndarray
        电势 (V)
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    dist_mm = np.sqrt(rho**2 + (z - z_e)**2)
    dist_m = dist_mm * 1e-3
    # 避免奇点
    dist_m = np.where(dist_m < 1e-6, 1e-6, dist_m)
    V = I_e / (4.0 * np.pi * sigma * dist_m)
    return V


def multi_electrode_superposition(electrode_positions, electrode_currents,
                                   query_points, sigma):
    """
    多电极电势叠加原理。

    线性介质中满足叠加原理:
        V_total = Σ_e V_e

    Parameters
    ----------
    electrode_positions : ndarray, shape (N_e, 2)
        电极位置 (mm)
    electrode_currents : ndarray, shape (N_e,)
        电极电流 (A)
    query_points : ndarray, shape (N_q, 2)
        查询点 (mm)
    sigma : float
        电导率 (S/m)

    Returns
    -------
    V : ndarray, shape (N_q,)
    """
    electrode_positions = np.asarray(electrode_positions, dtype=float)
    electrode_currents = np.asarray(electrode_currents, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    V = np.zeros(query_points.shape[0])
    for pos, I in zip(electrode_positions, electrode_currents):
        if abs(I) < 1e-15:
            continue
        dists_mm = np.linalg.norm(query_points - pos, axis=1)
        dists_m = dists_mm * 1e-3
        dists_m = np.where(dists_m < 1e-6, 1e-6, dists_m)
        V += I / (4.0 * np.pi * sigma * dists_m)

    return V
