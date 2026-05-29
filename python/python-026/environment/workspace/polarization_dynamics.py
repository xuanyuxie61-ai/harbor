# -*- coding: utf-8 -*-
"""
polarization_dynamics.py

基于 circle_map (circle map dots / matrix mapping of unit circle)
的激光偏振态演化与琼斯矩阵计算模块。

原项目 180_circle_map 研究了 2x2 矩阵将单位圆映射为椭圆的几何性质。
在激光-等离子体物理中，这一思想被提升到:
    1. 激光在磁化等离子体中传播时的偏振面旋转 (Faraday 旋转)。
    2. 等离子体双折射导致的偏振椭圆演化。
    3. 琼斯矩阵方法计算输出偏振态。

核心物理模型:
    激光在磁化等离子体中的介电张量 (cold plasma approximation):
        ε = I - ω_p^2/(ω^2 - ω_c^2) * [I - (ω_c/ω) * i * σ · b̂]
    其中 ω_c = eB/m_e 为电子回旋频率，b̂ 为磁场方向单位向量，
    σ 为 Pauli 矩阵向量。

    法拉第旋转角:
        θ_F = (e^3 / (2 ε_0 m_e^2 c ω^2)) * ∫ n_e B_∥ dz

    琼斯矢量演化:
        dE/dz = i * (ω/c) * M * E
    其中 M 为 2x2 耦合矩阵，描述双折射与旋光效应。

    偏振椭圆参数:
        电场矢量端点满足 E† * J * E = 1
        其中 J 为与偏振态相关的 2x2 厄米矩阵。
        椭圆率 ε = b/a (短轴/长轴)，方位角 ψ 满足 tan(2ψ) = S2/S1
        （S1, S2 为 Stokes 参数）。
"""

import numpy as np


def jones_matrix_propagation(omega, omega_p, omega_c, b_hat, dz):
    """
    计算激光在磁化等离子体中传播一小段距离 dz 的琼斯传输矩阵。

    冷等离子体近似下的 2x2 耦合矩阵:
        M = (ω_p^2 / (2ω^2)) * [η_o + η_e,   i*g
                                  -i*g,       η_o + η_e]
    其中 g 为旋光耦合项，与 ω_c 成正比。

    简化的 Faraday 旋转模型:
        T = exp(i * φ * σ_z) * exp(i * δ * σ_x)
    其中 φ = (ω_p^2 ω_c)/(2c ω^3) dz 为旋转角,
          δ = (ω_p^2)/(4c ω^3) (ω_c^2/ω) dz 为相位延迟。

    Parameters
    ----------
    omega : float
        激光角频率 [rad/s]。
    omega_p : float
        等离子体频率 [rad/s]。
    omega_c : float
        电子回旋频率 [rad/s]。
    b_hat : ndarray, shape (3,)
        磁场方向单位向量。
    dz : float
        传播距离 [m]。

    Returns
    -------
    T : ndarray, shape (2, 2)
        琼斯传输矩阵（复数）。
    """
    from physics_constants import C_LIGHT

    if omega <= 0 or dz < 0:
        raise ValueError("omega 必须为正，dz 必须非负。")

    # 简化 Faraday 旋转角 (沿传播方向 z 的分量)
    # 假设传播方向沿 z 轴，b_z = b_hat[2]
    bz = float(b_hat[2]) if b_hat is not None else 1.0
    bz = np.clip(bz, -1.0, 1.0)

    # 旋转角 φ = (ω_p^2 ω_c bz) / (2 c ω^3) * dz
    phi = (omega_p**2 * omega_c * bz) / (2.0 * C_LIGHT * omega**3) * dz

    # 双折射相位差 (Cotton-Mouton 效应，二阶小量)
    delta = (omega_p**2 * omega_c**2) / (4.0 * C_LIGHT * omega**4) * dz

    # 传输矩阵: T = R_z(φ) * diag(exp(i*δ), exp(-i*δ))
    T = np.array([
        [np.cos(phi) + 1j * np.sin(phi) * np.cos(delta), -np.sin(phi) + 1j * np.sin(phi) * np.sin(delta)],
        [np.sin(phi) - 1j * np.sin(phi) * np.sin(delta), np.cos(phi) - 1j * np.sin(phi) * np.cos(delta)]
    ], dtype=complex)

    return T


def apply_jones_matrix(T, E_in):
    """
    应用琼斯矩阵到输入偏振态。

    Parameters
    ----------
    T : ndarray, shape (2, 2)
        琼斯矩阵。
    E_in : ndarray, shape (2,)
        输入琼斯矢量（复数）。

    Returns
    -------
    E_out : ndarray, shape (2,)
        输出琼斯矢量。
    """
    E_in = np.asarray(E_in, dtype=complex)
    if E_in.shape != (2,):
        raise ValueError("E_in 必须是二维复向量。")
    E_out = T @ E_in
    return E_out


def polarization_ellipse_parameters(E):
    """
    由琼斯矢量提取偏振椭圆参数。

    公式:
        强度 I = |E_x|^2 + |E_y|^2
        斯托克斯参数:
            S0 = I
            S1 = |E_x|^2 - |E_y|^2
            S2 = 2 Re(E_x* conj(E_y))
            S3 = 2 Im(E_x* conj(E_y))
        椭圆方位角:
            ψ = 0.5 * arctan2(S2, S1)
        椭圆率角:
            χ = 0.5 * arcsin(S3 / S0)   (S0 > 0)
        椭圆率:
            ε = tan(|χ|)

    Parameters
    ----------
    E : ndarray, shape (2,)
        琼斯矢量。

    Returns
    -------
    psi : float
        椭圆方位角 [rad]。
    chi : float
        椭圆率角 [rad]。
    epsilon : float
        椭圆率 (b/a)。
    S : ndarray, shape (4,)
        斯托克斯参数 [S0, S1, S2, S3]。
    """
    E = np.asarray(E, dtype=complex)
    if E.shape != (2,):
        raise ValueError("E 必须是二维复向量。")

    Ex, Ey = E[0], E[1]
    S0 = abs(Ex)**2 + abs(Ey)**2
    if S0 < 1e-30:
        return 0.0, 0.0, 0.0, np.array([0.0, 0.0, 0.0, 0.0])

    S1 = abs(Ex)**2 - abs(Ey)**2
    S2 = 2.0 * (Ex.real * Ey.real + Ex.imag * Ey.imag)
    S3 = 2.0 * (Ex.real * Ey.imag - Ex.imag * Ey.real)
    S = np.array([S0, S1, S2, S3])

    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3 / S0, -1.0, 1.0))
    epsilon = np.tan(abs(chi))
    epsilon = min(epsilon, 1.0)

    return psi, chi, epsilon, S


def faraday_rotation_integral(ne_profile, B_parallel, z_vals, omega):
    """
    计算沿传播路径的法拉第旋转角积分。

    公式:
        θ_F = (e^3 / (2 ε_0 m_e^2 c ω^2)) * ∫ n_e(z) B_∥(z) dz

    Parameters
    ----------
    ne_profile : ndarray
        电子密度剖面 [m^{-3}]。
    B_parallel : ndarray
        平行磁场分量 [T]。
    z_vals : ndarray
        路径坐标 [m]。
    omega : float
        激光角频率 [rad/s]。

    Returns
    -------
    theta_F : float
        法拉第旋转角 [rad]。
    """
    from physics_constants import E_CHARGE, E_MASS, EPSILON_0, C_LIGHT

    if len(z_vals) < 2 or len(ne_profile) != len(z_vals) or len(B_parallel) != len(z_vals):
        raise ValueError("输入数组长度不一致。")

    prefactor = E_CHARGE**3 / (2.0 * EPSILON_0 * E_MASS**2 * C_LIGHT * omega**2)
    integrand = ne_profile * B_parallel
    theta_F = prefactor * np.trapezoid(integrand, z_vals)
    return float(theta_F)


def circle_map_matrix_polarization(A, n_points=200):
    """
    基于原 circle_map_dots 的矩阵映射思想，计算偏振态的椭圆映射。

    给定一个 2x2 琼斯矩阵 A，将单位圆上的偏振态 (cos θ, sin θ) 映射为
    输出偏振椭圆上的点 A * [cos θ; sin θ]。

    这直接对应于原 circle_map 中 "matrix maps unit circle to ellipse" 的几何。

    Parameters
    ----------
    A : ndarray, shape (2, 2)
        2x2 实矩阵（或复矩阵的实部近似）。
    n_points : int, optional
        采样点数。

    Returns
    -------
    input_circle : ndarray, shape (2, n_points)
        输入圆上的点。
    output_ellipse : ndarray, shape (2, n_points)
        输出椭圆上的点。
    aspect_ratio : float
        椭圆长轴/短轴比（条件数的几何体现）。
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A 必须是 2x2 矩阵。")

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    x_in = np.array([np.cos(theta), np.sin(theta)])
    x_out = A @ x_in

    # 通过 SVD 计算椭圆轴长
    U, s, Vt = np.linalg.svd(A)
    aspect_ratio = s[0] / max(s[1], 1e-30)

    return x_in, x_out, aspect_ratio


def stokes_to_poincare_sphere(S):
    """
    将归一化 Stokes 参数映射到庞加莱球面。

    公式:
        X = S1 / S0
        Y = S2 / S0
        Z = S3 / S0
        满足 X^2 + Y^2 + Z^2 = 1

    Parameters
    ----------
    S : ndarray, shape (4,)
        Stokes 参数 [S0, S1, S2, S3]。

    Returns
    -------
    XYZ : ndarray, shape (3,)
        庞加莱球面上的点。
    """
    S = np.asarray(S, dtype=float)
    S0 = S[0]
    if S0 < 1e-30:
        return np.array([1.0, 0.0, 0.0])
    XYZ = S[1:4] / S0
    # 数值保护
    norm = np.linalg.norm(XYZ)
    if norm > 1e-10:
        XYZ = XYZ / norm
    return XYZ


def evolve_polarization_along_ray(omega, ne_func, B_func, z_vals, E0):
    """
    沿激光射线传播路径逐步演化偏振态。

    Parameters
    ----------
    omega : float
        激光角频率。
    ne_func : callable
        密度函数 ne(z)。
    B_func : callable
        磁场函数 B(z) -> ndarray(3,)。
    z_vals : ndarray
        路径坐标 [m]。
    E0 : ndarray, shape (2,)
        初始琼斯矢量。

    Returns
    -------
    E_history : ndarray, shape (len(z_vals), 2)
        各点上的琼斯矢量。
    stokes_history : ndarray, shape (len(z_vals), 4)
        各点上的 Stokes 参数。
    """
    from physics_constants import plasma_frequency, E_CHARGE, E_MASS

    z_vals = np.asarray(z_vals, dtype=float)
    N = len(z_vals)
    E_history = np.zeros((N, 2), dtype=complex)
    stokes_history = np.zeros((N, 4), dtype=float)

    E = np.asarray(E0, dtype=complex)
    E_history[0, :] = E
    _, _, _, S = polarization_ellipse_parameters(E)
    stokes_history[0, :] = S

    for i in range(1, N):
        dz = z_vals[i] - z_vals[i - 1]
        if dz <= 0:
            E_history[i, :] = E
            stokes_history[i, :] = S
            continue

        z_mid = 0.5 * (z_vals[i] + z_vals[i - 1])
        ne = ne_func(z_mid)
        B = B_func(z_mid)
        omega_p = plasma_frequency(ne)
        B_mag = np.linalg.norm(B)
        omega_c = E_CHARGE * B_mag / E_MASS if B_mag > 0 else 0.0
        b_hat = B / B_mag if B_mag > 0 else np.array([0.0, 0.0, 1.0])

        T = jones_matrix_propagation(omega, omega_p, omega_c, b_hat, dz)
        E = apply_jones_matrix(T, E)
        E_history[i, :] = E
        _, _, _, S = polarization_ellipse_parameters(E)
        stokes_history[i, :] = S

    return E_history, stokes_history
