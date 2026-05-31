"""
collective_dynamics.py
=======================
核集体运动阻尼振荡动力学模块

基于种子项目 020_artery_pde 的受迫阻尼谐振子模型，
将血液动脉壁的受迫振动映射到原子核的集体激发模式
(巨共振、表面振动、转动) 的阻尼振荡动力学。

核心公式
--------
核集体运动的受迫阻尼谐振子方程:
    μ d²Q_λ/dt² + γ dQ_λ/dt + C_λ Q_λ = F_λ(t)

其中:
    Q_λ    : λ 阶集体坐标 (如四极矩 Q_20)
    μ      : 集体质量参数 (B_λ)
    γ      : 阻尼宽度 (Γ_λ / 2)
    C_λ    : 恢复力参数 (C_λ = μ ω_λ²)
    F_λ(t) : 外部场耦合 (如入射粒子导致的时变多极场)

能量依赖的宽度 (巨共振):
    Γ(E) = Γ_0 * (E² + (πT)²) / E_0²

其中 T 为核温度，E_0 为共振能量。

Bohr-Mottelson 集体哈密顿量:
    H = Σ_λ [B_λ/2 * ᵒQ̇_λ² + C_λ/2 * Q_λ²]

本模块包含:
1. 受迫阻尼谐振子的时间演化
2. 巨共振截面计算 (Breit-Wigner + Lorentz)
3. 集体质量与恢复力参数估计
"""

import numpy as np
from scipy.integrate import solve_ivp


def collective_mass_parameter(A, lam):
    """
    计算 λ 阶集体质量参数 B_λ。

    采用无旋流体模型 (Inglis 公式):
        B_λ = (3 / 4π) * A * m * R_0² / λ

    其中 m 为核子质量，R_0 为核半径。
    """
    m_nucleon = 931.5  # MeV/c^2
    r0 = 1.2  # fm
    R0 = r0 * (A ** (1.0 / 3.0))
    B_lam = (3.0 / (4.0 * np.pi)) * A * m_nucleon * (R0 ** 2) / lam
    return B_lam


def restoring_force_parameter(A, lam):
    """
    计算恢复力参数 C_λ。

    采用液滴模型:
        C_λ = (λ - 1)(λ + 2) * a_s A^{2/3} - (2λ - 1) a_c Z² / A^{1/3}

    其中 a_s 为表面张力系数，a_c 为库仑系数。
    """
    a_s = 17.8  # MeV
    a_c = 0.711  # MeV
    # 简化 Z 估计
    Z = A / 2.0
    C_lam = ((lam - 1.0) * (lam + 2.0) * a_s * (A ** (2.0 / 3.0))
             - (2.0 * lam - 1.0) * a_c * (Z ** 2) / (A ** (1.0 / 3.0)))
    return C_lam


def resonance_energy(A, lam):
    """
    巨共振能量 (经验公式)。

    四极巨共振 (GQR):
        E_GQR ≈ 63 A^{-1/3} MeV

    偶极巨共振 (GDR):
        E_GDR ≈ 31.2 A^{-1/3} + 20.6 A^{-1/6} MeV
    """
    if lam == 1:
        return 31.2 * (A ** (-1.0 / 3.0)) + 20.6 * (A ** (-1.0 / 6.0))
    elif lam == 2:
        return 63.0 * (A ** (-1.0 / 3.0))
    elif lam == 3:
        return 110.0 * (A ** (-1.0 / 3.0))
    else:
        return 60.0 * (A ** (-1.0 / 3.0))


def damping_width(A, lam, E_exc, T_nucleus=0.0):
    """
    计算集体激发的阻尼宽度 Γ。

    经验公式:
        Γ = Γ_0 + c * E_exc

    其中 Γ_0 为基态宽度，c 为能量依赖系数。
    """
    if lam == 1:
        Gamma_0 = 4.0 + 0.05 * A  # MeV, GDR 宽度
    elif lam == 2:
        Gamma_0 = 2.5 + 0.03 * A  # GQR 宽度
    else:
        Gamma_0 = 1.5 + 0.02 * A

    # 温度展宽 (热核)
    thermal_broadening = 4.0 * (np.pi * T_nucleus) ** 2 / (E_exc + 1e-6)
    return Gamma_0 + 0.1 * E_exc + thermal_broadening


def forced_damped_oscillator(t_span, Q0, dQ0, B, C, Gamma, F_func, n_steps=1000):
    """
    求解受迫阻尼集体振子的时间演化。

    方程: B * Q'' + Γ * B * Q' + C * Q = F(t)
    或: Q'' + (Γ/B) Q' + (C/B) Q = F(t)/B

    参数参照 020_artery_pde 的受迫阻尼谐振子结构。

    Parameters
    ----------
    t_span : tuple
        (t0, tf) 时间范围 (fm/c)。
    Q0, dQ0 : float
        初始位移和速度。
    B, C, Gamma : float
        集体质量、恢复力、阻尼宽度。
    F_func : callable
        外力函数 F(t)。
    n_steps : int
        输出步数。

    Returns
    -------
    t : ndarray
        时间数组。
    Q : ndarray
        集体坐标演化。
    dQ : ndarray
        集体速度演化。
    """
    omega0_sq = C / B
    gamma_damp = Gamma / B
    F_scale = 1.0 / B

    def ode_system(t, y):
        Q, V = y
        dQdt = V
        dVdt = -gamma_damp * V - omega0_sq * Q + F_scale * F_func(t)
        return [dQdt, dVdt]

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(ode_system, t_span, [Q0, dQ0], t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0, :], sol.y[1, :]


def giant_resonance_cross_section(E_gamma, A, lam=1):
    """
    计算巨共振的光核反应截面 (Breit-Wigner 型)。

    σ(E) = σ_max * (Γ²/4) / [(E - E_R)² + Γ²/4]

    采用 Lorentz 线型 (Breit-Wigner 近似)。

    Parameters
    ----------
    E_gamma : float or ndarray
        γ 射线能量 (MeV)。
    A : int
        质量数。
    lam : int
        多极阶数 (1=偶极, 2=四极)。

    Returns
    -------
    sigma : ndarray
        截面 (mb)。
    """
    E_gamma = np.asarray(E_gamma, dtype=float)
    E_R = resonance_energy(A, lam)
    Gamma_R = damping_width(A, lam, E_R)

    # 峰截面 (经典偶极求和规则)
    if lam == 1:
        sigma_max = 60.0 * A / (2.0 * np.pi)  # mb·MeV 量级
    elif lam == 2:
        sigma_max = 25.0 * A / (2.0 * np.pi)
    else:
        sigma_max = 10.0 * A / (2.0 * np.pi)

    sigma = sigma_max * ((Gamma_R / 2.0) ** 2) / ((E_gamma - E_R) ** 2 + (Gamma_R / 2.0) ** 2)
    return sigma


def energy_weighted_sum_rule(A, lam):
    """
    能量权重求和规则 (EWSR)。

    对于 λ 阶巨共振:
        S_λ = (2λ+1) ħ² / (2m) * A * <r^{2λ-2}> / (4π)

    本函数给出 EWSR 的理论上限 (用于校验截面积分)。
    """
    hbar2_over_2m = 20.7  # MeV·fm²
    r0 = 1.2  # fm
    R0 = r0 * (A ** (1.0 / 3.0))
    r2lam2 = (R0 ** (2.0 * lam - 2.0)) * (3.0 / (2.0 * lam + 1.0))
    S = (2.0 * lam + 1.0) * hbar2_over_2m * A * r2lam2 / (4.0 * np.pi)
    return S


def strength_function_integral(E_gamma, sigma, method='trapezoid'):
    """
    计算强度函数积分:

    S = ∫ σ(E) dE

    用于校验求和规则。
    """
    if method == 'trapezoid':
        return np.trapezoid(sigma, E_gamma)
    else:
        from scipy.integrate import simpson
        return simpson(sigma, x=E_gamma)


def time_dependent_multipole_field(t, omega, E0, lam):
    """
    外部时变多极场 (模拟入射粒子引起的微扰)。

    F_λ(t) = E_0 * cos(ωt) * exp(-t/τ) * sqrt(2λ+1)
    """
    tau = 50.0  # fm/c, 特征时间
    return E0 * np.cos(omega * t) * np.exp(-t / tau) * np.sqrt(2.0 * lam + 1.0)


if __name__ == "__main__":
    # 自检
    A = 56
    B2 = collective_mass_parameter(A, 2)
    C2 = restoring_force_parameter(A, 2)
    E2 = resonance_energy(A, 2)
    Gamma2 = damping_width(A, 2, E2)
    print(f"A={A}, λ=2: B={B2:.2f}, C={C2:.2f}, E_R={E2:.2f} MeV, Γ={Gamma2:.2f} MeV")

    # 时间演化
    omega = E2 / 197.3  # fm^{-1} (自然单位)
    F = lambda t: time_dependent_multipole_field(t, omega, 1.0, 2)
    t, Q, dQ = forced_damped_oscillator((0, 200), 0.0, 0.0, B2, C2, Gamma2, F, n_steps=500)
    print(f"集体坐标最终振幅: {Q[-1]:.6f} fm")

    # 巨共振截面
    E_range = np.linspace(5, 25, 200)
    sigma = giant_resonance_cross_section(E_range, A, lam=2)
    S_int = strength_function_integral(E_range, sigma)
    print(f"GQR 强度积分: {S_int:.2f} mb·MeV")
    print(f"EWSR 上限: {energy_weighted_sum_rule(A, 2):.2f} mb·MeV (注意单位换算)")
