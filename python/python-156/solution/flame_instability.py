"""
flame_instability.py
====================
基于分段非线性ODE的火焰不稳定性动力学模块。

核心算法源自 rubber_band_ode (Project 1049)，并改造用于描述
火焰前锋的 Darrieus-Landau 热扩散不稳定性和热声振荡。

原始橡皮筋方程：
    y'' + 0.01 y' + a y(+) - b y(-) = 10 + λ sin(μ t)

在火焰不稳定性中，火焰前锋位移 ξ(t) 满足类似的受迫振子方程：

    d²ξ/dt² + 2γ dξ/dt + ω_0² ξ = F_thermal(t) + F_acoustic(t)

其中：
    γ 为阻尼系数（与热扩散和粘性耗散相关），
    ω_0 为火焰前锋固有频率，
    F_thermal 为热膨胀驱动力（分段线性：火焰面两侧密度突变），
    F_acoustic 为热声振荡驱动力。

Darrieus-Landau 不稳定性增长率：
    σ_DL = (S_L k) * (ρ_u / (ρ_u + ρ_b)) * [
        sqrt( (ρ_u + ρ_b) / ρ_b + (ρ_u - ρ_b)² / ρ_b² ) - (ρ_u + ρ_b) / ρ_b
    ]

其中 k 为扰动波数，S_L 为层流火焰速度，ρ_u 和 ρ_b 分别为
未燃和已燃气体密度。

热声耦合（简化模型）：
    F_acoustic = p' * A_flame / m_flame

其中 p' 为压力脉动，A_flame 为火焰面积，m_flame 为火焰质量。
"""

import numpy as np


def flame_instability_deriv(t, y, S_L=0.4, rho_u=1.2, rho_b=0.2,
                            gamma_damp=50.0, omega_0=200.0,
                            lam_thermal=100.0, mu_acoustic=500.0):
    """
    火焰不稳定性ODE的右端函数（类似 rubber_band_deriv）。

    状态向量 y = [ξ, dξ/dt]，其中 ξ 为火焰前锋位移。

    方程：
        dξ/dt = v
        dv/dt = -2γ v - ω_0² ξ + F_thermal + F_acoustic

    热膨胀力（分段线性，模拟密度突变）：
        F_thermal = lam * sin(μ t) - a * max(ξ, 0) + b * max(-ξ, 0)

    Parameters
    ----------
    t : float
        时间。
    y : ndarray, shape (2,)
        [ξ, v]。
    S_L : float
        层流火焰速度，m/s。
    rho_u : float
        未燃气体密度，kg/m³。
    rho_b : float
        已燃气体密度，kg/m³。
    gamma_damp : float
        阻尼系数。
    omega_0 : float
        固有频率，rad/s。
    lam_thermal : float
        热声耦合强度。
    mu_acoustic : float
        热声振荡频率，rad/s。

    Returns
    -------
    dydt : ndarray, shape (2,)
        [dξ/dt, dv/dt]。
    """
    xi = y[0]
    v = y[1]

    # 热膨胀恢复力（分段线性，模拟 Rayleigh-Taylor 型不稳定性）
    a_thermal = rho_u * S_L ** 2
    b_thermal = rho_b * S_L ** 2

    F_thermal = (lam_thermal * np.sin(mu_acoustic * t)
                 - a_thermal * max(xi, 0.0)
                 + b_thermal * max(-xi, 0.0))

    # 热声驱动力（Rayleigh 准则）
    # 当压力脉动与热释放脉动同相时，不稳定增长
    phase_lag = 0.1  # 相位滞后
    F_acoustic = 50.0 * np.cos(mu_acoustic * t + phase_lag) * max(xi, 0.0)

    dxi_dt = v
    dv_dt = -2.0 * gamma_damp * v - omega_0 ** 2 * xi + F_thermal + F_acoustic

    return np.array([dxi_dt, dv_dt])


def integrate_flame_instability(t_span, y0, dt=1.0e-5, **kwargs):
    """
    使用四阶 Runge-Kutta 法积分火焰不稳定性方程。

    Parameters
    ----------
    t_span : tuple
        (t_start, t_end)。
    y0 : ndarray, shape (2,)
        初始条件 [ξ0, v0]。
    dt : float
        时间步长。
    **kwargs : dict
        传递给 flame_instability_deriv 的参数。

    Returns
    -------
    t_arr : ndarray
        时间数组。
    y_arr : ndarray, shape (N, 2)
        解数组 [ξ, v]。
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_arr = np.linspace(t_start, t_end, n_steps)

    y_arr = np.zeros((n_steps, 2))
    y_arr[0] = y0

    for i in range(n_steps - 1):
        t = t_arr[i]
        y = y_arr[i]

        k1 = flame_instability_deriv(t, y, **kwargs)
        k2 = flame_instability_deriv(t + dt / 2.0, y + dt / 2.0 * k1, **kwargs)
        k3 = flame_instability_deriv(t + dt / 2.0, y + dt / 2.0 * k2, **kwargs)
        k4 = flame_instability_deriv(t + dt, y + dt * k3, **kwargs)

        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 边界处理
        y_new[0] = np.clip(y_new[0], -0.1, 0.1)  # 位移限制
        y_new[1] = np.clip(y_new[1], -10.0, 10.0)  # 速度限制

        y_arr[i + 1] = y_new

    return t_arr, y_arr


def darrieus_landau_growth_rate(k, S_L=0.4, rho_u=1.2, rho_b=0.2):
    """
    计算 Darrieus-Landau 不稳定性增长率。

    公式：
        σ = S_L k * [ ρ_u/(ρ_u+ρ_b) * sqrt( (ρ_u+ρ_b)/ρ_b + (ρ_u-ρ_b)²/ρ_b² ) - ρ_u/ρ_b ]

    Parameters
    ----------
    k : float or ndarray
        扰动波数，m⁻¹。
    S_L : float
        层流火焰速度。
    rho_u, rho_b : float
        未燃/已燃密度。

    Returns
    -------
    sigma : float or ndarray
        增长率，s⁻¹。
    """
    alpha = rho_u / rho_b
    term = (alpha / (1.0 + alpha)) * np.sqrt((1.0 + alpha) + (alpha - 1.0) ** 2)
    sigma = S_L * k * (term - alpha)

    # 边界处理：短波被热扩散稳定
    k_stabilize = 2.0 * np.pi / 1.0e-3  # 稳定波数阈值
    stabilization = np.exp(-(k / k_stabilize) ** 2)
    sigma = sigma * stabilization

    return sigma
