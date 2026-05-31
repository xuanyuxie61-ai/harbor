"""
fusion_kinetics.py
氘-氚聚变反应动力学与功率计算。

核心物理模型：
  D + T → He⁴(3.52 MeV) + n(14.06 MeV)

  反应速率方程组（类比化学动力学 A + B → C）：

      dn_D/dt  =  - <σv>_DT n_D n_T / (1 + δ_D)  +  S_D(t)  -  n_D / τ_p
      dn_T/dt  =  - <σv>_DT n_D n_T / (1 + δ_T)  +  S_T(t)  -  n_T / τ_p
      dn_He/dt =  + <σv>_DT n_D n_T                -  n_He / τ_He
      dn_n/dt  =  + <σv>_DT n_D n_T                -  n_n / τ_n

  其中 <σv>_DT 为 D-T 反应速率参数，温度依赖关系采用 Bosch-Hale 参数化：

      <σv> = C1 · θ · √(ξ / (m_r c² T_i³)) · exp(-3ξ)   [m³/s]

      θ = T_i / (1 - T_i (C2 + T_i (C4 + T_i C6)) / (1 + T_i (C3 + T_i (C5 + T_i C7))))
      ξ = ( (b_g² / (4θ))^{1/3} )

  简化模型中采用近似 Maxwellian 平均截面：

      <σv>(T_i) ≈ 10^{-18} · exp( -20 / T_i^{1/3} )   [m³/s], T_i in keV

  聚变功率密度：

      P_fus = n_D n_T <σv>_DT E_fus / 4    [W/m³]

      (因子 1/4 来源于 n_D = n_T = n_e/2 的假设)

  能量约束时间 τ_E 采用 ITER89-P 缩放律：

      τ_E = 0.048 · I_p^{0.85} B_t^{0.2} n_e^{0.1} P_{loss}^{-0.5} R^{1.5} a^{0.3} κ^{0.5} M^{0.5}   [s]

  其中 M = 2.5 (DT 混合平均质量)，P_{loss} = P_α + P_oh - dW/dt。
"""

import numpy as np
from parameters import (
    get_fusion_params, MD, MT, QE, DT_ENERGY_FUS
)


def dt_reactivity_bosch_hale(Ti_keV):
    """
    Bosch-Hale 参数化的 D-T 反应速率 <σv> [m³/s]。

    参考：
        H.-S. Bosch, G.M. Hale, Nucl. Fusion 32 (1992) 611.

    参数
    ------
    Ti_keV : float or ndarray
        离子温度 [keV]。

    返回
    ------
    sigmav : float or ndarray
        反应速率参数 [m³/s]。
    """
    Ti = np.asarray(Ti_keV, dtype=float)
    Ti = np.clip(Ti, 0.1, 100.0)

    # Bosch-Hale 系数 (DT)
    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.35000e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5
    bg = 34.3827
    mr = 1124656.0  # m_r c² [keV]

    theta = Ti / (1.0 - Ti * (c2 + Ti * (c4 + Ti * c6)) /
                   (1.0 + Ti * (c3 + Ti * (c5 + Ti * c7))))
    xi = (bg * bg / (4.0 * theta)) ** (1.0 / 3.0)

    sigmav = c1 * theta * np.sqrt(xi / (mr * Ti ** 3)) * np.exp(-3.0 * xi)
    return sigmav


def simplified_reactivity(Ti_keV):
    """
    简化 D-T 反应速率（避免数值溢出，用于快速计算）。

    公式
    ----
        <σv> = 1e-18 · exp( -18.0 / Ti^{0.35} )   [m³/s]

    参数
    ------
    Ti_keV : float or ndarray
        离子温度 [keV]。

    返回
    ------
    sigmav : float or ndarray
        反应速率 [m³/s]。
    """
    Ti = np.asarray(Ti_keV, dtype=float)
    Ti = np.clip(Ti, 0.2, 200.0)
    return 1.0e-18 * np.exp(-18.0 / (Ti ** 0.35))


def fusion_derivative(t, y, k_eff, tau_p, tau_He, S_D, S_T):
    """
    D-T 聚变反应动力学导数。

    参数
    ------
    t : float
        时间 [s]。
    y : ndarray, shape (4,)
        [n_D, n_T, n_He, n_n] 数密度 [m^-3]。
    k_eff : float
        有效反应速率 [m³/s]。
    tau_p : float
        燃料粒子约束时间 [s]。
    tau_He : float
        灰粒子约束时间 [s]。
    S_D, S_T : float
        燃料注入源 [m^-3/s]。

    返回
    ------
    dydt : ndarray, shape (4,)
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (4,):
        raise ValueError("状态向量 y 必须为 4 维 [n_D, n_T, n_He, n_n]")
    nD, nT, nHe, nn = y

    # 确保非负
    nD = max(nD, 0.0)
    nT = max(nT, 0.0)
    nHe = max(nHe, 0.0)
    nn = max(nn, 0.0)

    reaction_rate = k_eff * nD * nT

    dnD_dt = -reaction_rate + S_D - nD / tau_p
    dnT_dt = -reaction_rate + S_T - nT / tau_p
    dnHe_dt = reaction_rate - nHe / tau_He
    dnn_dt = reaction_rate - nn / tau_He

    return np.array([dnD_dt, dnT_dt, dnHe_dt, dnn_dt], dtype=float)


def simulate_fusion_burn(fusion_params=None, Ti_keV=15.0, n_steps=2000):
    """
    模拟托卡马克中 D-T 聚变燃烧过程。

    参数
    ------
    fusion_params : dict or None
    Ti_keV : float
        等离子体离子温度 [keV]（假设空间均匀）。
    n_steps : int
        RK2 积分步数。

    返回
    ------
    t_arr : ndarray
    y_arr : ndarray, shape (n_steps+1, 4)
    P_fus_arr : ndarray
        聚变功率密度历史 [W/m³]。
    Q_factor : ndarray
        增益因子 Q = P_fus / P_heat 历史。
    """
    if fusion_params is None:
        fusion_params = get_fusion_params()

    t0 = fusion_params["t0"]
    y0 = fusion_params["y0"].copy()
    tstop = fusion_params["tstop"]

    # 修正初始条件为4维
    if len(y0) == 3:
        y0 = np.array([y0[0], y0[1], y0[2], 0.0])

    k_eff = simplified_reactivity(Ti_keV)
    tau_p = 2.0      # 燃料约束时间 [s]
    tau_He = 5.0     # He 灰约束时间 [s]
    S_D = 5.0e18     # D 注入源 [m^-3/s]
    S_T = 5.0e18     # T 注入源 [m^-3/s]
    P_heat = 5.0e5   # 外部加热功率密度 [W/m³]

    def deriv(t, y):
        return fusion_derivative(t, y, k_eff, tau_p, tau_He, S_D, S_T)

    # 使用简单欧拉法（足够用于演示动力学）
    h = (tstop - t0) / n_steps
    t_arr = np.linspace(t0, tstop, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, 4))
    y_arr[0, :] = y0

    for n in range(n_steps):
        yn = y_arr[n, :]
        dydt = deriv(t_arr[n], yn)
        y_arr[n + 1, :] = yn + h * dydt
        # 非负截断
        y_arr[n + 1, :] = np.maximum(y_arr[n + 1, :], 0.0)

    # 聚变功率密度
    nD = y_arr[:, 0]
    nT = y_arr[:, 1]
    P_fus_arr = nD * nT * k_eff * DT_ENERGY_FUS * QE / 4.0

    # 增益因子 Q
    Q_factor = np.zeros_like(P_fus_arr)
    nonzero = P_heat > 1e-30
    Q_factor[nonzero] = P_fus_arr[nonzero] / P_heat

    return t_arr, y_arr, P_fus_arr, Q_factor


def compute_bremsstrahlung(n_e, T_e_eV, Z_eff=1.8):
    """
    计算轫致辐射功率损失密度。

    公式
    ----
        P_brem = C_b · Z_eff · n_e² · √T_e   [W/m³]

        C_b = 1.69e-38 [W·m³/√eV]

    参数
    ------
    n_e : float or ndarray
        电子密度 [m^-3]。
    T_e_eV : float or ndarray
        电子温度 [eV]。
    Z_eff : float
        有效电荷数。

    返回
    ------
    P_brem : float or ndarray
        轫致辐射功率密度 [W/m³]。
    """
    C_b = 1.69e-38
    T_e = np.asarray(T_e_eV)
    T_e_safe = np.where(T_e < 1.0, 1.0, T_e)
    return C_b * Z_eff * np.asarray(n_e) ** 2 * np.sqrt(T_e_safe)


def compute_alpha_heating(n_e, Ti_keV):
    """
    计算 α 粒子加热功率密度。

    公式
    ----
        P_α = n_D n_T <σv> E_α
        在 n_D = n_T = n_e/2 假设下：
        P_α = (n_e² / 4) <σv>(T_i) E_α

    参数
    ------
    n_e : float or ndarray
        电子密度 [m^-3]。
    Ti_keV : float or ndarray
        离子温度 [keV]。

    返回
    ------
    P_alpha : float or ndarray
        α 加热功率密度 [W/m³]。
    """
    E_alpha = 3.52e6  # [eV]
    sigmav = simplified_reactivity(Ti_keV)
    return 0.25 * np.asarray(n_e) ** 2 * sigmav * E_alpha * QE


def lawson_criterion(Ti_keV, eta=0.3):
    """
    计算给定温度下的 Lawson 判据 nτ_E。

    公式
    ----
        n τ_E ≥ (3 k_B T_i) / ( (η / 4 + (1-η)/4 ) <σv> E_fus - ... )

    简化 nτ_E 判据（D-T，T_i in keV）：
        nτ_E ≥ 10^{20} · f(T_i)   [s·m^{-3}]

    参数
    ------
    Ti_keV : float or ndarray
        离子温度 [keV]。
    eta : float
        能量转换效率。

    返回
    ------
    ntau : float or ndarray
        Lawson 判据值 [s·m^{-3}]。
    """
    Ti = np.asarray(Ti_keV)
    sigmav = simplified_reactivity(Ti)
    # 简化 Lawson 判据：3 n k_B T / (P_fus - P_loss) ...
    # 采用近似公式
    E_fus_J = DT_ENERGY_FUS * QE
    Ti_J = Ti * 1e3 * QE
    ntau = 3.0 * Ti_J / (0.25 * sigmav * E_fus_J * eta)
    return ntau
