"""
光合生化模型模块：基于 Farquhar-von Caemmerer-Berry (FvCB) 模型，
结合 diff_center 中心差分计算温度敏感性导数。

核心公式：
  1. 净光合速率：
      A_n = min(W_c, W_j) - R_d

  2. Rubisco 限制速率：
      W_c = V_cmax * (C_i - Gamma_star) / (C_i + K_c * (1 + O_i / K_o))

  3. RuBP 再生限制速率：
      W_j = J * (C_i - Gamma_star) / (4*C_i + 8*Gamma_star)

  4. 电子传递速率 J（非矩形双曲线）：
      theta * J^2 - J * (I_abs * alpha + J_max) + I_abs * alpha * J_max = 0
      J = ( (I_abs*alpha + J_max) - sqrt((I_abs*alpha + J_max)^2 - 4*theta*I_abs*alpha*J_max) ) / (2*theta)

  5. 温度响应（Arrhenius + 高温失活）：
      V_cmax(T) = V_cmax_25 * exp( E_a*(T-298.15)/(298.15*R*T) )
                  * ( 1 + exp( (298.15*dS - H_d)/(298.15*R) ) )
                  / ( 1 + exp( (T*dS - H_d)/(R*T) ) )

  6. 中心差分温度敏感性：
      dA_n/dT ≈ (A_n(T+h) - A_n(T-h)) / (2h)
"""
import numpy as np

R = 8.314  # J/(mol*K)


def arrhenius_with_deactivation(value_25, T, E_a, dS, H_d):
    """
    Arrhenius 温度响应函数（含高温失活）。
    T: 温度 (K)
    E_a: 活化能 (J/mol)
    dS: 熵变 (J/(mol*K))
    H_d: 失活焓 (J/mol)
    """
    T = float(T)
    if T <= 0:
        return 0.0
    term1 = np.exp(E_a * (T - 298.15) / (298.15 * R * T))
    term2_num = 1.0 + np.exp((298.15 * dS - H_d) / (298.15 * R))
    term2_den = 1.0 + np.exp((T * dS - H_d) / (R * T))
    return value_25 * term1 * term2_num / max(term2_den, 1e-14)


def electron_transport_rate(i_abs, j_max, alpha_e=0.425, theta=0.7):
    """
    非矩形双曲线电子传递速率。
    i_abs: 吸收光强 (umol/m^2/s)
    j_max: 最大电子传递速率 (umol/m^2/s)
    """
    a = theta
    b = -(i_abs * alpha_e + j_max)
    c = i_abs * alpha_e * j_max
    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        disc = 0.0
    j = (-b - np.sqrt(disc)) / (2.0 * a)
    return max(j, 0.0)


def farquhar_photosynthesis(ci, oi, t_k, i_abs,
                            vcmax_25=80.0, jmax_25=136.0,
                            rd_25=1.2, kc_25=404.9, ko_25=278.4,
                            gamma_star_25=36.9,
                            Ea_vcmax=65330.0, Ea_jmax=43540.0,
                            Ea_rd=46390.0, Ea_kc=79430.0,
                            Ea_ko=36380.0, Ea_gamma=37830.0,
                            dS_vcmax=485.0, Hd_vcmax=150000.0,
                            dS_jmax=495.0, Hd_jmax=152000.0):
    """
    FvCB 模型计算净光合速率 A_n (umol CO2 / m^2 / s)。
    ci: 胞间 CO2 (umol/mol)
    oi: 胞间 O2 (mmol/mol)
    t_k: 叶片温度 (K)
    i_abs: 吸收光强 (umol/m^2/s)
    """
    # 参数温度修正
    vcmax = arrhenius_with_deactivation(vcmax_25, t_k, Ea_vcmax, dS_vcmax, Hd_vcmax)
    jmax = arrhenius_with_deactivation(jmax_25, t_k, Ea_jmax, dS_jmax, Hd_jmax)
    rd = arrhenius_with_deactivation(rd_25, t_k, Ea_rd, 490.0, 150000.0)
    kc = arrhenius_with_deactivation(kc_25, t_k, Ea_kc, 650.0, 150000.0)
    ko = arrhenius_with_deactivation(ko_25, t_k, Ea_ko, 650.0, 150000.0)
    gamma_star = arrhenius_with_deactivation(gamma_star_25, t_k, Ea_gamma, 650.0, 150000.0)

    # TODO: 实现 Rubisco 限制速率 W_c 与 RuBP 限制速率 W_j 的计算，
    # 并返回净光合速率 A_n = min(W_c, W_j) - R_d
    # 关键公式：
    #   W_c = V_cmax * (C_i - Gamma_star) / (C_i + K_c * (1 + O_i / K_o))
    #   W_j = J * (C_i - Gamma_star) / (4*C_i + 8*Gamma_star)
    #   J 由 electron_transport_rate(i_abs, jmax) 计算
    raise NotImplementedError("Hole 1: 请补全 FvCB 光合模型核心公式")


def temperature_sensitivity_centered(ci, oi, t_k, i_abs, h=0.5, **kwargs):
    """
    中心差分计算 A_n 对温度的敏感性 dA_n/dT。
    h: 温度步长 (K)
    """
    an_plus, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k + h, i_abs, **kwargs)
    an_minus, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k - h, i_abs, **kwargs)
    d_adt = (an_plus - an_minus) / (2.0 * h)
    return d_adt


def canopy_photosynthesis_integrated(z_levels, lai_profile, radiation_profile,
                                     temperature_profile, ci, oi,
                                     vcmax_25=80.0, jmax_25=136.0, **kwargs):
    """
    在冠层垂直剖面上积分计算总光合速率。
    返回: A_total (umol/m^2/s per ground area)
    """
    dz = np.diff(z_levels, prepend=0.0)
    dz = np.maximum(dz, 1e-6)
    a_total = 0.0
    for i in range(len(z_levels)):
        lai = lai_profile[i]
        i_abs = radiation_profile[i] * 0.85  # 假设 85% 吸收
        t_k = temperature_profile[i] + 273.15
        an, _, _, _, _ = farquhar_photosynthesis(ci, oi, t_k, i_abs,
                                                  vcmax_25=vcmax_25,
                                                  jmax_25=jmax_25, **kwargs)
        a_total += max(an, 0.0) * lai * dz[i]
    return a_total
