"""
吸积盘物理核心模块

整合所有输入项目的科学计算能力，实现：
  1. Shakura-Sunyaev 薄盘径向结构方程
  2. 磁离心喷流（Blandford-Payne）机制
  3. 黑洞引力势与广义相对论修正
  4. 粘滞角动量输运
  5. 盘不稳定性和热平衡
"""
import numpy as np


# ===========================
# Physical Constants
# ===========================

G_GRAV = 6.67430e-11          # m^3 kg^-1 s^-2
C_LIGHT = 2.99792458e8        # m/s
M_SUN = 1.98847e30            # kg
SIGMA_SB = 5.670374419e-8     # W m^-2 K^-4
K_BOLTZMANN = 1.380649e-23    # J/K
MP = 1.6726219e-27            # kg
MU = 0.6                      # 平均分子量（完全电离）
GAMMA_AD = 5.0 / 3.0          # 绝热指数


def keplerian_angular_velocity(r, M_bh):
    """
    开普勒角速度：
        Omega_K(r) = sqrt(G*M_bh / r^3)

    参数:
        r: 半径（m）
        M_bh: 黑洞质量（kg）

    返回:
        Omega_K: 角速度（rad/s）
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)
    return np.sqrt(G_GRAV * M_bh / r ** 3)


def sound_speed(T, mu=MU, gamma=GAMMA_AD):
    """
    等温声速：
        c_s = sqrt(gamma * k_B * T / (mu * m_p))

    参数:
        T: 温度（K）
        mu: 平均分子量
        gamma: 绝热指数

    返回:
        c_s: 声速（m/s）
    """
    return np.sqrt(gamma * K_BOLTZMANN * T / (mu * MP))


def scale_height(r, M_bh, T, mu=MU):
    """
    吸积盘标高：
        H(r) = c_s / Omega_K = sqrt(gamma*k_B*T*r^3 / (mu*m_p*G*M_bh))

    这是薄盘理论的核心量，表征盘的垂直厚度。
    """
    cs = sound_speed(T, mu)
    omega = keplerian_angular_velocity(r, M_bh)
    return cs / omega


def shakura_sunyaev_sigma(r, M_dot, M_bh, alpha, mu=MU):
    """
    Shakura-Sunyaev 薄盘表面密度（无量纲化形式）。

    标准 SS 解：
        Sigma(r) = (M_dot / (3*pi*nu)) * [1 - sqrt(r_in/r)]

    其中运动粘度 nu = alpha * c_s * H。

    对于辐射主导区（内区），温度满足：
        T^4 = (3*G*M_bh*M_dot) / (8*pi*sigma_SB*r^3) * [1 - sqrt(r_in/r)]

    参数:
        r: 半径数组
        M_dot: 质量吸积率（kg/s）
        M_bh: 黑洞质量（kg）
        alpha: 粘滞参数（~0.01-0.1）
        mu: 平均分子量

    返回:
        Sigma: 表面密度（kg/m^2）
        T: 温度（K）
        H: 标高（m）
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)

    # ISCO 半径（Schwarzschild）
    r_isco = 6.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r_eff = np.maximum(r, r_isco * 1.001)

    # 温度（辐射主导）
    factor = np.clip(1.0 - np.sqrt(r_isco / r_eff), 1e-15, 1.0)
    T = ((3.0 * G_GRAV * M_bh * M_dot) /
         (8.0 * np.pi * SIGMA_SB * r_eff ** 3) * factor) ** 0.25
    T = np.where(T < 1.0, 1.0, T)

    # 标高
    H = scale_height(r_eff, M_bh, T, mu)
    H = np.where(H < 1e-10, 1e-10, H)

    # 声速
    cs = sound_speed(T, mu)
    cs = np.where(cs < 1e-10, 1e-10, cs)

    # 运动粘度
    nu = alpha * cs * H
    nu = np.where(nu < 1e-30, 1e-30, nu)

    # 表面密度
    Sigma = M_dot / (3.0 * np.pi * nu) * factor
    Sigma = np.where(Sigma < 0, 0.0, Sigma)

    return Sigma, T, H


def viscous_torque(r, Sigma, M_bh, alpha, mu=MU):
    """
    粘滞力矩：
        G(r) = 3*pi * nu * Sigma * r^2 * Omega_K

    这是驱动角动量向外输运的核心机制。
    """
    r = np.asarray(r, dtype=np.float64)
    omega = keplerian_angular_velocity(r, M_bh)
    Sigma = np.asarray(Sigma)

    # 标高和温度（近似）
    # 先估算温度
    cs_approx = alpha * omega * r * 0.1  # 简化估计
    T_approx = cs_approx ** 2 * mu * MP / (GAMMA_AD * K_BOLTZMANN)
    H = scale_height(r, M_bh, T_approx, mu)
    cs = sound_speed(T_approx, mu)
    nu = alpha * cs * H

    G = 3.0 * np.pi * nu * Sigma * r ** 2 * omega
    return G


def schwarzschild_potential(r, M_bh):
    """
    Schwarzschild 引力势（牛顿近似）：
        Phi(r) = -G*M_bh / r

    参数:
        r: 半径（m）
        M_bh: 黑洞质量（kg）

    返回:
        Phi: 引力势（J/kg）
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.abs(r) < 1e-3, 1e-3, r)
    return -G_GRAV * M_bh / r


def schwarzschild_metric_correction(r, M_bh):
    """
    Schwarzschild 度规对牛顿势的修正因子（后牛顿近似）：
        Phi_GR = Phi_Newton * (1 - 3*G*M_bh/(r*c^2))

    这是 Paczynski-Wiita 势的简化形式。
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.abs(r) < 1e-3, 1e-3, r)
    correction = 1.0 - 3.0 * G_GRAV * M_bh / (r * C_LIGHT ** 2)
    return correction


def paczynski_wiita_potential(r, M_bh):
    """
    Paczynski-Wiita 伪牛顿势：
        Phi_PW = -G*M_bh / (r - r_s)

    其中 r_s = 2*G*M_bh/c^2 为 Schwarzschild 半径。
    该势精确重现了 Schwarzschild 度规的 ISCO 和
    光子轨道半径。
    """
    r_s = 2.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r, r_s * 1.001)
    return -G_GRAV * M_bh / (r_safe - r_s)


def jet_launching_criterion(r, B_z, rho, M_bh):
    """
    Blandford-Payne 磁离心喷流判据。

    当 Alfven 半径处的离心加速度超过引力束缚时，
    磁力线可将物质加速到无穷远。

    判据（简化形式）：
        v_A / v_esc > 1

    其中 Alfven 速度：
        v_A = B_z / sqrt(4*pi*rho)

    逃逸速度：
        v_esc = sqrt(2*G*M_bh / r)

    参数:
        r: 半径（m）
        B_z: 垂直磁场（T）
        rho: 盘密度（kg/m^3）
        M_bh: 黑洞质量（kg）

    返回:
        launched: bool 数组，True 表示喷流可发射
        v_A: Alfven 速度
        v_esc: 逃逸速度
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)

    # TODO(Hole-1): 根据磁流体动力学计算 Alfven 速度
    # 科学公式: v_A = B / sqrt(4*pi*rho)
    v_A = None  # 需恢复科学公式
    v_esc = np.sqrt(2.0 * G_GRAV * M_bh / r)

    launched = v_A > v_esc
    return launched, v_A, v_esc


def magnetic_braking_torque(r, B_phi, B_r, Sigma, M_bh):
    """
    磁制动扭矩（MRI/盘风机制）：
        T_mag = (r^2 * B_r * B_phi) / (2*pi)

    参数:
        r: 半径
        B_phi: 环向磁场
        B_r: 径向磁场
        Sigma: 表面密度
        M_bh: 黑洞质量

    返回:
        T_mag: 磁扭矩（N*m/m）
    """
    r = np.asarray(r, dtype=np.float64)
    return r ** 2 * B_r * B_phi / (2.0 * np.pi)


def disk_spectrum_nu(nu_freq, r_in, r_out, M_dot, M_bh):
    """
    吸积盘多色黑体光谱（Shakura-Sunyaev）。

    光谱能量分布：
        L_nu = integral_{r_in}^{r_out} 2*pi*r * B_nu(T(r)) dr

    其中 Planck 函数：
        B_nu = (2*h*nu^3/c^2) / (exp(h*nu/(k_B*T)) - 1)

    参数:
        nu_freq: 频率数组（Hz）
        r_in, r_out: 内外半径（m）
        M_dot: 质量吸积率（kg/s）
        M_bh: 黑洞质量（kg）

    返回:
        L_nu: 光谱光度（W/Hz）
    """
    h_planck = 6.62607015e-34
    nu = np.asarray(nu_freq, dtype=np.float64)

    # 径向网格
    n_r = 100
    r = np.linspace(r_in, r_out, n_r)
    dr = r[1] - r[0]

    # 温度
    Sigma, T, H = shakura_sunyaev_sigma(r, M_dot, M_bh, alpha=0.1)

    # 计算光谱
    L_nu = np.zeros_like(nu)
    for i in range(n_r):
        T_i = T[i]
        if T_i < 1.0 or np.isnan(T_i) or np.isinf(T_i):
            continue
        x = h_planck * nu / (K_BOLTZMANN * T_i)
        # 避免溢出
        x = np.clip(x, 1e-10, 700.0)
        exp_x = np.exp(x)
        B_nu = (2.0 * h_planck * nu ** 3 / C_LIGHT ** 2) / (exp_x - 1.0)
        B_nu = np.where(np.isfinite(B_nu), B_nu, 0.0)
        L_nu += 2.0 * np.pi * r[i] * B_nu * dr

    L_nu = np.where(np.isfinite(L_nu), L_nu, 0.0)
    return L_nu


def disk_instability_criterion(Sigma, T_actual, r, M_bh, alpha, mu=MU):
    """
    吸积盘热不稳定性判据（热平衡分析）。

    当冷却时标 t_cool 远短于粘滞时标 t_visc 时，盘可能
    出现热不稳定（极限环行为）。

    时标：
        t_visc = r^2 / nu
        t_cool = Sigma * c_s^2 / (2*sigma_SB*T^4)

    参数:
        Sigma: 表面密度
        T_actual: 实际温度（K）
        r: 半径
        M_bh: 黑洞质量
        alpha: 粘滞参数
        mu: 平均分子量

    返回:
        unstable: bool 数组
        t_visc: 粘滞时标
        t_cool: 冷却时标
    """
    r = np.asarray(r, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    T = np.asarray(T_actual, dtype=np.float64)
    T = np.where(T < 1.0, 1.0, T)

    H = scale_height(r, M_bh, T, mu)
    cs = sound_speed(T, mu)
    nu = alpha * cs * H
    nu = np.where(nu < 1e-30, 1e-30, nu)

    t_visc = r ** 2 / nu
    t_visc = np.where(t_visc < 1e-10, 1e-10, t_visc)

    t_cool = Sigma * cs ** 2 / (2.0 * SIGMA_SB * T ** 4)
    t_cool = np.where(t_cool < 1e-10, 1e-10, t_cool)

    # 使用比值判据：当冷却远快于粘滞扩散时不稳定
    ratio = t_cool / t_visc
    unstable = ratio < 1e-4
    return unstable, t_visc, t_cool


def compute_radial_velocity(Sigma, r, M_bh, alpha, M_dot):
    """
    吸积盘径向速度（SS 解）：
        v_r = - (3*nu / (2*r)) * [1 - sqrt(r_in/r)] / [1 - (2/3)*sqrt(r_in/r)]

    参数:
        Sigma: 表面密度
        r: 半径
        M_bh: 黑洞质量
        alpha: 粘滞参数
        M_dot: 质量吸积率

    返回:
        v_r: 径向速度（m/s），负值表示向内吸积
    """
    r = np.asarray(r, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    r_isco = 6.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r_eff = np.maximum(r, r_isco)

    # 估算 nu
    omega = keplerian_angular_velocity(r_eff, M_bh)
    cs_approx = alpha * omega * r_eff * 0.1
    T_approx = cs_approx ** 2 * MU * MP / (GAMMA_AD * K_BOLTZMANN)
    H = scale_height(r_eff, M_bh, T_approx, MU)
    cs = sound_speed(T_approx, MU)
    nu = alpha * cs * H

    sqrt_ratio = np.sqrt(r_isco / r_eff)
    numerator = 1.0 - sqrt_ratio
    denominator = 1.0 - (2.0 / 3.0) * sqrt_ratio
    denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)

    v_r = -1.5 * nu / r_eff * numerator / denominator
    return v_r
