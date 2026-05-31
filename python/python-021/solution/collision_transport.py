"""
collision_transport.py
等离子体碰撞统计与输运系数计算。

核心物理模型：
  托卡马克等离子体中的库仑碰撞是各向异性的：
  小角度偏转占主导，累积形成扩散过程。

  1. 碰撞频率（Spitzer 公式）：
        ν_ei = (n_e Z_eff e⁴ ln Λ) / (3 (2π)^{3/2} ε₀² m_e^{1/2} (k_B T_e)^{3/2})

     其中 ln Λ 为 Coulomb 对数：
        ln Λ ≈ 31.3 - ln(√n_e / T_e)   (n_e in m^{-3}, T_e in eV)

  2. 碰撞时间：
        τ_e = 3 (2π)^{3/2} ε₀² √(m_e) (k_B T_e)^{3/2} / (n_e Z_eff e⁴ ln Λ)

  3. 平均自由程：
        λ_e = v_{th,e} / ν_ei
        v_{th,e} = √(2 k_B T_e / m_e)

  4. 速度空间扩散：
     在超球面 S^{m-1} 上，两个随机速度矢量夹角 θ 的统计特性
     决定了碰撞后的能量交换效率。

  5. 碰撞距离统计（类比矩形距离问题）：
     将磁面局部近似为矩形区域，计算粒子间平均碰撞距离
     的统计分布，用于估算输运步长。
"""

import numpy as np
from parameters import (
    EPS0, QE, ME, KB, N_E_AXIS, T_E_AXIS, Z_EFF
)


def coulomb_logarithm(n_e, T_e_eV):
    """
    计算 Coulomb 对数。

    公式
    ----
        ln Λ = 31.3 - ln( √n_e / T_e )    [n_e: m^{-3}, T_e: eV]

    参数
    ------
    n_e : float or ndarray
        电子密度 [m^-3]。
    T_e_eV : float or ndarray
        电子温度 [eV]。

    返回
    ------
    lnLambda : float or ndarray
        Coulomb 对数。
    """
    n_e = np.asarray(n_e)
    T_e = np.asarray(T_e_eV)
    T_e_safe = np.where(T_e < 1.0, 1.0, T_e)
    lnL = 31.3 - np.log(np.sqrt(n_e) / T_e_safe)
    return np.clip(lnL, 5.0, 25.0)


def electron_ion_collision_frequency(n_e, T_e_eV, Z_eff=Z_EFF):
    """
    电子-离子碰撞频率。

    公式
    ----
        ν_ei = (n_e Z_eff e⁴ ln Λ) / (3 (2π)^{3/2} ε₀² √m_e (k_B T_e)^{3/2})

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
    nu_ei : float or ndarray
        碰撞频率 [Hz]。
    """
    n_e = np.asarray(n_e)
    T_e = np.asarray(T_e_eV) * QE  # 转为 [J]
    T_e_safe = np.where(T_e < 1e-20, 1e-20, T_e)
    lnL = coulomb_logarithm(n_e, T_e_eV)

    numerator = n_e * Z_eff * (QE ** 4) * lnL
    denominator = 3.0 * (2.0 * np.pi) ** 1.5 * (EPS0 ** 2) * np.sqrt(ME) * (T_e_safe ** 1.5)
    nu = numerator / (denominator + 1e-50)
    return nu


def thermal_velocity_electron(T_e_eV):
    """
    电子热速度。

    公式
    ----
        v_{th,e} = √(2 k_B T_e / m_e)

    参数
    ------
    T_e_eV : float or ndarray
        电子温度 [eV]。

    返回
    ------
    v_th : float or ndarray
        热速度 [m/s]。
    """
    T_e_J = np.asarray(T_e_eV) * QE
    return np.sqrt(2.0 * KB * T_e_J / ME)


def mean_free_path(n_e, T_e_eV, Z_eff=Z_EFF):
    """
    电子平均自由程。

    公式
    ----
        λ_e = v_{th,e} / ν_ei

    参数
    ------
    n_e, T_e_eV, Z_eff

    返回
    ------
    mfp : float or ndarray
        平均自由程 [m]。
    """
    vth = thermal_velocity_electron(T_e_eV)
    nu = electron_ion_collision_frequency(n_e, T_e_eV, Z_eff)
    return vth / (nu + 1e-50)


def hypersphere_velocity_sampling(m_dim, n_samples, T_e_eV):
    """
    在 m 维超球面上采样随机速度方向，并统计夹角分布。

    基于原 hypersphere_angle 项目：
      在高维速度空间中，随机抽取两个单位矢量，
      计算其夹角 θ = arccos(|p·q|)。

    物理意义：
      在 m 维速度空间中，碰撞前后速度方向改变的统计特性。
      m=3 对应三维物理空间。

    公式
    ----
        p, q ~ Uniform(S^{m-1})
        cos θ = |p·q|
        E[cos θ] → 0  as m → ∞
        E[θ] → π/2  as m → ∞

    参数
    ------
    m_dim : int
        空间维数。
    n_samples : int
        采样数。
    T_e_eV : float
        电子温度 [eV]（用于标注统计条件）。

    返回
    ------
    stats : dict
        包含 cosθ 和 θ 的均值、标准差。
    """
    if m_dim < 2:
        raise ValueError("维数必须 ≥ 2")

    costs = np.zeros(n_samples)
    for i in range(n_samples):
        # 从标准正态分布采样并归一化 -> 均匀分布在球面上
        p = np.random.randn(m_dim)
        p /= np.linalg.norm(p)
        q = np.random.randn(m_dim)
        q /= np.linalg.norm(q)
        costs[i] = abs(np.dot(p, q))

    costs = np.clip(costs, 0.0, 1.0)
    thetas = np.arccos(costs)

    stats = {
        "dim": m_dim,
        "temperature_eV": T_e_eV,
        "cos_mean": float(np.mean(costs)),
        "cos_std": float(np.std(costs)),
        "theta_mean_rad": float(np.mean(thetas)),
        "theta_std_rad": float(np.std(thetas)),
        "theta_mean_deg": float(np.degrees(np.mean(thetas))),
    }
    return stats


def rectangle_collision_distance_stats(a, b, n_samples=100000):
    """
    矩形区域内随机点对距离统计（类比磁面局部patch）。

    基于原 rectangle_distance 项目：
      在 a × b 矩形内均匀随机选取两点，
      计算其欧氏距离 d = √((x1-x2)² + (y1-y2)²)。

    物理意义：
      磁面局部区域内粒子间的平均碰撞距离分布，
      用于估算步进输运模型的空间步长。

    参数
    ------
    a, b : float
        矩形边长 [m]。
    n_samples : int
        Monte Carlo 采样数。

    返回
    ------
    stats : dict
        距离统计量。
    """
    if a <= 0 or b <= 0:
        raise ValueError("矩形边长必须为正")

    x1 = np.random.uniform(0, a, n_samples)
    y1 = np.random.uniform(0, b, n_samples)
    x2 = np.random.uniform(0, a, n_samples)
    y2 = np.random.uniform(0, b, n_samples)

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    stats = {
        "a": a,
        "b": b,
        "mean_distance": float(np.mean(d)),
        "std_distance": float(np.std(d)),
        "max_distance": float(np.max(d)),
        "min_distance": float(np.min(d)),
        "median_distance": float(np.median(d)),
    }
    return stats


def compute_transport_coefficients(n_e, T_e_eV, B, Z_eff=Z_EFF, q=2.0, R0=6.2, a=2.0):
    """
    计算新经典与经典输运系数。

    公式
    ----
    经典扩散系数：
        D_cl = ν_ei ρ_e²
        ρ_e = m_e v_{th,e} / (e B)

    新经典扩散系数（简化香蕉区）：
        D_neo = q² ν_ei ρ_i² ε^{-3/2}
        ρ_i = √(2 m_i k_B T_i) / (Z_i e B)

    热导率：
        χ_e = D_cl · (m_e / m_i)^{-1/2}
        χ_i = D_neo

    参数
    ------
    n_e, T_e_eV, B : float
        电子密度、温度、磁场。
    Z_eff, q, R0, a : float

    返回
    ------
    coeffs : dict
        各类输运系数。
    """
    from parameters import MD

    nu_ei = electron_ion_collision_frequency(n_e, T_e_eV, Z_eff)
    vth_e = thermal_velocity_electron(T_e_eV)
    rho_e = ME * vth_e / (QE * B + 1e-30)

    # 假设 T_i ≈ T_e
    T_i_J = T_e_eV * QE
    vth_i = np.sqrt(2.0 * KB * T_i_J / MD)
    rho_i = MD * vth_i / (QE * B + 1e-30)

    epsilon = a / (R0 + 1e-20)
    epsilon_safe = max(epsilon, 1e-6)

    D_cl = nu_ei * rho_e ** 2
    D_neo = (q ** 2) * nu_ei * (rho_i ** 2) / (epsilon_safe ** 1.5)

    # 热导率
    chi_e = D_cl * np.sqrt(MD / ME)
    chi_i = D_neo

    return {
        "nu_ei_Hz": float(nu_ei),
        "lambda_e_m": float(vth_e / (nu_ei + 1e-50)),
        "rho_e_m": float(rho_e),
        "rho_i_m": float(rho_i),
        "D_classical_m2s": float(D_cl),
        "D_neoclassical_m2s": float(D_neo),
        "chi_e_m2s": float(chi_e),
        "chi_i_m2s": float(chi_i),
    }
