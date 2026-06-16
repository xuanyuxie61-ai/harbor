"""
reion_cosmology.py
==================
物理常数、宇宙学参数与基础标度关系

本模块集中管理再电离历史模拟所需的物理与宇宙学常数, 并提供红移-时间变换、
背景温度演化等基础标度关系. 所有量均采用 cgs 高斯单位制, 仅红移 z 与无量纲
密度参数 Omega 为纯数.

主要常量 (CODATA 2018 / Planck 2018):
    c_light          : 真空光速              [cm/s]
    k_B_cgs          : 玻尔兹曼常数          [erg/K]
    m_H_cgs          : 氢原子质量            [g]
    m_e_cgs          : 电子质量              [g]
    e_charge         : 基本电荷              [esu]
    h_planck         : 普朗克常数            [erg*s]
    sigma_T          : 汤姆孙散射截面        [cm^2]
    sigma_B          : Stefan-Boltzmann 常数 [erg cm^-2 s^-1 K^-4]
    alpha_B_H        : 氢 Case-B 复合系数    [cm^3/s] (T=10^4 K 近似)
    alpha_B_He       : 氦 Case-B 复合系数    [cm^3/s]
    E_ion_H          : 氢电离能              [erg] (13.6 eV)
    E_ion_He         : 氦 I 电离能           [erg]
    E_ion_HeII       : 氦 II 电离能          [erg]
    H100             : h*100 km/s/Mpc
    OmBh2, OmCh2, OmL: Planck 2018 密度参数
    A_s, n_s         : 原初功率谱振幅与谱指数
    T_CMB0           : CMB 当前温度          [K]
    Y_He             : 氦丰度 (质量分数)

标度关系:
    H(z)              : Hubble 参数随红移演化
    t_cosmic(z)       : 宇宙年龄 (z=0 起算)
    z_from_t(t)       : 由年龄反推红移
    T_CMB(z)          : CMB 温度随红移演化
    rho_crit(z)       : 临界密度
    rho_baryon(z)     : 重子数密度
    nH_bar(z)         : 平均氢数密度 (考虑氦丰度)
"""

import numpy as np

# =============================================================================
# 基本物理常数 (CODATA 2018, cgs 单位)
# =============================================================================
c_light     = 2.99792458e10       # 真空光速 [cm/s]
k_B_cgs     = 1.380649e-16        # 玻尔兹曼常数 [erg/K]
m_H_cgs     = 1.6735575e-24       # 氢原子质量 [g]
m_e_cgs     = 9.10938e-28         # 电子质量 [g]
e_charge    = 4.80320425e-10      # 基本电荷 [esu]
h_planck    = 6.62607015e-27      # 普朗克常数 [erg*s]
hbar        = h_planck / (2.0 * np.pi)
sigma_T     = 6.6524587158e-25    # 汤姆孙散射截面 [cm^2]
sigma_B     = 5.670374419e-5      # Stefan-Boltzmann 常数 [erg cm^-2 s^-1 K^-4]
a_rad       = 4.0 * sigma_B / c_light  # 辐射常数 [erg cm^-3 K^-4]
G_grav      = 6.67430e-8          # 万有引力常数 [cm^3 g^-1 s^-2]

# =============================================================================
# 原子物理常数 (电离能, 复合系数)
# =============================================================================
eV_to_erg   = 1.602176634e-12     # 1 eV = 1.602e-12 erg
E_ion_H     = 13.598 * eV_to_erg  # 氢基态电离能 [erg]
E_ion_He    = 24.587 * eV_to_erg  # 氦 I 第一电离能 [erg]
E_ion_HeII  = 54.418 * eV_to_erg  # 氦 II 电离能 [erg]
alpha_B_H   = 2.59e-13            # 氢 Case-B 复合系数 T=10^4 K [cm^3/s]
alpha_B_He  = 1.53e-12            # 氦 Case-B 复合系数 T=10^4 K [cm^3/s]

# =============================================================================
# 宇宙学参数 (Planck 2018 + 再电离典型取值)
# =============================================================================
H100        = 1.0e5 / 3.08567758e24   # 100 km/s/Mpc → 1/s
hubble      = 0.6766                  # 无量纲 Hubble 参数 h
OmBh2       = 0.02242                 # 重子密度参数 Omega_b * h^2
OmCh2       = 0.11933                 # 冷暗物质密度参数 Omega_c * h^2
OmL         = 0.6889                  # 暗能量密度参数 Omega_Lambda
OmM         = OmBh2 / hubble**2 + OmCh2 / hubble**2  # 物质密度参数
Omb         = OmBh2 / hubble**2       # 重子密度参数
A_s         = 2.100e-9                # 原初功率谱振幅
n_s         = 0.9665                  # 原初谱指数
tau_e_obs   = 0.054                   # Planck 2018 汤姆孙散射光学深度
sigma_tau   = 0.007                   # tau 观测误差

# CMB 当前温度
T_CMB0      = 2.7255                  # [K]

# 氦丰度 ( primordial helium mass fraction from BBN)
Y_He        = 0.245

# =============================================================================
# 宇宙学常用派生量
# =============================================================================
Mpc_to_cm   = 3.08567758e24           # 1 Mpc = 3.086e24 cm
Gyr_to_s    = 3.15576e16              # 1 Gyr  = 3.156e16 s
yr_to_s     = 3.15576e7               # 1 yr   = 3.156e7  s

# 临界密度 (z=0)
rho_crit_0  = 3.0 * (hubble * H100)**2 / (8.0 * np.pi * G_grav)  # [g/cm^3]

# 重子数密度 (z=0)
n_baryon_0  = Omb * rho_crit_0 / m_H_cgs  # [cm^-3]

# 平均氢数密度 (考虑氦丰度)
nH_0        = n_baryon_0 * (1.0 - Y_He)   # [cm^-3]


# =============================================================================
# 核心标度函数
# =============================================================================
def Hubble_parameter(z):
    """Hubble 参数 H(z) [s^-1].

    对于平坦 LCDM 宇宙:
        H(z) = H_0 * sqrt( Omega_m * (1+z)^3 + Omega_Lambda )

    Parameters
    ----------
    z : float or array
        红移 (z >= 0)

    Returns
    -------
    H : float or array
        Hubble 参数 [s^-1]
    """
    z = np.asarray(z, dtype=float)
    z = np.maximum(z, 0.0)
    return hubble * H100 * np.sqrt(OmM * (1.0 + z)**3 + OmL)


def dt_dz(z):
    """红移对宇宙时的导数 dt/dz [s].

    由 1+z = a_0 / a, 有
        dt/dz = -1 / ((1+z) * H(z))

    取绝对值使 dt_dz > 0 (用于红移→时间积分).
    """
    z = np.asarray(z, dtype=float)
    z = np.maximum(z, 1.0e-10)
    return 1.0 / ((1.0 + z) * Hubble_parameter(z))


def t_cosmic(z):
    """宇宙年龄 (从大爆炸到红移 z 的时差) [s].

    通过对 dt/dz 从 z 积分到 z_max = 1100 (复合时期) 获得:
        t(z) = int_{z}^{z_max} |dt/dz'| dz'
    """
    from scipy.integrate import quad
    z = np.asarray(z, dtype=float)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)
    t_arr = np.zeros_like(z)
    for i, zi in enumerate(z.flat):
        res, _ = quad(dt_dz, zi, 1100.0, limit=200)
        t_arr.flat[i] = res
    return t_arr[0] if scalar_input else t_arr


def z_from_t(t_target, z_lo=5.0, z_hi=30.0, tol=1.0e-6):
    """由宇宙时反推红移 (二分法).

    Parameters
    ----------
    t_target : float
        目标宇宙时 [s]
    z_lo, z_hi : float
        搜索红移区间
    tol : float
        时间相对容差
    """
    t_hi = t_cosmic(z_lo)
    t_lo = t_cosmic(z_hi)
    if not (t_lo <= t_target <= t_hi):
        # 越界时返回最近端点
        if t_target < t_lo:
            return z_hi
        return z_lo
    for _ in range(100):
        z_mid = 0.5 * (z_lo + z_hi)
        t_mid = t_cosmic(z_mid)
        if abs(t_mid - t_target) / max(abs(t_target), 1.0e-30) < tol:
            return z_mid
        if t_mid < t_target:
            z_hi = z_mid
        else:
            z_lo = z_mid
    return 0.5 * (z_lo + z_hi)


def T_CMB(z):
    """CMB 温度随红移演化 [K].

        T_CMB(z) = T_CMB0 * (1+z)
    """
    return T_CMB0 * (1.0 + np.asarray(z, dtype=float))


def rho_crit(z):
    """临界密度 [g/cm^3].

        rho_crit(z) = 3 H(z)^2 / (8 pi G)
    """
    H_z = Hubble_parameter(z)
    return 3.0 * H_z**2 / (8.0 * np.pi * G_grav)


def nH_bar(z):
    """平均氢数密度 [cm^-3] (均匀宇宙背景).

        n_H(z) = n_H0 * (1+z)^3
    """
    return nH_0 * (1.0 + np.asarray(z, dtype=float))**3


def jeans_length_comoving(z, T_gas, xHII=1.0):
    """气体 Jeans 长度 (共动) [cm (共动)].

    对于部分电离气体, 平均分子量:
        mu = 1 / (2*xHII + 1 - Y_He) * (1/(1-Y_He))

    声速:
        c_s = sqrt(k_B * T_gas / (mu * m_H))

    Jeans 长度:
        lambda_J = c_s * sqrt(pi / (G * rho_m))

    Parameters
    ----------
    z : float
        红移
    T_gas : float
        气体温度 [K]
    xHII : float
        电离分数 (0-1)
    """
    T_gas = max(T_gas, 10.0)
    xHII = np.clip(xHII, 0.0, 1.0)
    # 平均分子量 (考虑部分电离)
    mu_inv = (2.0 * xHII + 1.0 - Y_He) / (1.0 - Y_He)
    mu = 1.0 / max(mu_inv, 1.0e-10)
    c_s = np.sqrt(k_B_cgs * T_gas / (mu * m_H_cgs))
    rho_m = rho_crit(z) * OmM * (1.0 + z)**3
    rho_m = max(rho_m, 1.0e-40)
    return c_s * np.sqrt(np.pi / (G_grav * rho_m)) * (1.0 + z)  # 转为共动


def comoving_to_proper(length_comoving, z):
    """共动长度 → 物理长度."""
    return np.asarray(length_comoving) / (1.0 + np.asarray(z, dtype=float))


def proper_to_comoving(length_proper, z):
    """物理长度 → 共动长度."""
    return np.asarray(length_proper) * (1.0 + np.asarray(z, dtype=float))
