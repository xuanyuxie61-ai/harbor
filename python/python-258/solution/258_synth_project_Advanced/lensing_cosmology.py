"""
lensing_cosmology.py — 宇宙学距离与临界密度计算
===============================================

来源种子: 1201_de-ranit_GPP-ModelParamVariation (参数扫描),
          1115_sandyherho_inerOsci (ODE/数值积分)
科学角色: 计算宇宙学角直径距离、临界表面质量密度、
         红移分布函数和透镜效率因子。

核心公式:
=========
1. 共动距离:
   χ(z) = (c/H₀) ∫₀ᶻ dz' / E(z')
   其中 E(z) = H(z)/H₀ = √(Ω_m(1+z)³ + Ω_Λ)  (平坦 ΛCDM)

2. 角直径距离:
   D_A(z₁,z₂) = (1/(1+z₂)) × (c/H₀) × ∫_{z₁}^{z₂} dz'/E(z')

3. 临界表面质量密度:
   Σ_crit = (c²/(4πG)) × D_S / (D_L × D_LS)

4. 源红移分布 (Smail et al. 1994):
   n(z) ∝ z^α × exp(-(z/z₀)^β)
   典型值: α=2, β=1.5, z₀=z_m/√2

5. 透镜效率函数:
   g(z_s) = ∫₀^{z_s} n(z_l) (1 - D_L/D_S) dz_l
"""

import numpy as np
from lensing_config import COSMO, GRID


def E_z(z):
    """
    无量纲 Hubble 参数 E(z) = H(z)/H₀

    对于平坦 ΛCDM:
        E(z) = √(Ω_m(1+z)³ + Ω_Λ)

    Parameters
    ----------
    z : float or ndarray
        红移值 (z ≥ 0)

    Returns
    -------
    Ez : float or ndarray
        E(z) 值
    """
    z = np.asarray(z, dtype=np.float64)
    z = np.maximum(z, 0.0)
    Ez2 = COSMO.Omega_m * (1.0 + z)**3 + COSMO.Omega_L
    return np.sqrt(np.maximum(Ez2, 1.0e-30))


def integrand_comoving(z):
    """
    共动距离被积函数: 1/E(z)

    dχ/dz = c / (H₀ E(z))
    """
    return 1.0 / E_z(z)


def comoving_distance(z, n_points=512):
    """
    共动距离 χ(z) (Mpc)

    χ(z) = (c/H₀) ∫₀ᶻ dz'/E(z')

    使用复合 Simpson 规则进行高精度数值积分。
    误差阶: O(h⁴) 其中 h = z/n_points

    Parameters
    ----------
    z : float or ndarray
        目标红移
    n_points : int
        积分点数 (需为偶数以保证 Simpson 精度)

    Returns
    -------
    chi : float or ndarray
        共动距离 (Mpc)
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    result = np.zeros_like(z_arr)

    prefactor = COSMO.c_km_s / COSMO.H0  # c/H₀ in Mpc

    for i, zi in enumerate(z_arr):
        if zi <= 0.0:
            result[i] = 0.0
            continue
        zz = np.linspace(0.0, zi, n_points)
        hh = zz[1] - zz[0]
        ff = integrand_comoving(zz)
        # 复合 Simpson 规则
        integral = (hh / 3.0) * (
            ff[0] + ff[-1]
            + 4.0 * np.sum(ff[1:-1:2])
            + 2.0 * np.sum(ff[2:-2:2])
        )
        result[i] = prefactor * integral

    if np.isscalar(z) or (np.ndim(z) == 0):
        return float(result[0])
    return result


def angular_diameter_distance(z1, z2, n_points=512):
    """
    角直径距离 D_A(z₁, z₂) (Mpc)

    在平坦 ΛCDM 中:
        D_A(z₁,z₂) = (1/(1+z₂)) × (c/H₀) × ∫_{z₁}^{z₂} dz'/E(z')

    物理意义: 角直径距离是描述天体物理角大小的有效距离，
    在大红移时会因宇宙膨胀而出现 D_A 随 z 下降的"角直径距离翻转"现象。

    Parameters
    ----------
    z1 : float
        近端红移 (透镜)
    z2 : float
        远端红移 (源)
    n_points : int
        积分精度

    Returns
    -------
    D_A : float
        角直径距离 (Mpc)
    """
    if z2 <= z1:
        return 0.0

    prefactor = COSMO.c_km_s / COSMO.H0
    zz = np.linspace(z1, z2, n_points)
    hh = zz[1] - zz[0]
    ff = integrand_comoving(zz)
    integral = (hh / 3.0) * (
        ff[0] + ff[-1]
        + 4.0 * np.sum(ff[1:-1:2])
        + 2.0 * np.sum(ff[2:-2:2])
    )
    D_A = prefactor * integral / (1.0 + z2)
    return max(D_A, 0.0)


def sigma_crit(z_l, z_s, n_points=512):
    """
    临界表面质量密度 Σ_crit (M_sun / Mpc²)

    Σ_crit = (c²/(4πG)) × D_S / (D_L × D_LS)

    其中:
        D_S = D_A(0, z_s)    到源的距离
        D_L = D_A(0, z_l)    到透镜的距离
        D_LS = D_A(z_l, z_s) 透镜到源的距离

    物理意义: Σ_crit 是产生强透镜所需的最低表面质量密度。
    当 Σ > Σ_crit 时，透镜能够在源平面上产生多重像。

    Parameters
    ----------
    z_l : float
        透镜红移
    z_s : float
        源红移 (z_s > z_l)
    n_points : int
        积分精度

    Returns
    -------
    Sigma_crit : float
        临界表面质量密度 (M_sun / Mpc²)
    """
    if z_s <= z_l or z_l <= 0.0:
        return np.inf

    D_L = angular_diameter_distance(0.0, z_l, n_points)
    D_S = angular_diameter_distance(0.0, z_s, n_points)
    D_LS = angular_diameter_distance(z_l, z_s, n_points)

    if D_L <= 0.0 or D_LS <= 0.0:
        return np.inf

    # c²/(4πG) in SI → convert to M_sun/Mpc²
    # c² = (299792458)² m²/s²
    # G = 6.674e-11 m³/(kg·s²)
    # c²/(4πG) = 299792458²/(4π × 6.674e-11) kg/m
    c2_4piG = COSMO.c_km_s**2 * 1.0e6 / (4.0 * np.pi * COSMO.G_newton)  # kg/m

    # Convert kg/m → M_sun/Mpc²:
    # 1 kg = 1/M_sun M_sun
    # 1/m = Mpc_m / Mpc × (1/m) = (1/Mpc_m) Mpc⁻¹
    # So kg/m × (1/M_sun) × Mpc_m = M_sun/Mpc² per unit length
    # Actually: c²/(4πG) × D_S/(D_L × D_LS) with distances in Mpc
    # = (kg/m) × (Mpc × 3.0857e22) / (Mpc × 3.0857e22 × Mpc × 3.0857e22)
    # = (kg/m) / (3.0857e22 Mpc)
    # Then convert kg → M_sun: divide by 1.989e30

    Sigma_crit_SI = c2_4piG * (D_S * COSMO.Mpc_m) / (D_L * COSMO.Mpc_m * D_LS * COSMO.Mpc_m)
    # SI units: kg/m / m = kg/m² (surface mass density)

    Sigma_crit_Msun_Mpc2 = Sigma_crit_SI * (COSMO.Mpc_m**2) / COSMO.M_sun
    return Sigma_crit_Msun_Mpc2


def source_redshift_distribution(z, z_m=1.0, alpha=2.0, beta=1.5):
    """
    源红移分布 n(z) (Smail, Ellis & Bridle 1995)

    n(z) ∝ z^α × exp(-(z/z₀)^β)

    其中 z₀ = z_m / √(2/β) 使得 n(z) 在 z_m 处达到峰值。

    归一化: ∫₀^∞ n(z) dz = 1

    Parameters
    ----------
    z : ndarray
        红移数组
    z_m : float
        中位红移
    alpha : float
        低红移端斜率
    beta : float
        高红移端衰减指数

    Returns
    -------
    nz : ndarray
        归一化的红移分布
    """
    z = np.asarray(z, dtype=np.float64)
    z0 = z_m / np.sqrt(2.0 / beta)
    nz = z**alpha * np.exp(-((z / z0)**beta))

    # 归一化 (Simpson over [0, 20])
    z_norm = np.linspace(0.0, 20.0, 2048)
    nz_norm = z_norm**alpha * np.exp(-((z_norm / z0)**beta))
    dz = z_norm[1] - z_norm[0]
    norm = (dz / 3.0) * (
        nz_norm[0] + nz_norm[-1]
        + 4.0 * np.sum(nz_norm[1:-1:2])
        + 2.0 * np.sum(nz_norm[2:-2:2])
    )
    if norm > 0.0:
        nz /= norm
    return np.maximum(nz, 0.0)


def lensing_efficiency(z_l, z_s_distribution, n_quad=64):
    """
    透镜效率因子 g(z_l)

    g(z_l) = ∫_{z_l}^∞ n(z_s) (1 - D_L(z_l, z_s)/D_S(z_s)) dz_s

    物理意义: 效率因子描述了在给定透镜红移 z_l 下，
    所有背景源星系被偏折的平均效率。g → 0 当 z_l → 0
    (没有背景源) 或 z_l → ∞ (源和透镜在同一距离)。

    Parameters
    ----------
    z_l : float
        透镜红移
    z_s_distribution : callable
        源红移分布函数 n(z_s)
    n_quad : int
        积分点数

    Returns
    -------
    g : float
        透镜效率因子
    """
    z_max = 5.0
    zz = np.linspace(max(z_l + 1.0e-6, 0.01), z_max, n_quad)
    dz = zz[1] - zz[0]

    D_L = angular_diameter_distance(0.0, z_l)
    nz = z_s_distribution(zz)
    D_S_arr = np.array([angular_diameter_distance(0.0, zs) for zs in zz])
    D_LS_arr = np.array([angular_diameter_distance(z_l, zs) for zs in zz])

    # 几何因子: 1 - D_L/D_S = D_LS(1+z_s)/(D_S(1+z_l)) for flat cosmology
    # Simpler: for flat, D_LS/D_S gives the ratio
    ratio = np.zeros_like(zz)
    for i in range(len(zz)):
        if D_S_arr[i] > 0.0:
            ratio[i] = D_LS_arr[i] / D_S_arr[i]
        else:
            ratio[i] = 0.0

    integrand = nz * np.maximum(1.0 - ratio, 0.0)

    # 复合梯形规则 (避免 Simpson 在边界问题)
    g = np.trapz(integrand, zz)
    return max(g, 0.0)


def cosmographic_distances_array(z_l=0.3, z_s_max=2.0, n_z=50):
    """
    批量计算距离-红移关系 (用于参数变化分析)

    Parameters
    ----------
    z_l : float
        透镜红移
    z_s_max : float
        最大源红移
    n_z : int
        采样点数

    Returns
    -------
    z_array : ndarray
        红移数组
    D_L_array : ndarray
        透镜角直径距离
    D_S_array : ndarray
        源角直径距离
    Sigma_crit_array : ndarray
        临界密度数组
    """
    z_array = np.linspace(0.01, z_s_max, n_z)
    D_L_array = np.array([angular_diameter_distance(0.0, z_l) for _ in z_array])
    D_S_array = np.array([angular_diameter_distance(0.0, zs) for zs in z_array])
    Sigma_crit_array = np.array([sigma_crit(z_l, zs) for zs in z_array])

    # 处理无穷大值
    Sigma_crit_array = np.where(
        np.isfinite(Sigma_crit_array),
        Sigma_crit_array,
        1.0e10
    )

    return z_array, D_L_array, D_S_array, Sigma_crit_array
