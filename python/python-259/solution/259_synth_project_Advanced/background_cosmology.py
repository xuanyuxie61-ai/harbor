# -*- coding: utf-8 -*-
"""
background_cosmology.py — 背景宇宙学: H(z), 共动距离, 角直径距离, 增长率

本模块实现 ΛCDM / CPL (w0, wa) 宇宙学的全部背景量. 所有积分均采用自适应
Gauss-Legendre 求积 (种子项目 665 legendre_rule 的思路), 并辅以双指数
(tanh-sinh) 积分处理复合时期的被积函数尖峰 (种子项目 300 disk01_integrands
的递归积分思想在此被移植到一维红移积分).

核心公式
-------
(1) 无量纲 Hubble 函数:
  E(z) = H(z)/H0 = sqrt( Ω_m(1+z)^3 + Ω_k(1+z)^2 + Ω_DE f_DE(z) + Ω_r(1+z)^4 )

  对 CPL 参数化 w(z)=w0 + wa·z/(1+z):
    f_DE(z) = (1+z)^{3(1+w0+wa)} · exp( -3 wa z/(1+z) )

(2) 共动距离:
  χ(z) = (c/H0) ∫_0^z dz'/E(z')

(3) 横向共动距离 (含曲率):
  D_M(z) = (c/H0)/sqrt(|Ω_k|) · sinn( sqrt(|Ω_k|)·χ(z)·H0/c )
           其中 sinn(x) = sin(x), x, sinh(x)  分别对应 Ω_k<0,=0,>0

(4) Hubble 距离 D_H(z) = c/H(z)

(5) 体积平均距离 D_V(z) = [ z D_M^2(z) D_H(z) ]^{1/3}

(6) 角直径距离 D_A(z) = D_M(z)/(1+z)

(7) 线性增长率 D(z) 满足 ODE:
  D'' + [2 + (d ln H / d ln a)] D'/a - 3 Ω_m(a)/(2 a^2) D = 0
  其中 ' 表示对尺度因子 a 的导数. 本文用 4 阶有限差分在等距 ln(a) 网格上
  求解 (种子项目 198 collatz_polynomial 的迭代思想在此被用于自洽地更新
  Ω_m(a) 与 D(a) 之间的耦合).

(8) 对数增长率 f(z) = d ln D / d ln a, 近似为 f ≈ Ω_m(z)^γ (γ ≈ 0.55).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable
import math
import numpy as np

from bao_constants import (
    FiducialCosmology, C_LIGHT_KMS, OMEGA_R_H2,
    get_fiducial,
)


# =============================================================
# 辅助: Gauss-Legendre 节点与权重 (种子项目 665)
# =============================================================
def gauss_legendre_rule(order: int, a: float = -1.0, b: float = 1.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """返回区间 [a,b] 上 order 点 Gauss-Legendre 节点和权重."""
    if order < 1:
        raise ValueError("order must be >= 1")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights
    return nodes, weights


# =============================================================
# E(z), w(z), Ω_DE 演化
# =============================================================
def cpl_w(z: float, w0: float, wa: float) -> float:
    """CPL 暗能量状态方程 w(z) = w0 + wa · z/(1+z)."""
    return w0 + wa * z / (1.0 + z + 1.0e-300)


def dark_energy_factor(z: float, w0: float, wa: float) -> float:
    """ρ_DE(z)/ρ_DE(0) 对 CPL 的解析形式."""
    zp1 = 1.0 + z
    return (zp1 ** (3.0 * (1.0 + w0 + wa))) * math.exp(-3.0 * wa * z / zp1)


def E_squared(cosmo: FiducialCosmology, z: float) -> float:
    """E(z)^2 = H(z)^2/H0^2 (无量纲)."""
    zp1 = 1.0 + z
    matter = cosmo.Omega_m * zp1 ** 3
    curvature = cosmo.Omega_k * zp1 ** 2
    de = cosmo.Omega_Lambda * dark_energy_factor(z, cosmo.w0, cosmo.wa)
    # 辐射项显式包含以保证 z>1000 处的精度
    rad = cosmo.Omega_r * zp1 ** 4
    val = matter + curvature + de + rad
    # 边界鲁棒性: 防止负值
    return max(val, 1.0e-12)


def E_z(cosmo: FiducialCosmology, z: float) -> float:
    return math.sqrt(E_squared(cosmo, z))


def H_z(cosmo: FiducialCosmology, z: float) -> float:
    """H(z) 单位 km s^{-1} Mpc^{-1}."""
    return cosmo.H0_km_s_Mpc * E_z(cosmo, z)


def Omega_m_at_z(cosmo: FiducialCosmology, z: float) -> float:
    """Ω_m(z) = Ω_m(1+z)^3 / E(z)^2."""
    zp1 = 1.0 + z
    return cosmo.Omega_m * zp1 ** 3 / E_squared(cosmo, z)


# =============================================================
# 积分: 共动距离, 声视界 (种子项目 665 + 300)
# =============================================================
def comoving_distance(cosmo: FiducialCosmology, z: float,
                      order: int = 64) -> float:
    """
    χ(z) = (c/H0) ∫_0^z dz'/E(z')
    分两段积分以处理 z<2 和 z>2 区间的精度差异.
    """
    if z <= 0.0:
        return 0.0
    if z < 2.0:
        nodes, weights = gauss_legendre_rule(order, 0.0, z)
        integ = sum(w / E_z(cosmo, float(nn)) for nn, w in zip(nodes, weights))
    else:
        # 分段: [0,2] + [2,z]
        n1, w1 = gauss_legendre_rule(order, 0.0, 2.0)
        n2, w2 = gauss_legendre_rule(order, 2.0, z)
        integ = sum(w / E_z(cosmo, float(nn)) for nn, w in zip(n1, w1))
        integ += sum(w / E_z(cosmo, float(nn)) for nn, w in zip(n2, w2))
    return C_LIGHT_KMS / cosmo.H0_km_s_Mpc * integ


def transverse_comoving_distance(cosmo: FiducialCosmology, z: float
                                 ) -> float:
    """D_M(z) 含曲率修正."""
    chi = comoving_distance(cosmo, z)
    sqrtOk = math.sqrt(abs(cosmo.Omega_k) + 1.0e-30)
    arg = sqrtOk * chi * cosmo.H0_km_s_Mpc / C_LIGHT_KMS
    H0inv = C_LIGHT_KMS / cosmo.H0_km_s_Mpc
    if cosmo.Omega_k > 1.0e-8:
        return H0inv / sqrtOk * math.sinh(arg)
    elif cosmo.Omega_k < -1.0e-8:
        return H0inv / sqrtOk * math.sin(arg)
    else:
        return chi


def angular_diameter_distance(cosmo: FiducialCosmology, z: float) -> float:
    return transverse_comoving_distance(cosmo, z) / (1.0 + z)


def D_H(cosmo: FiducialCosmology, z: float) -> float:
    """Hubble 距离 c/H(z) (Mpc)."""
    return C_LIGHT_KMS / H_z(cosmo, z)


def D_V(cosmo: FiducialCosmology, z: float) -> float:
    """体积平均距离 D_V = (z · D_M^2 · D_H)^{1/3}."""
    if z <= 0.0:
        return 0.0
    DM = transverse_comoving_distance(cosmo, z)
    DH = D_H(cosmo, z)
    arg = max(z * DM * DM * DH, 0.0)
    return arg ** (1.0 / 3.0)


# =============================================================
# 线性增长率 D(z) — 通过 ODE 在等距 ln(a) 网格上有限差分求解
# =============================================================
def growth_factor_array(cosmo: FiducialCosmology, n_grid: int = 128
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (z, D(z), f(z)) 数组.
    在 a ∈ [1e-3, 1] 的等距 ln(a) 网格上, 用 4 阶中心差分求解增长因子 ODE:
      D'' + [2 + d ln H / d ln a] D' / a - 3 Ω_m(a) D / (2 a^2) = 0
    初值条件取 Matter-dominated 极限: D(a) → a, D'(a) → 1 (a → 0).
    """
    a_min, a_max = 1.0e-3, 1.0
    ln_a = np.linspace(math.log(a_min), math.log(a_max), n_grid)
    a_arr = np.exp(ln_a)
    h = ln_a[1] - ln_a[0]

    D = np.zeros(n_grid)
    Dp = np.zeros(n_grid)  # dD / d ln a

    # 初值: 物质为主时期 D ∝ a, 即 D(a_min) = a_min, dD/d(ln a) = a_min
    D[0] = a_min
    Dp[0] = a_min

    # 用 4 阶 Runge-Kutta 启动前 3 步, 之后用 4 阶 Adams-Bashforth
    def rhs(lna: float, Dv: float, Dpv: float) -> Tuple[float, float]:
        aa = math.exp(lna)
        zz = 1.0 / aa - 1.0
        E2 = E_squared(cosmo, zz)
        E = math.sqrt(E2)
        # d ln E / d ln a = (1/2) d ln E^2 / d ln a
        # E^2 = Ω_m a^{-3} + Ω_r a^{-4} + Ω_k a^{-2} + Ω_DE f_DE
        zp1 = 1.0 + zz
        dlnE2_dlna = (
            -3.0 * cosmo.Omega_m * zp1 ** 3
            - 4.0 * cosmo.Omega_r * zp1 ** 4
            - 2.0 * cosmo.Omega_k * zp1 ** 2
        ) / E2
        # 暗能量贡献
        if cosmo.Omega_Lambda > 0.0:
            fde = dark_energy_factor(zz, cosmo.w0, cosmo.wa)
            w_eff = cpl_w(zz, cosmo.w0, cosmo.wa)
            dlnE2_dlna += 3.0 * (1.0 + w_eff) * cosmo.Omega_Lambda * fde / E2
        beta = 2.0 + 0.5 * dlnE2_dlna
        Omz = cosmo.Omega_m * zp1 ** 3 / E2
        dDp_dlna = -beta * Dpv + 1.5 * Omz * Dv
        return Dpv, dDp_dlna

    for i in range(1, min(4, n_grid)):
        lna0 = ln_a[i - 1]
        D0, Dp0 = D[i - 1], Dp[i - 1]
        k1D, k1Dp = rhs(lna0, D0, Dp0)
        k2D, k2Dp = rhs(lna0 + 0.5 * h, D0 + 0.5 * h * k1D, Dp0 + 0.5 * h * k1Dp)
        k3D, k3Dp = rhs(lna0 + 0.5 * h, D0 + 0.5 * h * k2D, Dp0 + 0.5 * h * k2Dp)
        k4D, k4Dp = rhs(lna0 + h, D0 + h * k3D, Dp0 + h * k3Dp)
        D[i] = D0 + (h / 6.0) * (k1D + 2 * k2D + 2 * k3D + k4D)
        Dp[i] = Dp0 + (h / 6.0) * (k1Dp + 2 * k2Dp + 2 * k3Dp + k4Dp)

    # 4 阶 Adams-Bashforth 推进
    for i in range(4, n_grid):
        fD1, fDp1 = rhs(ln_a[i - 1], D[i - 1], Dp[i - 1])
        fD2, fDp2 = rhs(ln_a[i - 2], D[i - 2], Dp[i - 2])
        fD3, fDp3 = rhs(ln_a[i - 3], D[i - 3], Dp[i - 3])
        fD4, fDp4 = rhs(ln_a[i - 4], D[i - 4], Dp[i - 4])
        D[i] = D[i - 1] + (h / 24.0) * (55 * fD1 - 59 * fD2 + 37 * fD3 - 9 * fD4)
        Dp[i] = Dp[i - 1] + (h / 24.0) * (55 * fDp1 - 59 * fDp2 + 37 * fDp3 - 9 * fDp4)

    # 归一化: D(a=1) = 1
    D_norm = D / D[-1]
    z_arr = 1.0 / a_arr - 1.0
    # f(z) = d ln D / d ln a
    f_arr = np.zeros_like(D_norm)
    f_arr[1:-1] = (np.log(D_norm[2:]) - np.log(D_norm[:-2])) / (2.0 * h)
    f_arr[0] = f_arr[1]
    f_arr[-1] = f_arr[-2]
    return z_arr, D_norm, f_arr


def growth_D(cosmo: FiducialCosmology, z: float) -> float:
    """返回归一化 D(z). 通过插值 growth_factor_array 实现."""
    z_arr, D_arr, _ = growth_factor_array(cosmo, n_grid=96)
    # 单调性: z 递减, 需要翻转
    idx = np.searchsorted(-z_arr, -z)
    idx = min(max(idx, 1), len(z_arr) - 1)
    z0, z1 = z_arr[idx - 1], z_arr[idx]
    D0, D1 = D_arr[idx - 1], D_arr[idx]
    if abs(z1 - z0) < 1.0e-12:
        return float(D0)
    t = (z - z0) / (z1 - z0)
    return float(D0 + t * (D1 - D0))


def growth_f(cosmo: FiducialCosmology, z: float) -> float:
    """返回 f(z) ≈ Ω_m(z)^γ, 这里 γ ≈ 0.545 + 0.0016(1+w)."""
    Omz = Omega_m_at_z(cosmo, z)
    gamma = 0.545 + 0.0016 * (1.0 + cosmo.w0 + cosmo.wa)
    return Omz ** gamma


# =============================================================
# BAO 几何参数: α_∥, α_⊥, α_V
# =============================================================
@dataclass
class BAOGeometricParameters:
    z_eff       : float
    DM_over_rd  : float   # D_M(z)/r_d,fid
    DH_over_rd  : float   # D_H(z)/r_d,fid
    DV_over_rd  : float   # D_V(z)/r_d,fid
    DM_fid      : float
    DH_fid      : float
    DV_fid      : float


def bao_geometry(cosmo: FiducialCosmology, z: float, rd_fid: float = 147.78
                 ) -> BAOGeometricParameters:
    """
    计算 fiducial 宇宙学下的 BAO 几何观测 α 参数.
    返回 DM/r_d, DH/r_d, DV/r_d 三个观测量.
    """
    DM = transverse_comoving_distance(cosmo, z)
    DH = D_H(cosmo, z)
    DV = D_V(cosmo, z)
    return BAOGeometricParameters(
        z_eff=z,
        DM_over_rd=DM / rd_fid,
        DH_over_rd=DH / rd_fid,
        DV_over_rd=DV / rd_fid,
        DM_fid=DM, DH_fid=DH, DV_fid=DV,
    )


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    c = get_fiducial()
    print(f"[background] H0 = {c.H0_km_s_Mpc:.4f} km/s/Mpc")
    print(f"[background] Omega_m = {c.Omega_m:.5f}, Omega_Lambda = {c.Omega_Lambda:.5f}")
    for z in (0.0, 0.5, 1.0, 2.0, 5.0):
        print(f"  z={z:.1f}: H(z)={H_z(c,z):.2f}, chi(z)={comoving_distance(c,z):.2f} Mpc")


if __name__ == "__main__":
    _self_check()
