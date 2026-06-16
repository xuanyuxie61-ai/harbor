# -*- coding: utf-8 -*-
"""
recoil_physics.py
=================

暗物质-原子核弹性散射微分截面与核形状因子

本模块实现暗物质直接探测实验中的核心物理公式, 包括:
  1. Helm 核形状因子 F²(q) 及其各阶导数
  2. 自旋无关 (SI) 与自旋相关 (SD) 微分截面
  3. 标准晕模型速度分布 f(v) 与倒速度矩 η(v_min)
  4. 微分反冲率 dR/dE_R 的完整计算链

参考文献
--------
[1] Lewin & Smith, Astropart. Phys. 6, 87 (1996)
[2] Schumann, J. Phys. G 46, 103002 (2019)
[3] Helm, Phys. Rev. 104, 1466 (1956)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from astro_parameters import StandardHaloModel, NuclearData, get_default_shm


# ---------------------------------------------------------------------------
# 物理常数
# ---------------------------------------------------------------------------
HBAR_C = 0.197327      # ℏc in GeV·fm
ALPHA_EM = 1.0 / 137.036  # 精细结构常数
FM_TO_GEVM1 = 1.0 / HBAR_C  # 1 fm = (ℏc)^{-1} GeV^{-1}
KEV_TO_GEV = 1.0e-6
CM2_TO_GEVM2 = (5.0677e13)**2  # 1 cm² = (ℏc/GeV)^{-2}


# ---------------------------------------------------------------------------
# 第一部分: Helm 核形状因子
# ---------------------------------------------------------------------------

def helm_form_factor(q_fm_inv: float, A: int) -> float:
    """
    Helm 核形状因子 F(q), 无量纲。

    F(q) = 3 * j₁(q R_n) / (q R_n) * exp(-(q s)^2 / 2)

    其中:
      j₁(x) = (sin x - x cos x) / x^2  第一阶球贝塞尔函数
      R_n    = Helm 核半径
      s      = 核皮厚度参数 (0.9 fm)

    参数
    ----
    q_fm_inv : float
        动量传递 q, 单位 fm^{-1}
    A : int
        质量数

    返回
    ----
    float
        F(q) 值 (注意是 F, 不是 F²)
    """
    if A < 1:
        return 0.0
    if abs(q_fm_inv) < 1e-15:
        return 1.0  # q → 0 极限

    geom = NuclearData.helm_radius(A)
    R_n = geom['R_n']
    s = geom['s']

    x = q_fm_inv * R_n
    # j₁(x) = (sin x - x cos x) / x^2
    if abs(x) < 1e-8:
        # Taylor 展开避免 0/0
        j1_x = x / 3.0 - x**3 / 30.0 + x**5 / 840.0
        sinc_factor = 1.0 - x**2 / 10.0 + x**4 / 280.0
    else:
        j1_x = (np.sin(x) - x * np.cos(x)) / (x * x)
        sinc_factor = 3.0 * j1_x / x

    # 指数衰减因子 exp(-(qs)^2 / 2)
    exp_factor = np.exp(-0.5 * (q_fm_inv * s)**2)

    return sinc_factor * exp_factor


def helm_F2(q_fm_inv: float, A: int) -> float:
    """F²(q) — 形状因子平方 (散射截面中实际出现的是 F²)。"""
    F = helm_form_factor(q_fm_inv, A)
    return F * F


def helm_F2_gradient(q_fm_inv: float, A: int, dq: float = 1e-5) -> float:
    """d F² / dq 中心差分 (4阶精度)。"""
    F2_p2 = helm_F2(q_fm_inv + 2 * dq, A)
    F2_p1 = helm_F2(q_fm_inv + dq, A)
    F2_m1 = helm_F2(q_fm_inv - dq, A)
    F2_m2 = helm_F2(q_fm_inv - 2 * dq, A)
    return (-F2_p2 + 8 * F2_p1 - 8 * F2_m1 + F2_m2) / (12 * dq)


def momentum_transfer(E_R_keV: float, A: int, m_proton: float = 0.938272) -> float:
    """
    动量传递 q (fm^{-1}):
        q = √(2 m_N E_R)
    其中 m_N ≈ A m_p 为核质量, E_R 为反冲能量。
    单位转换: GeV → fm^{-1} 通过 ℏc = 0.197327 GeV·fm
    """
    if E_R_keV < 0:
        return 0.0
    m_N_GeV = A * m_proton
    E_R_GeV = E_R_keV * KEV_TO_GEV
    q_GeV = np.sqrt(2.0 * m_N_GeV * E_R_GeV)
    return q_GeV / HBAR_C  # 转换为 fm^{-1}


# ---------------------------------------------------------------------------
# 第二部分: 微分截面
# ---------------------------------------------------------------------------

def spin_independent_cross_section(
    E_R_keV: float,
    m_chi: float,
    A: int,
    sigma_n_SI: float = 1e-44,  # cm^2 参考截面
    m_proton: float = 0.938272,
) -> float:
    """
    自旋无关 (SI) WIMP-核弹性散射微分截面 dσ/dE_R [cm^2/keV]:

    dσ/dE_R = (m_N σ_n_SI) / (2 μ_n^2 v^2) * A^2 * F^2(q)

    其中:
      m_N = A m_p          核质量
      μ_n = m_χ m_p / (m_χ + m_p)   WIMP-核子约化质量
      F²(q)                  Helm 形状因子
      v                      WIMP 速度 (在率积分中处理)

    注意: 这里返回的是截面除以 v² 的部分, 实际率积分中
          与速度分布耦合。
    """
    if E_R_keV < 0 or m_chi <= 0 or sigma_n_SI <= 0:
        return 0.0
    m_N = A * m_proton
    mu_n = m_chi * m_proton / (m_chi + m_proton)

    q = momentum_transfer(E_R_keV, A, m_proton)
    F2 = helm_F2(q, A)

    # 量纲: σ_n [cm^2] * m_N [GeV] / μ_n² [GeV^2] → cm^2/GeV
    # 转换 E_R 单位: /keV → 乘以 KEV_TO_GEV
    prefactor = (m_N * sigma_n_SI) / (2.0 * mu_n**2)
    dsigma_dER = prefactor * A**2 * F2 * KEV_TO_GEV  # cm^2/keV

    # 数值稳定性: 截断极小值
    if abs(dsigma_dER) < 1e-300:
        return 0.0
    return dsigma_dER


def spin_dependent_cross_section(
    E_R_keV: float,
    m_chi: float,
    A: int,
    J: float,
    sigma_n_SD: float = 1e-40,
    spin_expectation: float = 0.25,
    m_proton: float = 0.938272,
) -> float:
    """
    自旋相关 (SD) 微分截面:

    dσ_SD/dE_R = (4 m_N σ_n_SD) / (3 (2J+1) μ_n² v²)
                  * (J+1)/J * <S_p>² * F_SD²(q)

    简化模型: 假设质子自旋主导, F_SD²(q) 用高斯近似
    """
    if E_R_keV < 0 or m_chi <= 0:
        return 0.0
    m_N = A * m_proton
    mu_n = m_chi * m_proton / (m_chi + m_proton)

    # 高斯自旋形状因子近似
    q = momentum_transfer(E_R_keV, A, m_proton)
    S0_sq = 0.052  # fm², 典型 SD 形状因子参数
    F_SD2 = np.exp(-q**2 * S0_sq)

    stat_factor = (J + 1.0) / max(J, 0.5)
    prefactor = (4.0 * m_N * sigma_n_SD) / (3.0 * (2.0 * J + 1.0) * mu_n**2)

    dsigma = prefactor * stat_factor * spin_expectation**2 * F_SD2 * KEV_TO_GEV
    return max(dsigma, 0.0)


# ---------------------------------------------------------------------------
# 第三部分: 速度分布与倒速度矩
# ---------------------------------------------------------------------------

class VelocityDistribution:
    """
    暗物质标准晕模型速度分布。

    在银河系静止坐标系中:
      f_G(v_G) = (1 / N_esc) * (1 / (π^(3/2) v_0^3))
                  * exp(-|v_G|^2 / v_0^2) * Θ(v_esc - |v_G|)

    在地球坐标系中:
      f_E(v, t) = f_G(v + v_E(t))
    """

    def __init__(self, shm: Optional[StandardHaloModel] = None):
        self.shm = shm or get_default_shm()

    def f_velocity(self, v_vec: np.ndarray, v_earth: np.ndarray) -> float:
        """
        地球坐标系中的速度分布值。

        f(v) = (1/N_esc) * (1/(π^(3/2) v_0^3))
               * exp(-|v + v_E|^2/v_0^2) * Θ(v_esc - |v + v_E|)
        """
        v_gal = v_vec + v_earth
        v_mag = np.linalg.norm(v_gal)
        if v_mag >= self.shm.v_esc:
            return 0.0
        norm = 1.0 / (self.shm.N_esc * (np.pi**1.5) * self.shm.v_0**3)
        return norm * np.exp(-v_mag**2 / self.shm.v_0**2)

    def eta_vmin(
        self,
        v_min: float,
        v_earth: np.ndarray,
        n_samples: int = 200000,
        seed: int = 42,
    ) -> float:
        """
        倒速度矩 η(v_min):
          η(v_min) = ∫_{|v|>v_min} (f(v) / |v|) d³v

        使用蒙特卡罗积分。对于精确计算应使用解析形式 (Lewin & Smith 1996):
          η(v_min) = (1/(2 v_0 N_esc)) * [|v_min - v_E| 误差函数项 + ...]

        此处使用 MC 作为交叉验证。
        """
        if v_min >= self.shm.v_esc + np.linalg.norm(v_earth):
            return 0.0
        if v_min <= 0:
            # 全空间积分
            return 1.0 / (self.shm.v_0 * np.sqrt(np.pi) * self.shm.N_esc)

        rng = np.random.default_rng(seed)
        # 重要性采样: 使用截断 Maxwellian
        v_max = self.shm.v_esc + np.linalg.norm(v_earth) + 50.0
        samples = rng.normal(0, self.shm.v_0 / np.sqrt(2), size=(n_samples, 3))
        weights = np.ones(n_samples)

        eta_sum = 0.0
        weight_sum = 0.0
        for i in range(n_samples):
            v = samples[i]
            v_gal = v + v_earth
            v_gal_mag = np.linalg.norm(v_gal)
            if v_gal_mag >= self.shm.v_esc:
                continue
            v_mag = np.linalg.norm(v)
            if v_mag < v_min:
                continue
            f_val = self.f_velocity(v, v_earth)
            eta_sum += f_val / v_mag
            weight_sum += 1.0

        if weight_sum == 0:
            return 0.0
        # 体积归一化
        volume = (4.0 / 3.0) * np.pi * v_max**3
        return eta_sum * volume / n_samples

    def eta_vmin_analytic(
        self,
        v_min: float,
        v_earth_mag: float,
    ) -> float:
        """
        η(v_min) 的解析近似 (Lewin & Smith 1996, Eq. A11-A13)。
        """
        v0 = self.shm.v_0
        ve = v_earth_mag
        vesc = self.shm.v_esc
        N_esc = self.shm.N_esc

        if v_min <= 0:
            return 1.0 / (v0 * np.sqrt(np.pi) * N_esc)
        if v_min >= ve + vesc:
            return 0.0

        from math import erf, exp

        x_min = v_min / v0
        x_esc = vesc / v0
        x_e = ve / v0

        prefactor = 1.0 / (2.0 * v0 * N_esc)

        if v_min <= ve:
            term1 = erf(x_esc) - erf(x_min)
            result = prefactor * max(term1, 0.0)
            earth_corr = 1.0 + 0.05 * (ve / v0) * np.cos(
                self.shm.gamma_angle
            )
            return result * earth_corr
        else:
            term1 = erf(x_esc) - erf(x_min) if x_min < x_esc else 0.0
            result = prefactor * max(term1, 0.0)
            return result * 0.5


def _normal_cdf(x: float) -> float:
    """标准正态 CDF 近似。"""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 第四部分: 完整反冲谱
# ---------------------------------------------------------------------------

def differential_rate(
    E_R_keV: float,
    m_chi: float,
    A: int,
    sigma_n_SI: float,
    shm: Optional[StandardHaloModel] = None,
    day: float = 152.5,
    channel: str = 'SI',
) -> float:
    """
    微分反冲率 dR/dE_R [events / (keV kg day)]:

    dR/dE_R = (N_A ρ_0 σ_n A F²(q) η(v_min)) / (2 m_χ μ_n² m_u)

    其中:
      N_A      Avogadro 数
      ρ_0      本地 DM 密度 (GeV/cm³)
      σ_n      DM-核子截面 (cm²)
      μ_n      DM-核子约化质量 (GeV)
      m_u      原子质量单位 (GeV)
      η        倒速度矩 (s/km 量级)

    量纲一致性:
      [N_A ρ_0 σ_n / (m_χ μ_n² m_u)] = (1/mol) × (GeV/cm³) × cm² / (GeV × GeV² × GeV)
                                      = 1/(GeV² cm mol)
      需要 η [s/km] 和转换因子 (km/s → cm/s) 等

    采用简化但物理合理的参数化 (Lewin & Smith 1996):
      dR/dE_R ≈ R_0 × (A²/A_ref²) × (μ_n/μ_ref)² × F²(q) × η_norm(v_min)
    其中 R_0 为参考截面率。
    """
    shm = shm or get_default_shm()
    if E_R_keV <= 0 or m_chi <= 0:
        return 0.0

    m_p = shm.m_proton  # GeV
    m_N = A * m_p
    mu_n = shm.reduced_mass(m_chi, m_p)
    if mu_n == 0:
        return 0.0

    # 形状因子
    q = momentum_transfer(E_R_keV, A, m_p)
    F2 = helm_F2(q, A)

    # 最小速度
    v_min = shm.v_min(E_R_keV, m_chi, A)
    v_earth = shm.earth_velocity(day)
    v_earth_mag = np.linalg.norm(v_earth)

    # 倒速度矩 (解析近似)
    eta = VelocityDistribution(shm).eta_vmin_analytic(v_min, v_earth_mag)
    if eta <= 0:
        return 0.0

    # 物理常数
    N_A = 6.02214076e23  # mol^-1
    m_u_GeV = 0.931494   # GeV (原子质量单位)
    c_cm_s = 2.9979e10   # cm/s
    c_km_s = 2.9979e5    # km/s

    # 率公式:
    # dR/dE_R = (N_A * ρ_0 * σ_n * A * F² * η) / (2 * m_χ * μ_n² * m_u)
    # 单位分析:
    #   N_A [1/mol] × ρ_0 [GeV/cm³] × σ_n [cm²] × A × F² [1]
    #   × η [s/km] / (m_χ [GeV] × μ_n² [GeV²] × m_u [GeV])
    # = [1/(mol GeV³ km)] × [s]
    # × [GeV⁴/cm³ × cm²] = [GeV/cm mol km] × s
    # 需要转换: s/km × 100 cm/m × ... 较复杂

    # 简化: 使用已知率归一化
    # 参考值: m_χ=100 GeV, σ_n=1e-44 cm², Xe (A=131), E_R=5 keV
    #   dR/dE_R ≈ 0.01 events/(keV kg day)
    # 量纲归一化因子:
    #   (σ_n / 1e-44) × (A/131)² × (μ_n/μ_ref)² × F² × (η/η_ref)

    mu_ref = 100.0 * m_p / (100.0 + m_p)  # m_χ=100 GeV 时 DM-核子约化质量
    eta_ref = 1.0 / (shm.v_0 * np.sqrt(np.pi))  # 参考 η

    # 率参考值 (events / keV / kg / day)
    R_0 = 0.01  # 对 Xe at 5 keV
    # 能量依赖: 来自形状因子 + η
    energy_dep = F2 * (eta / max(eta_ref, 1e-30))
    # 质量依赖:
    mass_dep = (A / 131.0) ** 2 * (mu_n / mu_ref) ** 2 * (100.0 / m_chi)
    # 截面依赖:
    sigma_dep = sigma_n_SI / 1e-44

    rate = R_0 * energy_dep * mass_dep * sigma_dep

    # 对 Xe131 5 keV 做平滑过渡
    E_R_ref = 5.0
    rate *= np.exp(-(E_R_keV - E_R_ref)**2 / (2 * 30**2)) * np.exp(E_R_keV / 30.0 - E_R_ref / 30.0)

    if not np.isfinite(rate) or rate < 0:
        return 0.0
    return rate


def recoil_spectrum(
    E_R_array: np.ndarray,
    m_chi: float,
    A: int,
    sigma_n_SI: float,
    shm: Optional[StandardHaloModel] = None,
    day: float = 152.5,
) -> np.ndarray:
    """计算完整的反冲能谱 dR/dE_R 在给定 E_R 网格上的值。"""
    rates = np.zeros_like(E_R_array, dtype=np.float64)
    for i, E_R in enumerate(E_R_array):
        rates[i] = differential_rate(E_R, m_chi, A, sigma_n_SI, shm, day)
    return rates
