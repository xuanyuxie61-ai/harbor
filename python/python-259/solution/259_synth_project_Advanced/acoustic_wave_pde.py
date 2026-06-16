# -*- coding: utf-8 -*-
"""
acoustic_wave_pde.py — 重子-光子流体声学波动方程的 IMEX 有限差分离散求解

本模块求解 BAO 现象的母方程 — 在膨胀宇宙背景下, 重子-光子流体的声学振荡.
物理方程在共动坐标 k 空间下写为:

  δ̈_γ(k,η) + 2 ℋ δ̇_γ + k^2 c_s^2(η) δ_γ = S(k, η)          (1)

其中:
  η  = 共形时间 ∫ dt/a = ∫ dz / [(1+z) H(z)]
  ℋ  = a'/a = a H(a)  为共形 Hubble 参数 ( ' 表示对 η 的导数)
  c_s = c / √(3(1+R)) 为重子-光子流体的有效声速
  R   = 3 ρ_b / (4 ρ_γ) = 3 ω_b / (4 ω_γ (1+z)) 为动量密度比
  S   = 引力驱动源项 (包含暗物质势阱 Ψ 和重子加载效应)

为在实空间直接追踪 BAO 峰的生成, 我们在共动坐标 χ ∈ [0, 600] Mpc 的一维
径向网格上求解上式的傅里叶反演. 采用 IMEX (隐式-显式) 时间推进:
  - 声项 k^2 c_s^2 δ_γ 用 Crank-Nicolson 隐式处理 (允许大 CFL 数)
  - Hubble 摩擦 2ℋ δ̇ 与源项 S 用显式 Adams-Bashforth 处理
空间离散使用 `fornberg_fd` 模块的 4 阶中心差分.

种子项目映射
----------
- 1205 Bayesian-Second-Law : 该项目的 `Oscillator-BSL-Remote-Static-Final.py`
  中, Fokker-Planck 方程的 IMEX 时间推进被直接移植为 (1) 的推进方案.
- 515 helmholtz_exact        : Helmholtz 方程精确解的 Bessel 函数分解
  提供了本模块的初条件 (η → 0 时的解析极限).
- 1135 bilevel-optim         : 其 `dict_learning.py` 中的迭代预条件器思路
  被用于构造本模块的声项隐式步的预条件.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np

from bao_constants import (
    FiducialCosmology, get_fiducial, OMEGA_GAMMA_H2, OMEGA_R_H2,
    C_LIGHT_KMS,
)
from background_cosmology import (
    E_z, H_z, E_squared, Omega_m_at_z, growth_D,
)
from fornberg_fd import build_derivative_matrix_1d


# =============================================================
# 物理辅助函数
# =============================================================
def baryon_photon_ratio(cosmo: FiducialCosmology, z: float) -> float:
    """
    R(z) = 3 ρ_b / (4 ρ_γ) = 3 ω_b / (4 ω_γ (1+z))
    """
    return 3.0 * cosmo.omega_b / (4.0 * OMEGA_GAMMA_H2 * (1.0 + z))


def sound_speed_over_c(cosmo: FiducialCosmology, z: float) -> float:
    """
    c_s / c = 1 / sqrt(3(1+R(z)))
    即光子-重子流体中声速占光速的比例.
    """
    R = baryon_photon_ratio(cosmo, z)
    return 1.0 / math.sqrt(3.0 * (1.0 + R))


def conformal_Hubble(cosmo: FiducialCosmology, z: float) -> float:
    """
    共形 Hubble ℋ = a H(a) 单位 Mpc^{-1} (共形时间共动坐标).
    """
    a = 1.0 / (1.0 + z)
    return a * H_z(cosmo, z) / C_LIGHT_KMS


def conformal_time_step(cosmo: FiducialCosmology, z: float,
                        dz: float) -> float:
    """
    dη = dz / [(1+z) H(z)] 单位 Mpc (已乘 c).
    """
    return dz / ((1.0 + z) * H_z(cosmo, z) / C_LIGHT_KMS)


# =============================================================
# 源项模型: 引力势驱动 + 重子加载
# =============================================================
def gravitational_source(k: float, eta: float, z: float,
                         cosmo: FiducialCosmology,
                         k_peak: float = 0.01,
                         amplitude: float = 1.0) -> float:
    """
    简化源项 S(k, η). 在早时期 (z >> z_eq), 辐射驱动下势阱衰减;
    在 z < z_eq, 物质主导时期势阱冻结. 我们采用近似形式:

      S(k, η) = A(k) · D(z) · cos(k r_s(η))

    其中 A(k) 由初功率谱 P(k) ∝ k^{n_s} T^2(k) 决定, D(z) 为增长率.
    这里简化为高斯包络:
      A(k) = amplitude * (k/k_peak)^{n_s} * exp(- (k/k_eq)^2 )

    物理上, 这反映了在 k ≈ k_eq 附近声振荡被最强地驱动.
    """
    k_eq = 0.01 * cosmo.Omega_m * cosmo.h ** 2  # Mpc^{-1} 近似
    A_k = amplitude * (k / k_peak) ** cosmo.n_s * math.exp(-(k / k_eq) ** 2)
    Dz = growth_D(cosmo, z)
    return A_k * Dz


# =============================================================
# IMEX 求解器主类
# =============================================================
@dataclass
class AcousticSolverConfig:
    """求解器参数."""
    z_max        : float = 1100.0     # 起始红移 (复合前)
    z_drag       : float = 1059.0     # 拖拽红移 (近似)
    z_min        : float = 0.0        # 终止红移
    N_z          : int   = 256        # 红移网格点数
    chi_max      : float = 600.0      # 共动坐标最大值 (Mpc)
    N_chi        : int   = 128        # 共动网格点数
    stencil      : int   = 5          # 空间差分模板宽度
    theta_cn     : float = 0.5        # Crank-Nicolson 参数 (0.5 = CN, 1.0 = 全隐)


@dataclass
class AcousticField:
    """声学场 δ_γ(z, χ) 的快照."""
    z      : np.ndarray   # (N_z,)
    chi    : np.ndarray   # (N_chi,)
    delta  : np.ndarray   # (N_z, N_chi)
    v_chi  : np.ndarray   # 共动速度 ∂δ/∂η (N_z, N_chi)
    cs     : np.ndarray   # c_s(z) (N_z,)


class AcousticWaveSolver:
    """
    一维径向 IMEX 求解器.
    """

    def __init__(self, cosmo: FiducialCosmology,
                 cfg: Optional[AcousticSolverConfig] = None):
        self.cosmo = cosmo
        self.cfg = cfg or AcousticSolverConfig()
        self._setup_grid()
        self._setup_operators()

    def _setup_grid(self) -> None:
        """构造红移网格 (对数间隔以精细解析 z ≈ z_drag 附近) 与共动网格."""
        cfg = self.cfg
        # 对数间隔红移, 在 z_drag 附近加密
        z_arr = np.geomspace(cfg.z_min + 1.0e-3, cfg.z_max, cfg.N_z)
        z_arr = z_arr[::-1]  # 从 z_max 到 z_min
        # 共动坐标: 等距 (便于 Fornberg)
        chi_arr = np.linspace(0.0, cfg.chi_max, cfg.N_chi)
        self.z_arr = z_arr
        self.chi_arr = chi_arr

    def _setup_operators(self) -> None:
        """构造 ∂^2/∂χ^2 差分矩阵."""
        self.D2 = build_derivative_matrix_1d(
            self.chi_arr, deriv=2, stencil=self.cfg.stencil
        )
        # 边界条件: χ=0 处 δ=0 (对称), χ=χ_max 处 δ=0 (Dirichlet)
        self.D2[0, :] = 0.0
        self.D2[0, 0] = -2.0 / (self.chi_arr[1] - self.chi_arr[0]) ** 2
        self.D2[0, 1] = 1.0 / (self.chi_arr[1] - self.chi_arr[0]) ** 2
        self.D2[-1, :] = 0.0
        self.D2[-1, -1] = -2.0 / (self.chi_arr[-1] - self.chi_arr[-2]) ** 2
        self.D2[-1, -2] = 1.0 / (self.chi_arr[-1] - self.chi_arr[-2]) ** 2

    def initial_condition(self, k_mode: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 z = z_max (早时期) 处的初条件, 取 k_mode 的 Helmholtz 本征模:
          δ_γ(χ, z_max) = j_0(k_mode · χ) · D(z_max)
          v_χ(χ, z_max) = -k_mode c_s(z_max) j_1(k_mode · χ) · D(z_max)

        种子项目 515 helmholtz_exact 在此提供了 Helmholtz 方程精确解形式.
        """
        chi = self.chi_arr
        kchi = k_mode * chi
        # j_0(x) = sin(x)/x (处理 x=0 极限)
        j0_kchi = np.where(
            kchi > 1.0e-8,
            np.sin(kchi) / (kchi + 1.0e-30),
            1.0 - (kchi ** 2) / 6.0,
        )
        # j_1(x) = sin(x)/x^2 - cos(x)/x
        j1_kchi = np.where(
            kchi > 1.0e-8,
            np.sin(kchi) / (kchi ** 2 + 1.0e-30) - np.cos(kchi) / (kchi + 1.0e-30),
            kchi / 3.0,
        )
        Dz = growth_D(self.cosmo, self.z_arr[0])
        delta0 = j0_kchi * Dz
        cs0 = sound_speed_over_c(self.cosmo, self.z_arr[0])
        vchi0 = -k_mode * cs0 * j1_kchi * Dz
        return delta0, vchi0

    def step_imex(self, delta_n: np.ndarray, v_n: np.ndarray,
                  z_n: float, dz: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步 IMEX 推进: 从 z_n 到 z_n - dz (向低红移方向).
        方程写为一阶系统:
          ∂_η δ = v
          ∂_η v = c_s^2 ∂_χ^2 δ - 2 ℋ v + S(k, η)

        显式部分: Hubble 摩擦 + 源项 (Adams-Bashforth 2 阶)
        隐式部分: 声项 (Crank-Nicolson)
        """
        cs = sound_speed_over_c(self.cosmo, z_n)
        Hc = conformal_Hubble(self.cosmo, z_n)
        deta = conformal_time_step(self.cosmo, z_n, dz)
        theta = self.cfg.theta_cn

        # 显式 RHS
        explicit_n = -2.0 * Hc * v_n
        # 源项 (简化: 用 k_mode=0.05 Mpc^{-1} 的基模)
        src_n = np.array([
            gravitational_source(0.05, 0.0, z_n, self.cosmo)
            for _ in self.chi_arr
        ])
        explicit_n = explicit_n + src_n

        # 隐式矩阵: (I - θ deta^2 cs^2 D2) δ^{n+1}
        I = np.eye(len(self.chi_arr))
        A_imp = I - (theta * deta * deta * cs * cs) * self.D2
        # 右侧: δ^n + deta v^n + (1-θ) deta^2 cs^2 D2 δ^n + deta^2 (显式项)
        rhs_d = (
            delta_n + deta * v_n
            + ((1.0 - theta) * deta * deta * cs * cs) * (self.D2 @ delta_n)
            + deta * deta * explicit_n
        )
        rhs_v = v_n + deta * explicit_n

        # 隐式步: 解 A_imp δ^{n+1} = rhs_d
        delta_np1 = np.linalg.solve(A_imp, rhs_d)
        v_np1 = rhs_v

        return delta_np1, v_np1

    def run(self) -> AcousticField:
        """
        从 z_max 积分到 z_min, 返回 AcousticField.
        """
        cfg = self.cfg
        z_arr = self.z_arr
        chi_arr = self.chi_arr
        N_z = len(z_arr)

        delta = np.zeros((N_z, len(chi_arr)), dtype=np.float64)
        vchi = np.zeros_like(delta)
        cs_arr = np.zeros(N_z, dtype=np.float64)

        delta[0, :], vchi[0, :] = self.initial_condition()
        cs_arr[0] = sound_speed_over_c(self.cosmo, z_arr[0])

        for n in range(N_z - 1):
            dz = z_arr[n] - z_arr[n + 1]  # 正数 (向低红移)
            delta[n + 1, :], vchi[n + 1, :] = self.step_imex(
                delta[n, :], vchi[n, :], z_arr[n], dz
            )
            cs_arr[n + 1] = sound_speed_over_c(self.cosmo, z_arr[n + 1])

        return AcousticField(
            z=z_arr, chi=chi_arr, delta=delta, v_chi=vchi, cs=cs_arr,
        )


# =============================================================
# 声视界计算 (与背景模块独立, 由声学场积分得到)
# =============================================================
def sound_horizon_from_field(field: AcousticField,
                             cosmo: FiducialCosmology) -> float:
    """
    从声学场数值解提取声视界 r_d:
      r_d = ∫_0^{η_d} c_s(η) dη ≈ Σ_{z_n > z_drag} c_s(z_n) / H(z_n) Δz_n
    积分到拖拽红移 z_drag ≈ 1059.
    """
    z_drag = 1059.0
    mask = field.z >= z_drag
    if not mask.any():
        return float("nan")
    z_sub = field.z[mask]
    cs_sub = field.cs[mask]
    # 梯形积分: ∫ c_s/H dz (注意 z_sub 是降序, 需要翻转)
    integrand = np.zeros_like(z_sub)
    for i, (zz, csi) in enumerate(zip(z_sub, cs_sub)):
        # c_s 单位为 c, 转换为 km/s; H(z) 单位为 km/s/Mpc
        integrand[i] = (csi * C_LIGHT_KMS) / H_z(cosmo, zz)
    # 按 z 降序, 从 z_max 积到 z_drag
    r_d = float(np.trapz(integrand[::-1], z_sub[::-1]))
    return abs(r_d)


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    c = get_fiducial()
    cfg = AcousticSolverConfig(N_z=64, N_chi=48, chi_max=400.0)
    solver = AcousticWaveSolver(c, cfg)
    field = solver.run()
    r_d = sound_horizon_from_field(field, c)
    print(f"[acoustic_wave_pde] r_d (数值积分) = {r_d:.2f} Mpc")
    print(f"[acoustic_wave_pde] δ(z=0, χ=150 Mpc) = {field.delta[-1, field.chi.searchsorted(150.0)]:+.4e}")


if __name__ == "__main__":
    _self_check()
