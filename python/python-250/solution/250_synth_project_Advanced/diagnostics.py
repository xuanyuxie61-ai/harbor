"""
全局诊断量 (from 304_disk01_positive_rule 求积思想).

超新星爆发的全局诊断包括：
  - 引力结合能 E_grav = -∫ (G M_r / r) dm
  - 总动能 E_kin = (1/2) ∫ v^2 dm
  - 总内能 E_int = ∫ e dm
  - 总辐射能 E_rad = ∫ E_r dV
  - 熵产率 dS/dt
  - 中微子光度 L_ν
  - 激波半径 R_s(t)
  - 增益区质量 M_gain = ∫_{r_g}^{R_s} 4π r^2 ρ dr
  - 临界参数 α_crit = M_dot v_r / P 在激波处
    (Burrows-Goshy 1993): 若 α > α_crit ≈ 0.06, 爆发可能.

积分采用球对称求积 (from 304_disk01_positive_rule 思想):
  ∫ f(r) dm = ∫ f(r) 4π r^2 ρ dr
  采用复合 Simpson 规则或 Gauss-Legendre 在单元内部积分.

能量守恒检验:
  E_total(t) = E_grav(t) + E_kin(t) + E_int(t) + E_rad(t)
  dE/dt 应与中微子损失 + 边界通量一致.
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


def gravitational_binding_energy(r: np.ndarray, rho: np.ndarray,
                                  volume: np.ndarray) -> float:
    """引力结合能 E_grav = -∫ (G M_r / r) 4π r^2 ρ dr.

  M_r = ∫_0^r 4π r'^2 ρ(r') dr' (累计质量).
  """
    # 累计质量 (单元中心, 近似)
    dm = rho * volume
    M_r = np.cumsum(dm)
    r_safe = np.maximum(r, 1.0)
    integrand = -C.G_GRAV * M_r / r_safe * dm
    return float(np.sum(integrand))


def kinetic_energy(rho: np.ndarray, v: np.ndarray, volume: np.ndarray) -> float:
    """总动能 E_kin = (1/2) ∫ ρ v^2 dV."""
    return float(np.sum(0.5 * rho * v ** 2 * volume))


def internal_energy(rho: np.ndarray, T: np.ndarray, volume: np.ndarray,
                    eos) -> float:
    """总内能 E_int = ∫ ρ e dV."""
    e = eos.specific_internal_energy(rho, T)
    return float(np.sum(rho * e * volume))


def radiation_energy(E_rad_density: np.ndarray, volume: np.ndarray) -> float:
    """总辐射能 E_rad = ∫ E_r dV."""
    return float(np.sum(E_rad_density * volume))


def mass_enclosed(rho: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """累计质量剖面 M_r(r)."""
    dm = rho * volume
    return np.cumsum(dm)


def gain_layer_mass(r: np.ndarray, rho: np.ndarray, volume: np.ndarray,
                    r_gain: float, r_shock: float) -> float:
    """增益区质量 M_gain = ∫_{r_g}^{R_s} 4π r^2 ρ dr."""
    mask = (r >= r_gain) & (r <= r_shock)
    return float(np.sum(rho[mask] * volume[mask]))


def neutrino_luminosity_at_radius(L_nu_surface: float, r: np.ndarray,
                                   chi_a: np.ndarray) -> np.ndarray:
    """中微子光度剖面 (考虑吸收).

  L_ν(r) = L_ν(r_surface) exp(-∫_r^surface χ_a dr')
  """
    # 从外向内积分光学深度
    tau = np.zeros_like(r, dtype=np.float64)
    for i in range(len(r) - 2, -1, -1):
        dr = r[i + 1] - r[i]
        tau[i] = tau[i + 1] + chi_a[i] * dr
    L = L_nu_surface * np.exp(-tau)
    return L


def criticality_parameter(Mdot: float, v_r: float, P: float,
                          r: float, M_proto: float) -> float:
    """Burrows-Goshy (1993) 临界参数 α.

  α = |M_dot v_r| r / P
  若 α > α_crit ≈ 0.06, 系统超临界 (可爆发).
  """
    return abs(Mdot * v_r) * r / max(P, 1.0e-30)


def mass_accretion_rate(rho: np.ndarray, v: np.ndarray, r: np.ndarray) -> np.ndarray:
    """吸积率 M_dot(r) = 4π r^2 ρ |v| (g/s)."""
    return 4.0 * math.pi * r ** 2 * rho * np.abs(v)


def compactness_xi(r: np.ndarray, mass_enc: np.ndarray,
                   r_ref: float = 1.0e8) -> float:
    """致密度参数 ξ (O'Connor-Ott 2011).

  ξ_M = M / M_sun / (r_ref / 1000 km) |_{M_enc(r) = M}
  用于预测爆发难易度.
  """
    idx = np.searchsorted(mass_enc, 2.5 * C.M_SUN)
    idx = min(idx, len(r) - 1)
    r_at_25 = r[idx]
    xi = 2.5 / (r_at_25 / 1.0e8)
    return float(xi)


def global_diagnostics(U: np.ndarray, mesh, eos,
                       E_rad_density: np.ndarray | None = None) -> dict:
    """综合诊断输出."""
    rho, v, P, T = _primitive(U, mesh, eos)
    vol = mesh.volumes
    E_grav = gravitational_binding_energy(mesh.r_centers, rho, vol)
    E_kin = kinetic_energy(rho, v, vol)
    E_int = internal_energy(rho, T, vol, eos)
    E_rad = 0.0
    if E_rad_density is not None:
        E_rad = radiation_energy(E_rad_density, vol)
    M_total = float(np.sum(rho * vol))
    r_shock = mesh.shock_radius(rho, P)
    M_env = float(np.sum(rho * vol))
    return {'E_grav': E_grav, 'E_kin': E_kin, 'E_int': E_int,
            'E_rad': E_rad, 'M_total': M_total,
            'R_shock': r_shock, 'E_total': E_grav + E_kin + E_int + E_rad}


def _primitive(U, mesh, eos):
    """内部：守恒变量转原始变量."""
    rho = np.maximum(U[:, 0], C.TINY_RHO)
    v = U[:, 1] / rho
    e_kin = 0.5 * rho * v ** 2
    e_int = (U[:, 2] - e_kin) / rho
    e_int = np.maximum(e_int, C.TINY_E)
    T = eos.temperature_from_energy(rho, e_int)
    _, _, _, P = eos.pressures(rho, T)
    return rho, v, P, T
