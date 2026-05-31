"""
mechanical_stress.py
基于种子项目 122_buckling_spring (buckling spring lambda/mu calculation)
改造为钙钛矿太阳能电池薄膜热应力屈曲分析模块。

钙钛矿薄膜在升降温过程中，由于与衬底（FTO/TiO2）热膨胀系数失配，
会产生热应力。当应力超过临界值时，薄膜可能发生屈曲（buckling），
导致界面脱粘、晶界开裂，进而影响光电转换效率。

核心公式：
  1. 热应力（双轴应力状态）：
       σ_th = E / (1-ν) * (α_film - α_substrate) * ΔT
     其中 E 为杨氏模量，ν 为泊松比，α 为热膨胀系数。
  2. 屈曲临界应力（Euler 类型，薄膜-衬底系统）：
       σ_cr = π² E_film / (12 (1-ν²)) * (d/L)²
     d 为薄膜厚度，L 为特征屈曲波长。
  3. 屈曲后挠度（post-buckling）：
       w(x) = w_max * sin(π x / L)
  4. 屈曲引起的应变能释放：
       ΔU = (σ_th - σ_cr)² / (2 E') * V_debond
     E' = E / (1-ν²)
  5. 借用原项目 lambda 和 mu 的表达式形式：
       λ(L,θ) = (1 - L) cos(θ) + θ sin(θ) / (4 L)
       μ(L,θ) = -θ cos(θ) / (2 L) + 2 (1 - L) sin(θ)
     其中 L 为归一化薄膜长度，θ 为屈曲角。
"""

import numpy as np
from typing import Tuple


def thermal_stress(
    E_film: float,
    nu_film: float,
    alpha_film: float,
    alpha_substrate: float,
    delta_T: float,
) -> float:
    """
    计算双轴热应力 [Pa]。
    σ_th = E / (1-ν) * (α_film - α_substrate) * ΔT
    """
    if E_film <= 0 or nu_film >= 1.0 or nu_film < -1.0:
        raise ValueError("无效的弹性参数")
    return E_film / (1.0 - nu_film) * (alpha_film - alpha_substrate) * delta_T


def critical_buckling_stress(
    E_film: float,
    nu_film: float,
    thickness: float,
    wavelength: float,
) -> float:
    """
    计算薄膜屈曲临界应力 [Pa]。
    σ_cr = π² E / (12 (1-ν²)) * (d/λ)²
    """
    if thickness <= 0 or wavelength <= 0:
        return np.inf
    return (np.pi ** 2 * E_film) / (12.0 * (1.0 - nu_film ** 2)) * (thickness / wavelength) ** 2


def post_buckling_deflection(
    x: np.ndarray,
    L: float,
    sigma_th: float,
    sigma_cr: float,
    E_film: float,
    nu_film: float,
    thickness: float,
) -> np.ndarray:
    """
    计算屈曲后的挠度分布 w(x)。

    基于 von Karman 板理论简化：
      w_max = (2 L / π) * sqrt((σ_th - σ_cr) / (E' * (d/L)²))
    其中 E' = E / (1-ν²)。
    """
    x = np.asarray(x)
    E_prime = E_film / (1.0 - nu_film ** 2)
    if sigma_th <= sigma_cr or E_prime <= 0 or thickness <= 0:
        return np.zeros_like(x)

    delta_sigma = sigma_th - sigma_cr
    # 简化的后屈曲振幅公式
    w_max = (2.0 * L / np.pi) * np.sqrt(delta_sigma / (E_prime * (thickness / L) ** 2))
    w_max = min(w_max, thickness * 5.0)  # 物理限制
    return w_max * np.sin(np.pi * x / L)


def buckling_lambda_mu(
    L_norm: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接借用原项目 122_buckling_spring 的 lambda 和 mu 公式。
    映射为薄膜屈曲参数：
      L_norm = L / L0 （归一化长度）
      theta = 屈曲角 [rad]

    公式：
      λ = (1 - L_norm) * cos(θ) + θ * sin(θ) / (4 * L_norm)
      μ = -θ * cos(θ) / (2 * L_norm) + 2 * (1 - L_norm) * sin(θ)
    """
    L_norm = np.asarray(L_norm)
    theta = np.asarray(theta)
    # 数值鲁棒性：避免除零
    L_safe = np.where(L_norm > 1e-10, L_norm, 1e-10)

    lam = (1.0 - L_norm) * np.cos(theta) + theta * np.sin(theta) / (4.0 * L_safe)
    mu = -theta * np.cos(theta) / (2.0 * L_safe) + 2.0 * (1.0 - L_norm) * np.sin(theta)
    return lam, mu


def strain_energy_release(
    sigma_th: float,
    sigma_cr: float,
    E_film: float,
    nu_film: float,
    debond_area: float,
    thickness: float,
) -> float:
    """
    计算屈曲脱粘释放的应变能 [J]。
    ΔU = (σ_th - σ_cr)² / (2 E') * A_debond * d
    """
    if sigma_th <= sigma_cr or debond_area <= 0 or thickness <= 0:
        return 0.0
    E_prime = E_film / (1.0 - nu_film ** 2)
    delta_sigma = sigma_th - sigma_cr
    return (delta_sigma ** 2) / (2.0 * E_prime) * debond_area * thickness


def bandgap_shift_from_strain(
    strain: float,
    a_deformation_potential: float = 3.0,  # eV
    b_deformation_potential: float = -1.0,  # eV
) -> float:
    """
    根据形变势理论计算应变引起的带隙移动 [eV]。
    双轴应变下的带隙变化：
      ΔE_g = a * (ε_xx + ε_yy) + b * ε_zz
    对于薄膜双轴应力：ε_xx = ε_yy = ε, ε_zz = -2ν/(1-ν) * ε
    """
    if abs(strain) > 0.1:
        # 大应变非线性修正
        strain = np.copysign(0.1, strain)
    # 简化：ΔE_g ≈ a_deform * 2ε （面内双轴应变）
    delta_Eg = a_deformation_potential * 2.0 * strain
    return delta_Eg


def compute_buckling_impact_on_efficiency(
    delta_T: float = 50.0,
    E_film: float = 15.0e9,  # Pa (MAPbI3)
    nu_film: float = 0.25,
    alpha_film: float = 5.0e-5,  # /K
    alpha_substrate: float = 9.0e-6,  # /K (FTO glass)
    thickness: float = 500e-9,  # m
    wavelength: float = 10e-6,  # m
) -> dict:
    """
    计算热屈曲对太阳能电池效率的影响。

    Returns
    -------
    result : dict
        包含应力、临界应力、挠度、应变能、带隙移动和效率损失估计。
    """
    sigma_th = thermal_stress(E_film, nu_film, alpha_film, alpha_substrate, delta_T)
    sigma_cr = critical_buckling_stress(E_film, nu_film, thickness, wavelength)

    # 屈曲状态判断
    buckled = sigma_th > sigma_cr

    # 后屈曲挠度
    x = np.linspace(0, wavelength, 50)
    w = post_buckling_deflection(x, wavelength, sigma_th, sigma_cr, E_film, nu_film, thickness)

    # 应变能释放
    debond_area = wavelength ** 2
    energy_release = strain_energy_release(sigma_th, sigma_cr, E_film, nu_film, debond_area, thickness)

    # 应变估计
    strain = sigma_th * (1.0 - nu_film) / E_film if E_film > 0 else 0.0
    delta_Eg = bandgap_shift_from_strain(strain)

    # 效率损失估计：带隙移动导致的光谱失配
    # 简化模型：效率损失 ∝ |ΔE_g| / E_g0
    E_g0 = 1.57  # eV
    efficiency_loss = abs(delta_Eg) / E_g0 * 0.1  # 约 10% 的带隙变化转化为效率损失

    return {
        "thermal_stress_MPa": float(sigma_th / 1e6),
        "critical_stress_MPa": float(sigma_cr / 1e6),
        "buckled": bool(buckled),
        "max_deflection_nm": float(np.max(w) * 1e9),
        "strain_energy_release_uJ": float(energy_release * 1e6),
        "bandgap_shift_meV": float(delta_Eg * 1000),
        "estimated_efficiency_loss_percent": float(efficiency_loss * 100),
    }


if __name__ == "__main__":
    result = compute_buckling_impact_on_efficiency()
    print("热屈曲分析结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    L_arr = np.linspace(0.25, 1.75, 50)
    theta_val = np.pi / 8
    lam, mu = buckling_lambda_mu(L_arr, theta_val)
    print(f"\n屈曲参数 λ 范围: [{lam.min():.4f}, {lam.max():.4f}]")
