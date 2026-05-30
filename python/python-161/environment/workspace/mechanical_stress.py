
import numpy as np
from typing import Tuple


def thermal_stress(
    E_film: float,
    nu_film: float,
    alpha_film: float,
    alpha_substrate: float,
    delta_T: float,
) -> float:
    if E_film <= 0 or nu_film >= 1.0 or nu_film < -1.0:
        raise ValueError("无效的弹性参数")
    return E_film / (1.0 - nu_film) * (alpha_film - alpha_substrate) * delta_T


def critical_buckling_stress(
    E_film: float,
    nu_film: float,
    thickness: float,
    wavelength: float,
) -> float:
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
    x = np.asarray(x)
    E_prime = E_film / (1.0 - nu_film ** 2)
    if sigma_th <= sigma_cr or E_prime <= 0 or thickness <= 0:
        return np.zeros_like(x)

    delta_sigma = sigma_th - sigma_cr

    w_max = (2.0 * L / np.pi) * np.sqrt(delta_sigma / (E_prime * (thickness / L) ** 2))
    w_max = min(w_max, thickness * 5.0)
    return w_max * np.sin(np.pi * x / L)


def buckling_lambda_mu(
    L_norm: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    L_norm = np.asarray(L_norm)
    theta = np.asarray(theta)

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
    if sigma_th <= sigma_cr or debond_area <= 0 or thickness <= 0:
        return 0.0
    E_prime = E_film / (1.0 - nu_film ** 2)
    delta_sigma = sigma_th - sigma_cr
    return (delta_sigma ** 2) / (2.0 * E_prime) * debond_area * thickness


def bandgap_shift_from_strain(
    strain: float,
    a_deformation_potential: float = 3.0,
    b_deformation_potential: float = -1.0,
) -> float:
    if abs(strain) > 0.1:

        strain = np.copysign(0.1, strain)

    delta_Eg = a_deformation_potential * 2.0 * strain
    return delta_Eg


def compute_buckling_impact_on_efficiency(
    delta_T: float = 50.0,
    E_film: float = 15.0e9,
    nu_film: float = 0.25,
    alpha_film: float = 5.0e-5,
    alpha_substrate: float = 9.0e-6,
    thickness: float = 500e-9,
    wavelength: float = 10e-6,
) -> dict:
    sigma_th = thermal_stress(E_film, nu_film, alpha_film, alpha_substrate, delta_T)
    sigma_cr = critical_buckling_stress(E_film, nu_film, thickness, wavelength)


    buckled = sigma_th > sigma_cr


    x = np.linspace(0, wavelength, 50)
    w = post_buckling_deflection(x, wavelength, sigma_th, sigma_cr, E_film, nu_film, thickness)


    debond_area = wavelength ** 2
    energy_release = strain_energy_release(sigma_th, sigma_cr, E_film, nu_film, debond_area, thickness)


    strain = sigma_th * (1.0 - nu_film) / E_film if E_film > 0 else 0.0
    delta_Eg = bandgap_shift_from_strain(strain)



    E_g0 = 1.57
    efficiency_loss = abs(delta_Eg) / E_g0 * 0.1

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
