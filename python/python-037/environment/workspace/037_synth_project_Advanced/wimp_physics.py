
import numpy as np
from typing import Callable, Tuple
from utils import (
    spherical_bessel_j1,
    gauss_hermite_quadrature,
    gev_to_kg,
    KM_S_TO_M_S,
    AMU_KG,
    M_PROTON_GEV,
    RHO_LOCAL_GEV_CM3,
    V0_KM_S,
    VE_KM_S,
    VESC_KM_S,
    C_M_S,
)





def helm_form_factor(er_kev: float, a_mass: float) -> float:
    if er_kev <= 0.0:
        return 1.0
    if a_mass <= 0.0:
        raise ValueError("helm_form_factor: 质量数 A 必须为正")


    c_fm = 1.23 * (a_mass ** (1.0 / 3.0)) - 0.60
    a_fm = 0.52
    s_fm = 0.9


    r_n_fm = np.sqrt(c_fm * c_fm + (7.0 / 3.0) * (np.pi ** 2) * (a_fm ** 2) - 5.0 * (s_fm ** 2))
    r_n_m = r_n_fm * 1.0e-15


    m_n_kg = a_mass * AMU_KG



    er_joule = er_kev * 1.602176634e-16
    q = np.sqrt(2.0 * m_n_kg * er_joule) / (6.62607015e-34 / (2.0 * np.pi))


    if q < 1.0e-20:
        return 1.0

    qr = q * r_n_m
    qs = q * s_fm * 1.0e-15

    j1_val = spherical_bessel_j1(qr)
    f_val = (3.0 * j1_val / qr) * np.exp(-0.5 * qs * qs)
    return f_val ** 2






def reduced_mass(m_chi_gev: float, m_n_gev: float) -> float:
    return (m_chi_gev * m_n_gev) / (m_chi_gev + m_n_gev)


def vmin_recoil(er_kev: float, m_chi_gev: float, a_mass: float) -> float:
    m_n_gev = a_mass * M_PROTON_GEV
    mu_gev = reduced_mass(m_chi_gev, m_n_gev)

    if mu_gev <= 0.0:
        raise ValueError("vmin_recoil: 约化质量必须为正")


    er_joule = er_kev * 1.602176634e-16


    mu_kg = gev_to_kg(mu_gev)
    m_n_kg = gev_to_kg(m_n_gev)






    v_min_m_s = np.sqrt(m_n_kg * er_joule / (2.0 * mu_kg ** 2))
    return v_min_m_s / KM_S_TO_M_S






def velocity_distribution_mb(
    v_kms: np.ndarray,
    v0_kms: float = V0_KM_S,
    ve_kms: float = VE_KM_S,
    vesc_kms: float = VESC_KM_S,
) -> np.ndarray:
    v = np.asarray(v_kms, dtype=float)
    v0 = float(v0_kms)
    ve = float(ve_kms)
    vesc = float(vesc_kms)

    if v0 <= 0.0:
        raise ValueError("velocity_distribution_mb: v0 必须为正")


    k = vesc / v0
    z = ve / v0

    from utils import erf_approx

    norm = 0.5 * (erf_approx((z + k)) - erf_approx((z - k))) - (2.0 * z / np.sqrt(np.pi)) * np.exp(-k * k)


    v_max = vesc + ve + 100.0
    dv = 0.1
    v_grid = np.arange(0.0, v_max, dv)
    raw = np.exp(-((v_grid + ve) / v0) ** 2) - np.exp(-(vesc / v0) ** 2)
    raw = np.where(v_grid > (vesc + ve), 0.0, raw)
    raw = np.where(raw < 0.0, 0.0, raw)
    N_esc = np.trapezoid(raw, v_grid) / (np.sqrt(np.pi) * v0)

    result = np.exp(-((v + ve) / v0) ** 2) - np.exp(-(vesc / v0) ** 2)
    result = np.where(v > (vesc + ve), 0.0, result)
    result = np.where(result < 0.0, 0.0, result)
    result = result / (np.sqrt(np.pi) * v0 * N_esc)
    return result


def eta_function(vmin: float, v0_kms: float = V0_KM_S, ve_kms: float = VE_KM_S, vesc_kms: float = VESC_KM_S) -> float:
    if vmin < 0.0:
        vmin = 0.0





    v_max = max(vesc_kms + ve_kms + 200.0, vmin + 2000.0)
    n_points = max(2000, int((v_max - vmin) * 2))
    v_grid = np.linspace(vmin, v_max, n_points)
    f_vals = velocity_distribution_mb(v_grid, v0_kms, ve_kms, vesc_kms)
    integrand = f_vals / np.where(v_grid < 1.0e-3, 1.0e-3, v_grid)

    if vmin < 1.0e-3:
        v_grid_safe = np.linspace(1.0e-3, v_max, n_points)
        f_vals_safe = velocity_distribution_mb(v_grid_safe, v0_kms, ve_kms, vesc_kms)
        integrand_safe = f_vals_safe / v_grid_safe
        return float(np.trapezoid(integrand_safe, v_grid_safe))
    return float(np.trapezoid(integrand, v_grid))






def differential_rate(
    er_kev: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
) -> float:
    if er_kev <= 0.0:
        return 0.0
    if m_chi_gev <= 0.0 or sigma_pb < 0.0 or a_mass <= 0.0:
        raise ValueError("differential_rate: 物理参数必须为正")


    m_n_gev = a_mass * M_PROTON_GEV
    mu_gev = reduced_mass(m_chi_gev, m_n_gev)


    ff2 = helm_form_factor(er_kev, a_mass)


    v_min = vmin_recoil(er_kev, m_chi_gev, a_mass)


    eta_val = eta_function(v_min)


    sigma_cm2 = sigma_pb * 1.0e-36






    rho_gev_cm3 = RHO_LOCAL_GEV_CM3



    mu_kg = gev_to_kg(mu_gev)
    m_chi_kg = gev_to_kg(m_chi_gev)



















    conv_factor = 1.0e-15

    prefactor = (rho_gev_cm3 * sigma_pb * (a_mass ** 2) * ff2) / (2.0 * m_chi_gev * (mu_gev ** 2))
    rate = prefactor * eta_val * conv_factor

    if rate < 0.0 or not np.isfinite(rate):
        rate = 0.0
    return float(rate)


def total_events_in_range(
    e_min_kev: float,
    e_max_kev: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    n_bins: int = 200,
) -> float:
    if e_min_kev >= e_max_kev:
        return 0.0
    energies = np.linspace(e_min_kev, e_max_kev, n_bins)
    rates = np.array([
        differential_rate(e, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
        for e in energies
    ])

    rates = np.where(np.isfinite(rates), rates, 0.0)
    integral = np.trapezoid(rates, energies)
    return float(integral * target_mass_kg * exposure_days)






def annual_modulation_factor(t_day: float, t0_day: float = 152.0) -> float:
    T = 365.25
    S0 = 1.0
    Sm = 0.05
    return S0 + Sm * np.cos(2.0 * np.pi * (t_day - t0_day) / T)


def annual_modulated_rate(
    er_kev: float,
    t_day: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    t0_day: float = 152.0,
) -> float:
    base_rate = differential_rate(er_kev, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
    mod_factor = annual_modulation_factor(t_day, t0_day)
    return base_rate * mod_factor






if __name__ == "__main__":

    ff2 = helm_form_factor(1.0, 73.0)
    assert ff2 <= 1.0 and ff2 > 0.5, f"Helm 形状因子异常: {ff2}"


    vm = vmin_recoil(10.0, 50.0, 73.0)
    assert vm > 0.0, "vmin 必须为正"


    rate = differential_rate(10.0, 50.0, 1.0, 73.0, 1.0, 365.0)
    assert rate >= 0.0 and np.isfinite(rate), f"微分率异常: {rate}"


    nevt = total_events_in_range(0.5, 50.0, 50.0, 1.0, 73.0, 10.0, 365.0)
    assert nevt >= 0.0 and np.isfinite(nevt), f"总事件数异常: {nevt}"

    print("wimp_physics.py: 所有自测通过")
