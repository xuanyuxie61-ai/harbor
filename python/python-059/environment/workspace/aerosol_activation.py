
import numpy as np
from math import sqrt, pi, exp, log, erf


class ActivationError(Exception):
    pass


def logistic_exact(t, r, K, t0, f0):
    dt = r * (t - t0)

    if dt > 700:
        return float(K)
    exp_dt = exp(dt)
    numerator = K * f0 * exp_dt
    denominator = K + f0 * (exp_dt - 1.0)
    if abs(denominator) < 1e-15:
        return float(K)
    return numerator / denominator


def kohler_critical_supersaturation(
    temperature,
    surface_tension,
    molecular_weight_water,
    density_water,
    molecular_weight_solute,
    density_solute,
    vanthoff_factor,
    dry_radius,
    mass_solute,
):
    R = 8.314
    A = 2.0 * surface_tension * molecular_weight_water / (R * temperature * density_water)


    volume_solute = mass_solute / density_solute
    B = vanthoff_factor * molecular_weight_water * volume_solute / molecular_weight_solute

    if B <= 0:
        raise ActivationError("kohler_critical_supersaturation: B 参数必须为正")

    s_crit = sqrt(4.0 * A ** 3 / (27.0 * B))

    s_crit = min(s_crit, 0.5)
    return float(s_crit)


def activated_fraction_logistic(
    time,
    supersaturation,
    s_crit,
    sigma_g,
    r_growth=0.01,
    f0=0.001,
):
    if s_crit <= 0 or sigma_g <= 1.0:
        raise ActivationError("activated_fraction_logistic: 参数非法")


    ratio = s_crit / (supersaturation + 1e-12)
    log_ratio = log(ratio)
    log_sigma = log(sigma_g)
    if log_sigma < 1e-12:
        log_sigma = 1e-12

    K = 0.5 * (1.0 - erf(log_ratio / (sqrt(2.0) * log_sigma)))
    K = np.clip(K, 0.0, 1.0)

    t = np.asarray(time, dtype=np.float64)
    f_act = np.array([logistic_exact(ti, r_growth, K, 0.0, f0) for ti in t])
    return np.clip(f_act, 0.0, 1.0)


def ccn_spectrum_derivative(supersaturation, N_total, s_crit, sigma_g):
    if supersaturation <= 0 or s_crit <= 0:
        return 0.0
    ln_s = log(supersaturation)
    ln_sc = log(s_crit)
    ln_sigma = log(sigma_g)
    coeff = N_total / (sqrt(2.0 * pi) * ln_sigma)
    exponent = -0.5 * ((ln_s - ln_sc) / ln_sigma) ** 2
    return coeff * exp(exponent)


def compute_ccn_number_concentration(
    supersaturation_percent,
    N_total,
    r_median,
    sigma_g,
    temperature=298.0,
    surface_tension=0.072,
    molecular_weight_water=0.018,
    density_water=1000.0,
    molecular_weight_solute=0.132,
    density_solute=1760.0,
    vanthoff_factor=3.0,
):
    dry_radius = r_median * 1e-6
    mass_solute = (4.0 / 3.0) * pi * dry_radius ** 3 * density_solute

    s_crit = kohler_critical_supersaturation(
        temperature,
        surface_tension,
        molecular_weight_water,
        density_water,
        molecular_weight_solute,
        density_solute,
        vanthoff_factor,
        dry_radius,
        mass_solute,
    )

    s = supersaturation_percent / 100.0

    ratio = s_crit / (s + 1e-12)
    log_ratio = log(ratio)
    log_sigma = log(sigma_g)
    if log_sigma < 1e-12:
        log_sigma = 1e-12
    K = 0.5 * (1.0 - erf(log_ratio / (sqrt(2.0) * log_sigma)))
    K = np.clip(K, 0.0, 1.0)

    return N_total * K
