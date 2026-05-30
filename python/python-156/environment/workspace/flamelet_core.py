
import numpy as np
from scipy.special import erfinv


R_UNIVERSAL = 8.314462618
PRESSURE_ATM = 101325.0


MOL_WEIGHT_FUEL = 16.04e-3
MOL_WEIGHT_OXIDIZER = 28.97e-3
STOICHIOMETRIC_RATIO = 17.16


PRE_EXPONENTIAL = 2.0e6
ACTIVATION_ENERGY = 8.0e4
HEAT_RELEASE = 5.0e7


T_OXIDIZER = 300.0
T_FUEL = 300.0
ADIA_TEMP_STOIC = 2226.0


Y_FUEL_INLET = 1.0
Y_OXIDIZER_INLET = 0.232


Z_STOICHIOMETRIC = 1.0 / (1.0 + STOICHIOMETRIC_RATIO)


def scalar_dissipation_rate(Z, chi_st, Z_st=None):
    if Z_st is None:
        Z_st = Z_STOICHIOMETRIC

    if chi_st <= 0.0:
        raise ValueError("标量耗散率 chi_st 必须为正数。")

    Z = np.clip(Z, 0.0, 1.0)
    sigma_chi = 0.15

    exponent = -((Z - Z_st) ** 2) / (2.0 * sigma_chi ** 2)
    exponent = np.clip(exponent, -700.0, 0.0)

    chi = chi_st * np.exp(exponent)
    return chi


def mixture_molecular_weight(Z):
    Z = np.clip(Z, 0.0, 1.0)
    denom = Z / MOL_WEIGHT_FUEL + (1.0 - Z) / MOL_WEIGHT_OXIDIZER

    denom = np.where(np.abs(denom) < 1.0e-30, 1.0e-30, denom)
    return 1.0 / denom


def density_mixture(Z, T):
    T = np.maximum(T, 1.0)
    W = mixture_molecular_weight(Z)
    rho = PRESSURE_ATM * W / (R_UNIVERSAL * T)
    return rho


def reaction_rate_one_step(T, Y_F, Y_O, Z=None):
    T = np.maximum(T, 100.0)
    Y_F = np.clip(Y_F, 0.0, 1.0)
    Y_O = np.clip(Y_O, 0.0, 1.0)









    raise NotImplementedError("Hole 1: 请实现 reaction_rate_one_step 函数")


def temperature_equation_rhs(Z, T, Y_F, Y_O, chi_st):
    chi = scalar_dissipation_rate(Z, chi_st)
    rho = density_mixture(Z, T)

    omega_f = reaction_rate_one_step(T, Y_F, Y_O, Z)

    omega_T = HEAT_RELEASE * omega_f / (rho * 1200.0)

    kappa = rho * chi / 2.0

    kappa = np.maximum(kappa, 1.0e-12)

    return kappa, omega_T


def flamelet_boundary_conditions():
    return {
        'T_left': T_OXIDIZER,
        'T_right': T_FUEL,
        'Y_F_left': 0.0,
        'Y_F_right': Y_FUEL_INLET,
        'Y_O_left': Y_OXIDIZER_INLET,
        'Y_O_right': 0.0,
        'Z_st': Z_STOICHIOMETRIC,
        'T_ad': ADIA_TEMP_STOIC,
    }


def thermal_diffusivity_ref():
    lambda_gas = 0.026
    cp = 1200.0
    rho_ref = PRESSURE_ATM * MOL_WEIGHT_OXIDIZER / (R_UNIVERSAL * T_OXIDIZER)
    return lambda_gas / (rho_ref * cp)
