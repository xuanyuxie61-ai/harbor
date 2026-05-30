
import numpy as np
import math
from utils import newton_solve, ensure_positive, safe_sqrt, clamp


E_CHARGE = 1.602176634e-19
E_MASS   = 9.10938356e-31
EPS_0    = 8.854187817e-12
MU_0     = 4.0 * math.pi * 1e-7
C_LIGHT  = 299792458.0


def plasma_frequency(n_e: float) -> float:
    n_e = max(float(n_e), 1.0)
    omega_p = math.sqrt(n_e * E_CHARGE ** 2 / (E_MASS * EPS_0))
    return omega_p


def drude_permittivity(n_e: float, nu: float, omega: float) -> complex:









    raise NotImplementedError("Hole 1: drude_permittivity not implemented.")


def wave_number(n_e: float, nu: float, omega: float) -> complex:
    eps = drude_permittivity(n_e, nu, omega)
    k0 = omega / C_LIGHT
    k = k0 * np.sqrt(eps)
    return complex(k)


def collision_frequency_from_temperature(T_e: float, p_gas: float,
                                        gas_type: str = "air") -> float:
    T_e = max(float(T_e), 100.0)
    p_gas = max(float(p_gas), 1.0)
    k_B = 1.380649e-23
    T_gas = 300.0

    n_gas = p_gas / (k_B * T_gas)
    sigma_cn = 1.0e-19
    v_th = math.sqrt(8.0 * k_B * T_e / (math.pi * E_MASS))
    nu = n_gas * sigma_cn * v_th
    return nu


def skin_depth(n_e: float, nu: float, omega: float) -> float:
    k = wave_number(n_e, nu, omega)
    ki = float(k.imag)
    if abs(ki) < 1e-12:
        return 1e6
    return 1.0 / ki


def nonlinear_dispersion_relation(
    omega: float,
    n_e: float,
    nu: float,
    k_guess: float = None,
) -> tuple:
    omega = float(omega)
    n_e = float(n_e)
    nu = float(nu)
    eps_p = drude_permittivity(n_e, nu, omega)
    eps_m = 1.0 + 0j

    k0 = omega / C_LIGHT
    if k_guess is None:

        k_guess = k0 * 1.2

    def D(k):
        if k <= k0:
            return 1e6
        kappa_m = np.sqrt(k ** 2 - k0 ** 2 * eps_m)
        kappa_p = np.sqrt(k ** 2 - k0 ** 2 * eps_p)

        if kappa_m.real < 0:
            kappa_m = -kappa_m
        if kappa_p.real < 0:
            kappa_p = -kappa_p
        val = eps_p * kappa_m + eps_m * kappa_p
        return float(val.real)

    def Dp(k):
        if k <= k0:
            return -1e6
        kappa_m = np.sqrt(k ** 2 - k0 ** 2 * eps_m)
        kappa_p = np.sqrt(k ** 2 - k0 ** 2 * eps_p)
        if kappa_m.real < 0:
            kappa_m = -kappa_m
        if kappa_p.real < 0:
            kappa_p = -kappa_p
        dkappa_m = k / kappa_m if abs(kappa_m) > 1e-14 else 0.0
        dkappa_p = k / kappa_p if abs(kappa_p) > 1e-14 else 0.0
        return float((eps_p * dkappa_m + eps_m * dkappa_p).real)


    x = float(k_guess)
    for it in range(1, 201):
        fx = D(x)
        if abs(fx) < 1e-12:
            return x, it, True
        fpx = Dp(x)
        if abs(fpx) < 1e-14:
            break
        step = fx / fpx

        if abs(step) > 0.3 * x:
            step = 0.1 * step
        x_new = x - step

        if x_new <= k0:
            x_new = k0 + 0.05 * k0
        x = x_new

    fx = D(x)
    conv = abs(fx) < 1e-12
    return x, it, conv


def effective_permittivity_profile(
    z: np.ndarray,
    n_e_profile: np.ndarray,
    nu_profile: np.ndarray,
    omega: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    n_e_profile = ensure_positive(np.asarray(n_e_profile, dtype=float))
    nu_profile = ensure_positive(np.asarray(nu_profile, dtype=float))
    omega = float(omega)

    eps = np.empty_like(z, dtype=complex)
    for i in range(z.size):
        eps[i] = drude_permittivity(n_e_profile[i], nu_profile[i], omega)
    return eps


def conductivity(n_e: float, nu: float, omega: float) -> complex:
    omega_p = plasma_frequency(n_e)
    omega_p2 = omega_p ** 2
    denom = nu ** 2 + omega ** 2
    sigma_r = EPS_0 * omega_p2 * nu / denom
    sigma_i = EPS_0 * omega_p2 * omega / denom
    return complex(sigma_r, sigma_i)
