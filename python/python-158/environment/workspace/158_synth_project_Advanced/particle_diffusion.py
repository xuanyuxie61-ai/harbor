
import numpy as np
from utils import newton_raphson_scalar, gauss_legendre_nodes_weights






def spherical_bessel_j0(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sin(x) / x
    near_zero = np.abs(x) < 1e-12
    result = np.where(near_zero, 1.0 - x * x / 6.0, result)
    return result


def spherical_bessel_j0_prime(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.cos(x) / x - np.sin(x) / (x * x)
    near_zero = np.abs(x) < 1e-12
    result = np.where(near_zero, -x / 3.0, result)
    return result


def compute_j0_zeros(n_zeros: int) -> np.ndarray:
    zeros = np.zeros(n_zeros)
    for k in range(1, n_zeros + 1):

        x0 = k * np.pi
        f = lambda x: spherical_bessel_j0(x)
        df = lambda x: spherical_bessel_j0_prime(x)
        x_star, _, conv = newton_raphson_scalar(
            f, df, x0, tol=1e-12, max_iter=50,
            x_min=(k - 0.5) * np.pi + 1e-3,
            x_max=(k + 0.5) * np.pi - 1e-3
        )
        zeros[k - 1] = x_star if conv else k * np.pi
    return zeros






def thiele_modulus(R_p: float, k: float, D_eff: float) -> float:
    if R_p <= 0.0 or k <= 0.0 or D_eff <= 0.0:
        return 0.0
    return R_p * np.sqrt(k / D_eff)


def effectiveness_factor(phi: float) -> float:
    if phi < 1e-12:
        return 1.0
    if phi > 100.0:

        return 3.0 / phi
    tanh_phi = np.tanh(phi)
    if abs(tanh_phi) < 1e-300:
        return 3.0 / phi
    return (3.0 / phi) * (1.0 / tanh_phi - 1.0 / phi)


def effectiveness_factor_cylinder(phi: float) -> float:
    if phi < 1e-12:
        return 1.0
    if phi > 50.0:
        return 2.0 / phi



    i0 = 1.0
    i1 = phi / 2.0
    term0 = 1.0
    term1 = phi / 2.0
    for k in range(1, 30):
        term0 *= (phi / 2.0) ** 2 / (k * k)
        term1 *= (phi / 2.0) ** 2 / (k * (k + 1))
        i0 += term0
        i1 += term1
        if abs(term0) < 1e-30 and abs(term1) < 1e-30:
            break
    if abs(i0) < 1e-300:
        return 2.0 / phi
    return 2.0 * i1 / (phi * i0)






def effective_diffusivity(
    D_bulk: float, porosity: float, tortuosity: float, knudsen: bool = False,
    pore_radius: float = 1e-8, T: float = 1500.0, MW: float = 0.030
) -> float:
    if porosity <= 0.0 or tortuosity <= 0.0:
        return 0.0
    D_k = np.inf
    if knudsen and pore_radius > 0.0 and MW > 0.0:
        R_gas = 8.314462618
        D_k = (2.0 / 3.0) * pore_radius * np.sqrt(8.0 * R_gas * T / (np.pi * MW))
    if D_k < 1e30 and D_bulk > 0.0:
        D_eff = (porosity / tortuosity) / (1.0 / D_bulk + 1.0 / D_k)
    else:
        D_eff = (porosity / tortuosity) * D_bulk
    return max(D_eff, 0.0)






def concentration_profile_spectral(
    r: np.ndarray, R_p: float, C_surf: float, D_eff: float,
    k: float, n_modes: int = 20
) -> np.ndarray:
    if R_p <= 0.0 or D_eff <= 0.0:
        return np.zeros_like(r)
    phi = thiele_modulus(R_p, k, D_eff)
    r = np.clip(r, 0.0, R_p)
    
    if phi < 1e-12:
        return np.full_like(r, C_surf)
    

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = R_p / r
        ratio = np.where(r > 1e-30, ratio, 0.0)
        sinh_term = np.sinh(phi * r / R_p)
        profile = C_surf * ratio * sinh_term / np.sinh(phi)
        profile = np.where(r > 1e-30, profile, C_surf * phi / np.sinh(phi))
    

    if n_modes > 0:
        zeros = compute_j0_zeros(n_modes)
        profile_spectral = np.zeros_like(r)
        for lam in zeros:



            j1_lam = np.sin(lam) / (lam * lam) - np.cos(lam) / lam
            if abs(j1_lam) < 1e-300:
                continue
            ak = C_surf * 2.0 * ((-1) ** (int(round(lam / np.pi)) + 1)) / (lam * j1_lam)
            profile_spectral += ak * spherical_bessel_j0(lam * r / R_p)

        if np.max(np.abs(profile_spectral - profile)) / max(abs(C_surf), 1e-30) < 0.1:
            profile = 0.5 * (profile + profile_spectral)
    
    return np.clip(profile, 0.0, max(C_surf, 0.0) * 10.0)






def fuel_n_release_rate(
    R_p: float, T_p: float, Y_N: float, D_eff: float,
    rho_char: float = 1800.0, A_release: float = 2.0e5,
    E_release: float = 120.0e3
) -> float:
    R_gas = 8.314462618
    if T_p <= 0.0 or R_p <= 0.0 or Y_N <= 0.0 or D_eff <= 0.0:
        return 0.0
    
    arg = -E_release / (R_gas * T_p)
    if arg < -700.0:
        k_release = 0.0
    else:
        k_release = A_release * np.exp(arg)
    
    phi = thiele_modulus(R_p, k_release, D_eff)
    eta = effectiveness_factor(phi)
    
    return eta * k_release * rho_char * Y_N






def char_oxidation_rate(
    T_surf: float, P_O2_surf: float, A_char: float = 1.0e4,
    E_char: float = 135.0e3
) -> float:
    R_gas = 8.314462618
    if T_surf <= 0.0 or P_O2_surf <= 0.0:
        return 0.0
    arg = -E_char / (R_gas * T_surf)
    if arg < -700.0:
        return 0.0
    k_s = A_char * np.exp(arg)
    return k_s * (P_O2_surf ** 0.5)
