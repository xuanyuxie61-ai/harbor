"""
particle_diffusion.py
=====================
Intra-particle diffusion-reaction coupling for char combustion and
NOx precursor release.

Incorporates besselj_zero (080): Bessel function zeros are the eigenvalues
of the diffusion operator in spherical/cylindrical coordinates.

Scientific model:
    Inside a porous char particle, oxygen diffuses inward and reacts
    with char (C + O2 -> CO2) while fuel-N species (HCN, NH3) diffuse
    outward. The governing equation in spherical coordinates is:

        eps * dC_i/dt = D_eff,i * (1/r^2) * d/dr(r^2 * dC_i/dr) + R_i(C,T)

    with boundary conditions:
        dC_i/dr |_{r=0} = 0   (symmetry)
        C_i(R_p) = C_i,bulk   (surface equilibrium)

    For the quasi-steady diffusion limit (Thiele modulus analysis),
    the concentration profile is governed by:

        (1/r^2) * d/dr(r^2 * dC/dr) = (k/D_eff) * C = phi^2/R_p^2 * C

    where phi = R_p * sqrt(k/D_eff) is the Thiele modulus.

    The effectiveness factor for a first-order reaction in a sphere:

        eta = 3/phi * (1/tanh(phi) - 1/phi)

Bessel function connection:
    The eigenfunctions of the Laplacian in a sphere are spherical Bessel
    functions j_n(lambda_{nk} * r / R_p), where lambda_{nk} are zeros of j_n.
    For the radial diffusion problem (l=0), j_0(x) = sin(x)/x,
    and its zeros are at x = k*pi for k=1,2,3,...

    The exact transient solution can be expanded as:
        C(r,t) = sum_{k=1}^infty a_k * exp(-D_eff * lambda_{0,k}^2 * t / R_p^2)
                                   * j_0(lambda_{0,k} * r / R_p)
"""

import numpy as np
from utils import newton_raphson_scalar, gauss_legendre_nodes_weights


# ======================================================================
# 1. Bessel zeros for spherical diffusion eigenvalues
# ======================================================================

def spherical_bessel_j0(x) -> np.ndarray:
    """j_0(x) = sin(x) / x, vectorized."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sin(x) / x
    near_zero = np.abs(x) < 1e-12
    result = np.where(near_zero, 1.0 - x * x / 6.0, result)
    return result


def spherical_bessel_j0_prime(x) -> np.ndarray:
    """j_0'(x) = cos(x)/x - sin(x)/x^2, vectorized."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.cos(x) / x - np.sin(x) / (x * x)
    near_zero = np.abs(x) < 1e-12
    result = np.where(near_zero, -x / 3.0, result)
    return result


def compute_j0_zeros(n_zeros: int) -> np.ndarray:
    """
    Compute first n_zeros positive zeros of j_0(x) = sin(x)/x.
    Zeros are at x = k*pi, but we use Newton iteration for generality
    (adapting the algorithm from besselj_zero).
    """
    zeros = np.zeros(n_zeros)
    for k in range(1, n_zeros + 1):
        # Initial guess
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


# ======================================================================
# 2. Thiele modulus and effectiveness factor
# ======================================================================

def thiele_modulus(R_p: float, k: float, D_eff: float) -> float:
    """
    Thiele modulus for first-order reaction in sphere:
        phi = R_p * sqrt(k / D_eff)
    """
    if R_p <= 0.0 or k <= 0.0 or D_eff <= 0.0:
        return 0.0
    return R_p * np.sqrt(k / D_eff)


def effectiveness_factor(phi: float) -> float:
    """
    Effectiveness factor for first-order reaction in sphere:
        eta = (3/phi) * (1/tanh(phi) - 1/phi)
    As phi -> 0:  eta -> 1 (kinetically controlled)
    As phi -> inf: eta -> 3/phi (diffusion controlled)
    """
    if phi < 1e-12:
        return 1.0
    if phi > 100.0:
        # Asymptotic: eta -> 3/phi
        return 3.0 / phi
    tanh_phi = np.tanh(phi)
    if abs(tanh_phi) < 1e-300:
        return 3.0 / phi
    return (3.0 / phi) * (1.0 / tanh_phi - 1.0 / phi)


def effectiveness_factor_cylinder(phi: float) -> float:
    """
    Effectiveness factor for infinite cylinder:
        eta = 2 * I_1(phi) / (phi * I_0(phi))
    Using asymptotic forms for large phi.
    """
    if phi < 1e-12:
        return 1.0
    if phi > 50.0:
        return 2.0 / phi
    # Series expansion for modified Bessel I_0 and I_1
    # I_0(x) = sum_{k=0}^inf (x/2)^{2k} / (k!)^2
    # I_1(x) = sum_{k=0}^inf (x/2)^{2k+1} / (k! * (k+1)!)
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


# ======================================================================
# 3. Stefan-Maxwell effective diffusivity in porous media
# ======================================================================

def effective_diffusivity(
    D_bulk: float, porosity: float, tortuosity: float, knudsen: bool = False,
    pore_radius: float = 1e-8, T: float = 1500.0, MW: float = 0.030
) -> float:
    """
    Effective diffusivity in porous char particle:
        D_eff = (eps / tau) * (1 / (1/D_bulk + 1/D_Kn))
    where Knudsen diffusivity:
        D_K = (2/3) * r_pore * sqrt(8*R*T / (pi*MW))
    """
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


# ======================================================================
# 4. Intra-particle concentration profile (spectral method)
# ======================================================================

def concentration_profile_spectral(
    r: np.ndarray, R_p: float, C_surf: float, D_eff: float,
    k: float, n_modes: int = 20
) -> np.ndarray:
    """
    Compute quasi-steady concentration profile inside sphere using
    eigenfunction expansion with Bessel zeros.
    
    For first-order reaction with boundary condition C(R_p) = C_surf:
        C(r) = C_surf * (R_p/r) * sinh(phi * r/R_p) / sinh(phi)
    
    This is the exact solution; we also provide the spectral expansion
    for verification.
    """
    if R_p <= 0.0 or D_eff <= 0.0:
        return np.zeros_like(r)
    phi = thiele_modulus(R_p, k, D_eff)
    r = np.clip(r, 0.0, R_p)
    
    if phi < 1e-12:
        return np.full_like(r, C_surf)
    
    # Exact analytical solution
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = R_p / r
        ratio = np.where(r > 1e-30, ratio, 0.0)
        sinh_term = np.sinh(phi * r / R_p)
        profile = C_surf * ratio * sinh_term / np.sinh(phi)
        profile = np.where(r > 1e-30, profile, C_surf * phi / np.sinh(phi))
    
    # Spectral expansion verification (optional, for numerical validation)
    if n_modes > 0:
        zeros = compute_j0_zeros(n_modes)
        profile_spectral = np.zeros_like(r)
        for lam in zeros:
            # Coefficient for boundary condition C(R_p) = C_surf
            # a_k = C_surf * 2*(-1)^{k+1} / (lam * j_1(lam))
            # where j_1(x) = sin(x)/x^2 - cos(x)/x
            j1_lam = np.sin(lam) / (lam * lam) - np.cos(lam) / lam
            if abs(j1_lam) < 1e-300:
                continue
            ak = C_surf * 2.0 * ((-1) ** (int(round(lam / np.pi)) + 1)) / (lam * j1_lam)
            profile_spectral += ak * spherical_bessel_j0(lam * r / R_p)
        # Only use spectral if close to exact
        if np.max(np.abs(profile_spectral - profile)) / max(abs(C_surf), 1e-30) < 0.1:
            profile = 0.5 * (profile + profile_spectral)
    
    return np.clip(profile, 0.0, max(C_surf, 0.0) * 10.0)


# ======================================================================
# 5. Fuel-N release rate from particle
# ======================================================================

def fuel_n_release_rate(
    R_p: float, T_p: float, Y_N: float, D_eff: float,
    rho_char: float = 1800.0, A_release: float = 2.0e5,
    E_release: float = 120.0e3
) -> float:
    """
    Rate of fuel-N (as HCN + NH3) release from char particle [kg_N/(m^3·s)].
    
    First-order Arrhenius release inside particle, coupled with diffusion:
        k_release = A_release * exp(-E_release / (R * T_p))
        R_N = eta(phi_release) * k_release * rho_char * Y_N
    
    Args:
        R_p: particle radius [m]
        T_p: particle temperature [K]
        Y_N: mass fraction of fuel-bound nitrogen in char [-]
        D_eff: effective diffusivity of N-species [m^2/s]
        rho_char: char density [kg/m^3]
        A_release: pre-exponential factor [1/s]
        E_release: activation energy [J/mol]
    
    Returns:
        release_rate: volumetric release rate [kg_N/(m^3·s)]
    """
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


# ======================================================================
# 6. Char oxidation rate (heterogeneous reaction)
# ======================================================================

def char_oxidation_rate(
    T_surf: float, P_O2_surf: float, A_char: float = 1.0e4,
    E_char: float = 135.0e3
) -> float:
    """
    Heterogeneous char oxidation rate [kg_C/(m^2·s)] using nth-order kinetics:
        r_char = A_char * exp(-E_char / (R * T_surf)) * P_O2_surf^n
    with n = 0.5 (global approximation).
    
    Args:
        T_surf: particle surface temperature [K]
        P_O2_surf: O2 partial pressure at surface [Pa]
        A_char: pre-exponential [kg_C/(m^2·s·Pa^0.5)]
        E_char: activation energy [J/mol]
    
    Returns:
        r_char: surface reaction rate [kg_C/(m^2·s)]
    """
    R_gas = 8.314462618
    if T_surf <= 0.0 or P_O2_surf <= 0.0:
        return 0.0
    arg = -E_char / (R_gas * T_surf)
    if arg < -700.0:
        return 0.0
    k_s = A_char * np.exp(arg)
    return k_s * (P_O2_surf ** 0.5)
