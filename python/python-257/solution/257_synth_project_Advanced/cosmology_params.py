"""
cosmology_params.py
===================
Standard Lambda-CDM cosmological parameters and physical constants
used throughout the CMB power-spectrum estimation pipeline.

The six base parameters follow the Planck 2018 conventions:
    theta_s    : angular size of the sound horizon at recombination
    omega_b    : physical baryon density           Omega_b h^2
    omega_c    : physical cold-dark-matter density  Omega_c h^2
    tau        : Thomson optical depth to reionisation
    A_s        : primordial scalar amplitude at pivot k_p
    n_s        : primordial scalar spectral index

Secondary / derived parameters (N_eff, Y_He, h, ...) are derived
from the base set through standard fitting formulae.

Units: lengths in Mpc, energies in eV, temperature in K.
"""

from __future__ import annotations
import math
from typing import Dict

# ---------------------------------------------------------------------------
# Physical constants  (CODATA 2018 / PDG 2022)
# ---------------------------------------------------------------------------
C_LIGHT        = 2.99792458e8           # speed of light              [m/s]
K_BOLTZMANN    = 1.380649e-23           # Boltzmann constant          [J/K]
H_PLANCK       = 6.62607015e-34         # Planck constant             [J s]
HBAR           = H_PLANCK / (2.0 * math.pi)
G_NEWTON       = 6.67430e-11            # Newton constant             [m^3 kg^-1 s^-2]
M_PROTON       = 1.67262192e-27         # proton mass                 [kg]
M_ELECTRON     = 9.10938370e-31         # electron mass               [kg]
SIGMA_THOMSON  = 6.6524587158e-29       # Thomson cross-section       [m^2]
EV_TO_JOULE    = 1.602176634e-19        # 1 eV in joules              [J/eV]
T_CMB0         = 2.72548                # CMB monopole today          [K]
ALPHA_FS       = 7.2973525693e-3        # fine-structure constant
MPC_TO_M       = 3.085677581e22         # 1 Mpc in metres             [m]
G_EV           = 6.70883e-39            # G in (eV/c^2)^-1 (eV)^-2 (hbar c)^3


# ---------------------------------------------------------------------------
# Default Planck-2018 best-fit parameters  (TT,TE,EE+lowE+lensing)
# ---------------------------------------------------------------------------
PLANCK_2018: Dict[str, float] = {
    "H0"      : 67.36,        # Hubble constant today        [km/s/Mpc]
    "ombh2"   : 0.02237,      # physical baryon density      Omega_b h^2
    "omch2"   : 0.1200,       # physical CDM density         Omega_c h^2
    "tau"     : 0.0544,       # reionisation optical depth
    "ln10As"  : 3.044,        # log(1e10 A_s)
    "ns"      : 0.9649,       # scalar spectral index
    "NEFF"    : 3.046,        # effective number of neutrinos
    "m_nu"    : 0.06,         # sum of neutrino masses       [eV]
    "Y_he"    : 0.2454,       # primordial helium fraction
    "T_cmb"   : T_CMB0,       # CMB monopole temperature     [K]
}


def h_from_H0(H0: float) -> float:
    """Reduced Hubble parameter  h = H0 / (100 km/s/Mpc)."""
    return H0 / 100.0


def omega_b(ombh2: float, H0: float) -> float:
    """Baryon density fraction  Omega_b = omega_b h^2 / h^2."""
    return ombh2 / h_from_H0(H0) ** 2


def omega_c(omch2: float, H0: float) -> float:
    """Cold-dark-matter density fraction."""
    return omch2 / h_from_H0(H0) ** 2


def omega_gamma(T_cmb: float, H0: float) -> float:
    """Photon density fraction   Omega_gamma = 4 sigma_SB T^4 / (3 rho_c c^2)."""
    sigma_sb = (math.pi ** 2 / 60.0) * (K_BOLTZMANN ** 4) / (HBAR ** 3 * C_LIGHT ** 2)
    rho_crit = 3.0 * (H0 * 1.0e3 / MPC_TO_M) ** 2 / (8.0 * math.pi * G_NEWTON)
    return 4.0 * sigma_sb * T_cmb ** 4 / (3.0 * rho_crit * C_LIGHT ** 2)


def omega_nu(m_nu: float, T_cmb: float, NEFF: float, H0: float) -> float:
    """Massless + massive neutrino density fraction."""
    fg = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
    return fg * NEFF * omega_gamma(T_cmb, H0) * (1.0 + m_nu / (93.14 * T_cmb ** 2 * NEFF ** 0.75))


def omega_lambda(ombh2: float, omch2: float, T_cmb: float,
                 NEFF: float, m_nu: float, H0: float) -> float:
    """Dark-energy density from closure  Omega_Lambda = 1 - Omega_b - Omega_c - ..."""
    return 1.0 - omega_b(ombh2, H0) - omega_c(omch2, H0) \
           - omega_gamma(T_cmb, H0) - omega_nu(m_nu, T_cmb, NEFF, H0)


def primordial_ps(k: float, A_s: float, n_s: float, k_pivot: float = 0.05) -> float:
    """
    Primordial scalar power spectrum  (single-tilt power law):
        P_R(k) = A_s * (k / k_pivot)^(n_s - 1)
    k_pivot = 0.05 Mpc^-1  (Planck default).
    """
    if k <= 0.0 or A_s <= 0.0:
        raise ValueError("k and A_s must be strictly positive.")
    return A_s * (k / k_pivot) ** (n_s - 1.0)


def sound_horizon_rs(ombh2: float, omch2: float, omega_g: float) -> float:
    """
    Fitting formula for the sound horizon at drag epoch  r_d  (Eisenstein & Hu 1998).
        r_d ~ 44.5 ln(9.83 / Omega_m h^2) / sqrt(1 + 10 Omega_b^(3/4))   Mpc
    Returns r_d in Mpc.
    """
    omh2 = ombh2 + omch2
    ob = ombh2 / h_from_H0(67.36) ** 2  # approximate
    if omh2 <= 0.0:
        raise ValueError("Matter density must be positive.")
    ln_term = math.log(9.83 / omh2)
    b1_term = 1.0 + 10.0 * ob ** 0.75
    rs = 44.5 * ln_term / math.sqrt(b1_term)
    return rs


def angular_diameter_da_to_rec(ombh2: float, omch2: float,
                                T_cmb: float, NEFF: float,
                                m_nu: float, H0: float) -> float:
    """
    Comoving angular-diameter distance to recombination,
    approximated by a one-parameter matter-only integral:
        D_A(z*) = c / H0 * int_0^{z*} dz / E(z),     E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda)
    We use z* = 1089.92 (Planck 2018).
    """
    z_star = 1089.92
    n_quad = 256
    dz = z_star / n_quad
    om = omega_b(ombh2, H0) + omega_c(omch2, H0) + omega_nu(m_nu, T_cmb, NEFF, H0)
    oL = 1.0 - om - omega_gamma(T_cmb, H0)
    integral = 0.0
    for i in range(n_quad):
        z_mid = (i + 0.5) * dz
        E_z = math.sqrt(max(1.0e-30, om * (1.0 + z_mid) ** 3 + oL))
        integral += 1.0 / E_z
    integral *= dz
    return (C_LIGHT / (H0 * 1.0e3 / MPC_TO_M)) * integral


def theta_star(ombh2: float, omch2: float, T_cmb: float,
               NEFF: float, m_nu: float, H0: float) -> float:
    """Angular size of the sound horizon  theta_* = r_s / D_A(z*)."""
    rs = sound_horizon_rs(ombh2, omch2, omega_gamma(T_cmb, H0))
    da = angular_diameter_da_to_rec(ombh2, omch2, T_cmb, NEFF, m_nu, H0)
    if da <= 0.0:
        raise ValueError("Angular diameter distance must be positive.")
    return rs / da


def transfer_function_EH(k: float, ombh2: float, omch2: float,
                         T_cmb: float, NEFF: float, m_nu: float,
                         H0: float) -> float:
    """
    Eisenstein & Hu (1998) zero-baryon transfer function approximation:
        T(k) = L(q) * ln(1 + 2.34 q) / (2.34 q)
        q    = k / (Omega_m h^2 * Gamma)   with Gamma ~ Omega_m h
    """
    om = omega_b(ombh2, H0) + omega_c(omch2, H0) + omega_nu(m_nu, T_cmb, NEFF, H0)
    h  = h_from_H0(H0)
    Gamma = om * h * math.exp(-ombh2 * (1.0 + math.sqrt(2.0 * h) / om))
    if k <= 0.0 or Gamma <= 0.0:
        return 1.0
    q = k / (om * h ** 2 * 1.0) * (T_cmb / 2.7) ** 2
    arg = 2.34 * q
    if arg < 1.0e-12:
        return 1.0
    return math.log(1.0 + arg) / arg


def cmb_cl_theory(ell: int, params: Dict[str, float]) -> float:
    """
    Approximate theoretical C_l  for the temperature auto-spectrum:
        C_l = (2 pi / l(l+1)) * P_R(k_ell) * T(k_ell)^2 * (2 pi / l^2)
    with k_ell = (l + 1/2) / D_A.   (Limber approximation for low l.)
    Used as a quick sanity reference for the numerical pipeline.
    """
    H0    = params["H0"]
    ombh2 = params["ombh2"]
    omch2 = params["omch2"]
    ns    = params["ns"]
    ln10As = params["ln10As"]
    A_s   = 1.0e-10 * math.exp(ln10As)
    T_cmb = params.get("T_cmb", T_CMB0)
    NEFF  = params.get("NEFF", 3.046)
    m_nu  = params.get("m_nu", 0.06)

    da = angular_diameter_da_to_rec(ombh2, omch2, T_cmb, NEFF, m_nu, H0)
    if da <= 0.0 or ell < 2:
        return 0.0
    k_ell = (ell + 0.5) / da
    p_r   = primordial_ps(k_ell, A_s, ns)
    T_k   = transfer_function_EH(k_ell, ombh2, omch2, T_cmb, NEFF, m_nu, H0)
    return 2.0 * math.pi * p_r * T_k ** 2 * 2.0 * math.pi / (ell * (ell + 1.0))


# ---------------------------------------------------------------------------
# Sanity check (run only when module is imported as __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = PLANCK_2018.copy()
    As = 1.0e-10 * math.exp(p["ln10As"])
    p["As"] = As
    print("Planck 2018 derived parameters:")
    print(f"  h         = {h_from_H0(p['H0']):.6f}")
    print(f"  Omega_b   = {omega_b(p['ombh2'], p['H0']):.6f}")
    print(f"  Omega_c   = {omega_c(p['omch2'], p['H0']):.6f}")
    print(f"  Omega_g   = {omega_gamma(p['T_cmb'], p['H0']):.6e}")
    print(f"  Omega_L   = {omega_lambda(p['ombh2'], p['omch2'], p['T_cmb'], p['NEFF'], p['m_nu'], p['H0']):.6f}")
    print(f"  r_s       = {sound_horizon_rs(p['ombh2'], p['omch2'], omega_gamma(p['T_cmb'], p['H0'])):.3f} Mpc")
    print(f"  theta_*   = {theta_star(p['ombh2'], p['omch2'], p['T_cmb'], p['NEFF'], p['m_nu'], p['H0']):.6e}")
    for ell in [2, 10, 100, 500, 1000, 2000]:
        print(f"  C_{{l={ell}}}  = {cmb_cl_theory(ell, p):.4e}")
