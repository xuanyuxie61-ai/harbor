"""
plasma_drude_model.py
---------------------
Drude-model description of the complex dielectric function
of a collisional, non-magnetized plasma.

Physics
-------
For an unmagnetized plasma with electron density n_e and
electron-neutral collision frequency nu, the Drude model gives:

    eps_r(omega) = 1 - omega_p^2 / (omega^2 + nu^2)
    eps_i(omega) = nu * omega_p^2 / (omega * (omega^2 + nu^2))

where the plasma frequency is

    omega_p = sqrt( n_e * e^2 / (m_e * eps_0) )

The complex wave number inside the plasma is

    k = (omega / c) * sqrt(eps_r + i*eps_i)
      = k0 * sqrt(1 - omega_p^2/(omega^2 + nu^2)
                    + i*nu*omega_p^2/(omega*(omega^2+nu^2)))

This module also solves the nonlinear dispersion relation for
collisional surface waves using Newton iteration (from 808_nonlin_newton).
"""

import numpy as np
import math
from utils import newton_solve, ensure_positive, safe_sqrt, clamp

# Physical constants (SI)
E_CHARGE = 1.602176634e-19      # C
E_MASS   = 9.10938356e-31       # kg
EPS_0    = 8.854187817e-12      # F/m
MU_0     = 4.0 * math.pi * 1e-7 # H/m
C_LIGHT  = 299792458.0          # m/s


def plasma_frequency(n_e: float) -> float:
    """
    Compute plasma angular frequency omega_p [rad/s].

    Parameters
    ----------
    n_e : float
        Electron number density [m^-3].

    Returns
    -------
    omega_p : float
    """
    n_e = max(float(n_e), 1.0)
    omega_p = math.sqrt(n_e * E_CHARGE ** 2 / (E_MASS * EPS_0))
    return omega_p


def drude_permittivity(n_e: float, nu: float, omega: float) -> complex:
    """
    Compute complex relative permittivity eps = eps_r + i*eps_i.

    Parameters
    ----------
    n_e : float
        Electron density [m^-3].
    nu : float
        Collision frequency [rad/s].
    omega : float
        Angular wave frequency [rad/s].

    Returns
    -------
    eps : complex
    """
    # TODO: Implement the Drude-model complex permittivity.
    #
    # eps = eps_r + i*eps_i where
    #   eps_r = 1 - omega_p^2 / (omega^2 + nu^2)
    #   eps_i = nu * omega_p^2 / (omega * (omega^2 + nu^2))
    # and omega_p = sqrt(n_e * e^2 / (m_e * eps_0)).
    #
    # Apply numerical guards for extreme values.
    # Return a Python complex number.
    raise NotImplementedError("Hole 1: drude_permittivity not implemented.")


def wave_number(n_e: float, nu: float, omega: float) -> complex:
    """
    Compute complex wave number k = k_r + i*k_i [m^-1].

    k = (omega / c) * sqrt(eps)
    """
    eps = drude_permittivity(n_e, nu, omega)
    k0 = omega / C_LIGHT
    k = k0 * np.sqrt(eps)
    return complex(k)


def collision_frequency_from_temperature(T_e: float, p_gas: float,
                                        gas_type: str = "air") -> float:
    """
    Estimate electron-neutral collision frequency [rad/s] from
    electron temperature and gas pressure.

    Approximate model:
        nu = n_gas * sigma_cn * v_th

    where n_gas = p_gas/(k_B*T_gas) and
    v_th = sqrt(8*k_B*T_e/(pi*m_e)).

    Parameters
    ----------
    T_e : float
        Electron temperature [K].
    p_gas : float
        Gas pressure [Pa].
    gas_type : str
        Only "air" supported in this simplified model.

    Returns
    -------
    nu : float
    """
    T_e = max(float(T_e), 100.0)
    p_gas = max(float(p_gas), 1.0)
    k_B = 1.380649e-23  # J/K
    T_gas = 300.0       # assume ambient gas temperature

    n_gas = p_gas / (k_B * T_gas)
    sigma_cn = 1.0e-19  # m^2, approximate cross-section for air
    v_th = math.sqrt(8.0 * k_B * T_e / (math.pi * E_MASS))
    nu = n_gas * sigma_cn * v_th
    return nu


def skin_depth(n_e: float, nu: float, omega: float) -> float:
    """
    Compute the electromagnetic skin depth delta [m] inside the plasma.

    For a highly collisional plasma (nu >> omega) or highly overdense
    (omega_p >> omega), the skin depth is approximately

        delta ~ c / (omega_p * sqrt(1 - i*nu/omega))

    Here we compute the exact magnitude-based skin depth:
        delta = 1 / Im(k)
    """
    k = wave_number(n_e, nu, omega)
    ki = float(k.imag)
    if abs(ki) < 1e-12:
        return 1e6  # effectively transparent
    return 1.0 / ki


def nonlinear_dispersion_relation(
    omega: float,
    n_e: float,
    nu: float,
    k_guess: float = None,
) -> tuple:
    """
    Solve the nonlinear dispersion relation for a collisional plasma
    surface wave (s-polarized) on a planar interface:

        D(k) = k^2 + kappa_m^2 - (omega^2/c^2) * (eps_p - eps_m) = 0

    where kappa_m = sqrt(k^2 - (omega/c)^2 * eps_m) and eps_m = 1 (vacuum).
    This is a highly simplified model used to demonstrate Newton iteration.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    n_e : float
        Plasma density [m^-3].
    nu : float
        Collision frequency [rad/s].
    k_guess : float, optional
        Initial guess for wave number [m^-1]. Defaults to omega/c.

    Returns
    -------
    (k_solution, iterations, converged)
    """
    omega = float(omega)
    n_e = float(n_e)
    nu = float(nu)
    eps_p = drude_permittivity(n_e, nu, omega)
    eps_m = 1.0 + 0j

    k0 = omega / C_LIGHT
    if k_guess is None:
        # For surface waves, k > k0 is required; start slightly above k0
        k_guess = k0 * 1.2

    def D(k):
        """TM surface-wave dispersion: eps_p*kappa_m + eps_m*kappa_p = 0."""
        if k <= k0:
            return 1e6
        kappa_m = np.sqrt(k ** 2 - k0 ** 2 * eps_m)
        kappa_p = np.sqrt(k ** 2 - k0 ** 2 * eps_p)
        # Branch cut: positive real part
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

    # Custom Newton with boundary guard to keep k > k0
    x = float(k_guess)
    for it in range(1, 201):
        fx = D(x)
        if abs(fx) < 1e-12:
            return x, it, True
        fpx = Dp(x)
        if abs(fpx) < 1e-14:
            break
        step = fx / fpx
        # Damping if step is large
        if abs(step) > 0.3 * x:
            step = 0.1 * step
        x_new = x - step
        # Enforce k > k0 + small margin
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
    """
    Compute the complex permittivity profile along the coating depth z.

    Parameters
    ----------
    z : (Nz,) ndarray
        Depth coordinates [m], increasing into the coating.
    n_e_profile : (Nz,) ndarray
        Electron density at each depth [m^-3].
    nu_profile : (Nz,) ndarray
        Collision frequency at each depth [rad/s].
    omega : float
        Angular frequency [rad/s].

    Returns
    -------
    eps : (Nz,) ndarray of complex
    """
    z = np.asarray(z, dtype=float)
    n_e_profile = ensure_positive(np.asarray(n_e_profile, dtype=float))
    nu_profile = ensure_positive(np.asarray(nu_profile, dtype=float))
    omega = float(omega)

    eps = np.empty_like(z, dtype=complex)
    for i in range(z.size):
        eps[i] = drude_permittivity(n_e_profile[i], nu_profile[i], omega)
    return eps


def conductivity(n_e: float, nu: float, omega: float) -> complex:
    """
    Compute complex conductivity sigma = sigma_r + i*sigma_i [S/m].

    From the Drude model:
        sigma = eps_0 * omega_p^2 / (nu - i*omega)
              = eps_0 * omega_p^2 * (nu + i*omega) / (nu^2 + omega^2)
    """
    omega_p = plasma_frequency(n_e)
    omega_p2 = omega_p ** 2
    denom = nu ** 2 + omega ** 2
    sigma_r = EPS_0 * omega_p2 * nu / denom
    sigma_i = EPS_0 * omega_p2 * omega / denom
    return complex(sigma_r, sigma_i)
