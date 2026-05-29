"""
fresnel_coefficients.py
-----------------------
Fresnel reflection / transmission coefficients for stratified
plasma coatings, plus Fresnel-integral-based diffraction corrections
and least-squares inversion of reflection data.

Incorporates core ideas from:
  - 448_fresnel  (Fresnel sine/cosine integrals)
  - 692_llsq     (linear least-squares fitting)
"""

import numpy as np
import math
from utils import fresnel_cos, fresnel_sin, llsq_fit, llsq_fit_through_origin, safe_divide, clamp


def fresnel_reflection_coefficient(
    n1: complex, n2: complex, theta1: float, polarization: str = "TE"
) -> complex:
    """
    Compute the Fresnel reflection coefficient r at a planar interface
    between two media with complex refractive indices n1 and n2.

    For TE (s-polarization):
        r_TE = (n1*cos(theta1) - n2*cos(theta2)) /
               (n1*cos(theta1) + n2*cos(theta2))

    For TM (p-polarization):
        r_TM = (n2*cos(theta1) - n1*cos(theta2)) /
               (n2*cos(theta1) + n1*cos(theta2))

    where theta2 follows Snell's law:
        n1 * sin(theta1) = n2 * sin(theta2)

    Parameters
    ----------
    n1, n2 : complex
        Complex refractive indices (n = sqrt(eps_r)).
    theta1 : float
        Incidence angle [rad] measured from normal.
    polarization : str
        "TE" or "TM".

    Returns
    -------
    r : complex
        Reflection coefficient (amplitude).
    """
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    n1 = complex(n1)
    n2 = complex(n2)

    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1
    # Clamp to avoid numerical overflow inside asin
    sin_theta2 = complex(
        clamp(sin_theta2.real, -1.0, 1.0),
        sin_theta2.imag,
    )
    theta2 = np.arcsin(sin_theta2)

    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)

    if polarization.upper() == "TE":
        num = n1 * cos_theta1 - n2 * cos_theta2
        den = n1 * cos_theta1 + n2 * cos_theta2
    elif polarization.upper() == "TM":
        num = n2 * cos_theta1 - n1 * cos_theta2
        den = n2 * cos_theta1 + n1 * cos_theta2
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")

    r = safe_divide(num, den, fallback=0.0)
    return complex(r)


def fresnel_transmission_coefficient(
    n1: complex, n2: complex, theta1: float, polarization: str = "TE"
) -> complex:
    """
    Compute the Fresnel transmission coefficient t at a planar interface.

    For TE:
        t_TE = 2*n1*cos(theta1) / (n1*cos(theta1) + n2*cos(theta2))
    For TM:
        t_TM = 2*n1*cos(theta1) / (n2*cos(theta1) + n1*cos(theta2))
    """
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    n1 = complex(n1)
    n2 = complex(n2)

    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1
    sin_theta2 = complex(
        clamp(sin_theta2.real, -1.0, 1.0),
        sin_theta2.imag,
    )
    theta2 = np.arcsin(sin_theta2)

    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)

    if polarization.upper() == "TE":
        den = n1 * cos_theta1 + n2 * cos_theta2
        num = 2.0 * n1 * cos_theta1
    elif polarization.upper() == "TM":
        den = n2 * cos_theta1 + n1 * cos_theta2
        num = 2.0 * n1 * cos_theta1
    else:
        raise ValueError("polarization must be 'TE' or 'TM'.")

    t = safe_divide(num, den, fallback=0.0)
    return complex(t)


def reflection_power_ratio(r: complex) -> float:
    """Return reflected power ratio R = |r|^2."""
    return abs(r) ** 2


def transmission_power_ratio(
    r: complex, n1: complex, n2: complex, theta1: float
) -> float:
    """
    Return transmitted power ratio T for a non-magnetic medium.

    T = (n2.real / n1.real) * |t|^2 * (cos(theta2)/cos(theta1))
    """
    theta1 = clamp(float(theta1), 0.0, math.pi / 2.0 - 1e-6)
    t = 1.0 + r  # approximate for small r, but exact via energy conservation:
    # Better: use 1 - R - A_interface ... for now use exact from t
    # Recompute t
    n1 = complex(n1)
    n2 = complex(n2)
    sin_theta1 = math.sin(theta1)
    sin_theta2 = (n1 / n2) * sin_theta1
    sin_theta2 = complex(clamp(sin_theta2.real, -1.0, 1.0), sin_theta2.imag)
    theta2 = np.arcsin(sin_theta2)
    cos_theta1 = math.cos(theta1)
    cos_theta2 = np.cos(theta2)

    # For TE
    den = n1 * cos_theta1 + n2 * cos_theta2
    num = 2.0 * n1 * cos_theta1
    t_exact = safe_divide(num, den, fallback=0.0)
    t_exact = complex(t_exact)

    ratio = safe_divide(
        (n2.real / max(n1.real, 1e-15)) * abs(t_exact) ** 2 * (cos_theta2.real / cos_theta1),
        1.0,
        fallback=0.0,
    )
    # Clamp physically to [0, 1] approximately
    return clamp(float(ratio), 0.0, 2.0)


def diffraction_correction_fresnel(
    x: float, y: float, z: float, k0: float
) -> complex:
    """
    Compute a first-order diffraction correction factor using Fresnel integrals.

    For a wave passing through a knife-edge-like gradient in the plasma density,
    the phase perturbation can be approximated by the Fresnel parameter

        v = sqrt(2*k0/pi) * (y + z) / sqrt(x)

    and the diffraction factor is

        D_f = 0.5 * (1 + i) * [ C(v) + i*S(v) ]

    Parameters
    ----------
    x, y, z : float
        Spatial coordinates [m].
    k0 : float
        Free-space wave number [m^-1].

    Returns
    -------
    D_f : complex
    """
    x = max(float(x), 1e-6)
    y = float(y)
    z = float(z)
    k0 = max(float(k0), 1e-6)

    v = math.sqrt(2.0 * k0 / math.pi) * (y + z) / math.sqrt(x)
    c_v = fresnel_cos(v)
    s_v = fresnel_sin(v)

    D_f = 0.5 * (1.0 + 1j) * (c_v + 1j * s_v)
    return D_f


def invert_reflection_for_epsilon(
    frequencies: np.ndarray,
    measured_R: np.ndarray,
    theta: float = 0.0,
    polarization: str = "TE",
) -> tuple:
    """
    Use linear least-squares to fit a simplified Drude-like model to
    measured reflection data.

    Model (normal incidence, vacuum -> plasma):
        R(omega) ~ | (1 - sqrt(eps)) / (1 + sqrt(eps)) |^2

    For weakly collisional plasmas (nu << omega), eps_r ~ 1 - A/omega^2.
    We linearize and fit:
        sqrt( (1-sqrt(R)) / (1+sqrt(R)) )  ~  linear in 1/omega^2

    This is a simplified inversion for demonstration; full inversion
    requires nonlinear optimization.

    Parameters
    ----------
    frequencies : (N,) ndarray
        Frequencies [Hz].
    measured_R : (N,) ndarray
        Measured power reflection coefficients.
    theta : float
        Incidence angle [rad].
    polarization : str
        "TE" or "TM".

    Returns
    -------
    (A_fit, residual, rms_error)
        A_fit is the fitted coefficient related to omega_p^2.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    measured_R = np.asarray(measured_R, dtype=float)
    if frequencies.size != measured_R.size or frequencies.size < 3:
        raise ValueError("Need at least 3 frequency points.")

    # Guard against unphysical R values
    measured_R = np.clip(measured_R, 1e-6, 1.0 - 1e-6)
    omega = 2.0 * math.pi * frequencies

    # Transform variable: xi = 1/omega^2
    xi = 1.0 / (omega ** 2)

    # For normal incidence, approximate model:
    #   n = sqrt(eps) ~ 1 - 0.5*A/omega^2
    #   r = (1 - n)/(1 + n) ~ A/(4*omega^2)
    #   sqrt(R) ~ A/(4*omega^2)
    # So we fit  sqrt(R) = slope * xi
    # where slope = A/4

    sqrt_R = np.sqrt(measured_R)
    slope, residual = llsq_fit_through_origin(xi, sqrt_R)
    A_fit = 4.0 * slope

    # RMS error
    predicted = slope * xi
    rms = np.sqrt(np.mean((sqrt_R - predicted) ** 2))

    return A_fit, residual, rms


def multilayer_reflection_stack(
    n_layers: np.ndarray,
    d_layers: np.ndarray,
    omega: float,
    theta0: float = 0.0,
    polarization: str = "TE",
) -> complex:
    """
    Compute total reflection coefficient for a multilayer stack using
    the characteristic transfer-matrix method (TMM).

    Each layer j has complex refractive index n_j and thickness d_j.
    The stack is bounded by vacuum (n_0 = 1) on both sides.

    The transfer matrix for layer j is:
        M_j = [ cos(phi_j)       -i*sin(phi_j)/eta_j ;
               -i*eta_j*sin(phi_j)   cos(phi_j)      ]

    where phi_j = k0 * n_j * d_j * cos(theta_j)  (phase thickness)
    and   eta_j = n_j * cos(theta_j)   for TE
          eta_j = n_j / cos(theta_j)   for TM

    The total reflection coefficient is
        r = (M_11 + M_12*eta_s - eta_0*(M_21 + M_22*eta_s)) /
            (M_11 + M_12*eta_s + eta_0*(M_21 + M_22*eta_s))

    Parameters
    ----------
    n_layers : (N,) ndarray of complex
        Refractive index of each layer.
    d_layers : (N,) ndarray of float
        Thickness of each layer [m].
    omega : float
        Angular frequency [rad/s].
    theta0 : float
        Incidence angle in vacuum [rad].
    polarization : str
        "TE" or "TM".

    Returns
    -------
    r_total : complex
    """
    # TODO: Implement the transfer-matrix method (TMM) for multilayer reflection.
    #
    # Steps:
    # 1. Validate inputs and set k0 = omega / c.
    # 2. Initialize total transfer matrix M as 2x2 identity.
    # 3. For each layer j:
    #    - Apply Snell's law to get cos(theta_j).
    #    - Compute admittance eta_j (TE: nj*cos(theta_j), TM: nj/cos(theta_j)).
    #    - Compute phase thickness phi_j = k0 * nj * dj * cos(theta_j).
    #    - Build layer matrix Mj and accumulate M = M @ Mj.
    # 4. Compute eta0 and eta_s for vacuum substrate.
    # 5. Extract M elements and compute total reflection:
    #        r = (M11 + M12*eta_s - eta0*(M21 + M22*eta_s)) /
    #            (M11 + M12*eta_s + eta0*(M21 + M22*eta_s))
    # 6. Return r_total as a complex number.
    raise NotImplementedError("Hole 2: multilayer_reflection_stack not implemented.")
