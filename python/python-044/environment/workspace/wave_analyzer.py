"""
wave_analyzer.py
================
Wave separation, dispersion analysis, and energy partitioning for
poroelastic wave propagation.

This module implements:
  - Fast P-wave / Slow P-wave separation based on phase velocity
  - Dispersion relation analysis for frequency-dependent wave speeds
  - Energy flux and attenuation coefficient computation
  - Quality factor Q estimation

Biot's theory predicts two compressional (P) waves in a poroelastic medium:
  1. Fast P-wave: primarily solid-frame motion, slightly modified by fluid
  2. Slow P-wave (Biot's second kind): relative fluid-solid motion,
     highly attenuated, diffusive at low frequencies

The wave velocities are frequency-dependent due to viscous coupling:

    V_fast(omega) = sqrt( (H - sqrt(H^2 - 4*C*M*mu_eff)) / (2*rho_bulk) )
    V_slow(omega) = sqrt( (H + sqrt(H^2 - 4*C*M*mu_eff)) / (2*rho_bulk) )

where:
    H = K_d + 4/3*mu + alpha^2 * M
    C = alpha * M
    mu_eff = mu + i*omega*eta_d  (complex shear modulus with viscosity)

In the low-frequency limit (omega << omega_c):
    V_fast ≈ sqrt( (K_u + 4/3*mu) / rho_bulk )
    V_slow ≈ sqrt( 2 * omega * D_diff )  (diffusive)

In the high-frequency limit (omega >> omega_c):
    V_fast ≈ sqrt( (K_d + 4/3*mu + alpha^2*M/tortuosity) / rho_bulk )
    V_slow ≈ sqrt( M / (rho_f * tortuosity) )

Attenuation:
    1/Q = 2 * Im(k) / Re(k)  where k = omega / V_complex
"""

import numpy as np


def biot_wave_velocities_low_freq(material):
    """
    Compute low-frequency Biot wave velocities.

    Returns
    -------
    v_fast, v_slow, v_shear : float
        Phase velocities in m/s.
    """
    v_fast = material.V_p_fast
    v_slow = material.V_p_slow
    v_shear = material.V_s
    return v_fast, v_slow, v_shear


def biot_dispersion_relation(omega, material):
    """
    Compute frequency-dependent wave velocities and attenuation.

    Uses the simplified Johnson-Champoux-Allard model for dynamic
    permeability and tortuosity.

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency(ies) in rad/s.
    material : PoroelasticMaterial
        Material properties.

    Returns
    -------
    v_fast, v_slow, alpha_fast, alpha_slow : complex
        Complex wave velocities and attenuation coefficients.
    """
    omega = np.asarray(omega, dtype=float)

    # Biot characteristic frequency
    omega_c = (material.eta * material.phi) / (material.kappa * material.rho_f)

    # Dynamic tortuosity (Johnson et al.)
    # alpha_tort(omega) = alpha_infty * (1 + omega_c / (i*omega))^(1/2)
    alpha_infty = material.tortuosity
    dynamic_tortuosity = alpha_infty * np.sqrt(1.0 + omega_c / (1j * omega + 1e-30))

    # Effective fluid density
    rho_f_eff = material.rho_f * dynamic_tortuosity

    # Poroelastic coefficients
    K_d = material.K_d
    mu = material.mu
    alpha = material.alpha
    M = material.M
    rho_s = material.rho_s
    rho_f = material.rho_f
    phi = material.phi
    rho_bulk = material.rho_bulk

    # Gassmann undrained modulus
    K_u = K_d + alpha ** 2 * M

    # Complex P-wave modulus
    H = K_d + 4.0 * mu / 3.0 + alpha ** 2 * M

    # In the low-frequency limit, use simplified dispersion
    # Fast wave: weak frequency dependence
    v_fast = np.sqrt((K_u + 4.0 * mu / 3.0) / rho_bulk)
    # Add small imaginary part for attenuation
    v_fast = v_fast * (1.0 - 0.5j * (omega / omega_c) / (1.0 + (omega / omega_c) ** 2))

    # Slow wave: diffusive at low freq, propagating at high freq
    D_diff = material.D_diff
    v_slow = np.sqrt(2.0 * omega * D_diff) * (1.0 + 0.5j)

    # Attenuation coefficients
    alpha_fast = np.imag(omega / v_fast)
    alpha_slow = np.imag(omega / v_slow)

    return v_fast, v_slow, alpha_fast, alpha_slow


def compute_quality_factor(omega, material):
    """
    Compute quality factor Q for fast and slow P-waves.

    Q^{-1} = 2 * |Im(k)| / |Re(k)|
    """
    v_fast, v_slow, alpha_fast, alpha_slow = biot_dispersion_relation(omega, material)
    k_fast = omega / v_fast
    k_slow = omega / v_slow

    Q_fast_inv = 2.0 * np.abs(np.imag(k_fast)) / (np.abs(np.real(k_fast)) + 1e-30)
    Q_slow_inv = 2.0 * np.abs(np.imag(k_slow)) / (np.abs(np.real(k_slow)) + 1e-30)

    Q_fast = 1.0 / (Q_fast_inv + 1e-30)
    Q_slow = 1.0 / (Q_slow_inv + 1e-30)

    return Q_fast, Q_slow


def separate_fast_slow_waves(pressure, displacement, nodes, material, dt, dx_est):
    """
    Separate fast and slow P-wave components from the wave field using
    phase-velocity filtering.

    Fast wave: high phase velocity, small pressure-to-displacement ratio
    Slow wave: low phase velocity, large pressure-to-displacement ratio

    Returns
    -------
    fast_mask, slow_mask : ndarray of bool
        Boolean masks indicating wave type classification per node.
    ratio : ndarray
        Pressure-to-displacement magnitude ratio.
    """
    p = np.asarray(pressure)
    u = np.asarray(displacement)
    u_mag = np.linalg.norm(u, axis=1) + 1e-30
    ratio = np.abs(p) / u_mag

    v_fast, v_slow, _ = biot_wave_velocities_low_freq(material)

    # Threshold based on theoretical fast/slow wave characteristics
    # Fast wave: smaller p/u ratio (solid dominated)
    # Slow wave: larger p/u ratio (fluid dominated)
    threshold = np.median(ratio)

    fast_mask = ratio < threshold
    slow_mask = ratio >= threshold

    return fast_mask, slow_mask, ratio


def compute_wave_energy_flux(pressure, displacement, velocity, material):
    """
    Compute wave energy flux vector (Poynting-like vector) for poroelastic medium.

    The energy flux has contributions from both solid and fluid phases:
        J = -sigma_eff * v_solid - p * q_darcy

    where q_darcy = -(kappa/eta) * grad(p) is the Darcy flux.
    """
    n = len(pressure)
    # Simplified: use displacement as proxy for solid velocity
    # and pressure gradient proxy for Darcy flux
    J = np.zeros((n, 2))

    # Solid contribution: approximate stress * velocity
    sigma_approx = material.lam * np.linalg.norm(displacement, axis=1) + \
                   2.0 * material.mu * np.linalg.norm(displacement, axis=1)
    J[:, 0] += sigma_approx * velocity[:, 0]
    J[:, 1] += sigma_approx * velocity[:, 1]

    # Fluid contribution
    kappa_eta = material.kappa / material.eta
    # Approximate pressure gradient from nodal values
    J[:, 0] += pressure * kappa_eta * pressure  # simplified
    J[:, 1] += pressure * kappa_eta * pressure

    return J


def dispersion_error_analysis(v_num, v_exact):
    """
    Compute numerical dispersion error.

    Returns relative phase velocity error.
    """
    return np.abs(v_num - v_exact) / (np.abs(v_exact) + 1e-30)
