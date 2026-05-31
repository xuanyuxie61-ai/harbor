"""
Elliptic Integrals and Special Functions Module
================================================
Based on project 327_elfun.

Provides complete and incomplete elliptic integrals of the first,
second, and third kinds, Jacobi elliptic functions, and related
special functions. These arise in advanced structural mechanics for:

1. Large-deflection analysis of beams and plates (elastica problem):
   The curvature equation involves elliptic integrals.
   
2. Nonlinear vibration of structures:
   Period of pendulum-like oscillations involves K(k).
   
3. Contact mechanics and crack propagation:
   Stress intensity factors near elliptical holes.

Key formulas:
- Complete elliptic integral of 1st kind:
  K(k) = integral_0^{pi/2} (1 - k^2 sin^2 theta)^{-1/2} dtheta
  
- Complete elliptic integral of 2nd kind:
  E(k) = integral_0^{pi/2} (1 - k^2 sin^2 theta)^{1/2} dtheta
  
- Jacobi amplitude: am(u|m) where m = k^2
- Jacobi elliptic functions: sn(u|m), cn(u|m), dn(u|m)
"""

import numpy as np
from scipy.special import ellipk, ellipe, ellipj


def complete_elliptic_k(m):
    """
    Complete elliptic integral of the first kind K(m).
    
    K(m) = integral_0^{pi/2} (1 - m*sin^2(theta))^{-1/2} dtheta
    
    Parameter m = k^2 (parameter), not modulus k.
    
    Parameters
    ----------
    m : float or ndarray
        Parameter (0 <= m <= 1).
    
    Returns
    -------
    K : float or ndarray
    """
    m = np.asarray(m, dtype=float)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    return ellipk(m)


def complete_elliptic_e(m):
    """
    Complete elliptic integral of the second kind E(m).
    
    E(m) = integral_0^{pi/2} (1 - m*sin^2(theta))^{1/2} dtheta
    
    Parameters
    ----------
    m : float or ndarray
    
    Returns
    -------
    E : float or ndarray
    """
    m = np.asarray(m, dtype=float)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    return ellipe(m)


def jacobi_elliptic_functions(u, m):
    """
    Jacobi elliptic functions sn, cn, dn.
    
    sn(u|m) = sin(phi) where u = F(phi|m)
    cn(u|m) = cos(phi)
    dn(u|m) = sqrt(1 - m*sin^2(phi))
    
    Parameters
    ----------
    u : float or ndarray
    m : float
    
    Returns
    -------
    sn, cn, dn : tuple of float or ndarray
    """
    m = float(m)
    m = np.clip(m, 0.0, 1.0)
    u = np.asarray(u, dtype=float)
    sn, cn, dn = ellipj(u, m)
    return sn, cn, dn


def elastica_beam_deflection(P, EI, L, n_points=100):
    """
    Compute large-deflection shape of a cantilever beam under end load.
    
    The elastica equation:
    EI * d^2theta/ds^2 + P * sin(theta) = 0
    
    Solution in terms of elliptic integrals:
    theta_max = 2*arcsin(k)
    where k = sin(theta_max/2) is related to load parameter.
    
    The deflection curve:
    x(s) = s - (2*E(am(s*|P|/EI)|k^2) - F(am(...)|k^2)) / something
    
    For simplicity, we use the parametric form:
    x = (2*E(k) - F(phi,k)) / sqrt(P/EI)
    y = 2*k / sqrt(P/EI) * (1 - cos(phi))
    
    Parameters
    ----------
    P : float
        End load.
    EI : float
        Flexural rigidity.
    L : float
        Beam length.
    n_points : int
    
    Returns
    -------
    x, y : ndarray
        Deflection coordinates.
    theta_max : float
        Maximum tip angle.
    k_modulus : float
        Elliptic modulus.
    """
    if P <= 0 or EI <= 0 or L <= 0:
        return np.linspace(0, L, n_points), np.zeros(n_points), 0.0, 0.0
    
    # Load parameter: lambda^2 = P*L^2 / EI
    lam_sq = P * L ** 2 / EI
    lam = np.sqrt(lam_sq)
    
    # For cantilever with end load, theta_max satisfies:
    # F(theta_max/2 | k^2) = lam / 2, with k = sin(theta_max/2)
    # This is implicit; we solve numerically
    
    def residual(theta_m):
        if theta_m <= 0 or theta_m >= np.pi:
            return 1e10
        k = np.sin(theta_m / 2.0)
        m = k ** 2
        return complete_elliptic_k(m) - lam / 2.0
    
    # Simple bisection for theta_max
    th_lo, th_hi = 0.01, np.pi - 0.01
    for _ in range(50):
        th_mid = (th_lo + th_hi) / 2.0
        if residual(th_mid) > 0:
            th_hi = th_mid
        else:
            th_lo = th_mid
    
    theta_max = (th_lo + th_hi) / 2.0
    k_modulus = np.sin(theta_max / 2.0)
    m = k_modulus ** 2
    
    # Generate deflection curve parametrically
    phi = np.linspace(0, np.pi / 2.0, n_points)
    
    # x/L and y/L in terms of elliptic integrals
    # For the cantilever elastica:
    # x = sqrt(EI/P) * (2*E(k) - E(phi,k)) 
    # y = 2*k * sqrt(EI/P) * (1 - cos(phi))
    
    scale = np.sqrt(EI / P)
    
    # Approximate using Jacobi elliptic functions
    # sn(u|m) where u ranges from 0 to K(m)
    K_val = complete_elliptic_k(m)
    u_vals = np.linspace(0, K_val, n_points)
    
    sn, cn, dn = jacobi_elliptic_functions(u_vals, m)
    
    # Elastica parametric equations
    x = scale * (u_vals - complete_elliptic_e(m) * u_vals / K_val + 
                 np.cumsum(dn ** 2) * (K_val / n_points))
    # Better parametric form
    x = scale * (u_vals - ellipe(m) * u_vals / K_val)
    # Actually, let's use the standard formula more carefully
    # x(s) = integral_0^s cos(theta(t)) dt
    # theta(s) = 2 * arcsin(k * sn(s * sqrt(P/EI) | m))
    s_vals = np.linspace(0, L, n_points)
    u_param = s_vals * np.sqrt(P / EI)
    
    sn_s, cn_s, dn_s = jacobi_elliptic_functions(u_param, m)
    theta_s = 2.0 * np.arcsin(np.clip(k_modulus * sn_s, -1.0, 1.0))
    
    # Integrate to get x and y
    dx = np.cos(theta_s)
    dy = np.sin(theta_s)
    x = np.cumsum(dx) * (L / n_points)
    y = np.cumsum(dy) * (L / n_points)
    
    return x, y, theta_max, k_modulus


def nonlinear_vibration_period(amplitude, omega_linear, alpha_nonlin):
    """
    Compute period of nonlinear Duffing oscillator using elliptic integrals.
    
    Equation: d^2x/dt^2 + omega_0^2 * x + alpha * x^3 = 0
    
    For hardening spring (alpha > 0):
    T = 4 * K(k) / sqrt(omega_0^2 + alpha*A^2)
    where k^2 = alpha*A^2 / (2*(omega_0^2 + alpha*A^2))
    
    Parameters
    ----------
    amplitude : float
        Vibration amplitude A.
    omega_linear : float
        Linear natural frequency omega_0.
    alpha_nonlin : float
        Nonlinear stiffness coefficient.
    
    Returns
    -------
    T : float
        Period.
    T_linear : float
        Linear period for comparison.
    """
    if amplitude <= 0 or omega_linear <= 0:
        return 2 * np.pi / omega_linear, 2 * np.pi / omega_linear
    
    omega_sq = omega_linear ** 2
    
    if alpha_nonlin > 0:
        # Hardening spring
        denom = np.sqrt(omega_sq + alpha_nonlin * amplitude ** 2)
        m = alpha_nonlin * amplitude ** 2 / (2.0 * (omega_sq + alpha_nonlin * amplitude ** 2))
        m = np.clip(m, 0.0, 1.0)
        T = 4.0 * complete_elliptic_k(m) / denom
    elif alpha_nonlin < 0:
        # Softening spring (bounded amplitude)
        alpha_abs = abs(alpha_nonlin)
        if amplitude >= np.sqrt(omega_sq / alpha_abs):
            amplitude = 0.99 * np.sqrt(omega_sq / alpha_abs)
        denom = np.sqrt(omega_sq - alpha_abs * amplitude ** 2 / 2.0)
        m = alpha_abs * amplitude ** 2 / (2.0 * omega_sq - alpha_abs * amplitude ** 2)
        m = np.clip(m, 0.0, 1.0)
        T = 4.0 * complete_elliptic_k(m) / denom
    else:
        T = 2 * np.pi / omega_linear
    
    T_linear = 2 * np.pi / omega_linear
    return T, T_linear


def elliptical_hole_stress_concentration(a, b, sigma_inf, theta):
    """
    Stress concentration around elliptical hole in infinite plate.
    
    For an elliptical hole with semi-axes a (horizontal) and b (vertical)
    under remote tension sigma_inf perpendicular to major axis:
    
    sigma_theta = sigma_inf * (sinh(2*xi_0) + cos(2*eta) - exp(2*xi_0)*cos(2*(eta-theta))) /
                             (cosh(2*xi_0) - cos(2*eta))
    
    Simplified using elliptic coordinates (xi, eta):
    a = c*cosh(xi_0), b = c*sinh(xi_0), c = sqrt(a^2 - b^2)
    
    Maximum stress at hole tip (theta=0, eta=0):
    sigma_max = sigma_inf * (1 + 2*a/b)
    
    Parameters
    ----------
    a, b : float
        Semi-axes of ellipse (a >= b).
    sigma_inf : float
        Remote stress.
    theta : float or ndarray
        Angle around hole (radians).
    
    Returns
    -------
    sigma_theta : float or ndarray
        Tangential stress on hole boundary.
    stress_concentration_factor : float
    """
    if a < b:
        a, b = b, a
    
    # Stress concentration factor
    if b > 1e-14:
        kt = 1.0 + 2.0 * a / b
    else:
        kt = 1e10  # Crack limit
    
    # Angular variation on hole boundary
    # For tension perpendicular to major axis:
    # sigma_theta = sigma_inf * (1 + 2*a/b * cos(2*theta) - (a/b)^2 * cos(2*theta)) ...
    # Simplified: using Inglis solution
    sigma_theta = sigma_inf * (1.0 + 2.0 * a / b * np.cos(2.0 * theta) + 
                                (a / b) ** 2 * np.sin(theta) ** 2)
    
    return sigma_theta, kt
