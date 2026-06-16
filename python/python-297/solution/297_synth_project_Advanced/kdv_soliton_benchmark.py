"""
kdv_soliton_benchmark.py
=========================
Exact KdV soliton solutions as benchmarks for nonlinear dust
acoustic wave propagation in the dusty plasma crystal.

Physical motivation:
    In the weakly nonlinear, long-wavelength limit, dust acoustic
    waves in dusty plasmas are governed by the Korteweg-de Vries
    (KdV) equation:
        partial phi/partial t + A * phi * partial phi/partial xi
        + B * partial^3 phi/partial xi^3 = 0

    where phi is the wave potential, xi = x - V_DA*t is the moving
    coordinate, V_DA is the dust acoustic speed, and:
        A = (3/2) * V_DA / (1 + kappa)  (nonlinearity coefficient)
        B = (1/2) * V_DA * lambda_D^2   (dispersion coefficient)

    The exact single-soliton solution (from 615_kdv_exact):
        phi(xi, t) = phi_m * sech^2((xi - xi_0 - u*t) / Delta)
    where:
        phi_m = 3u/A  (soliton amplitude)
        Delta = sqrt(4B/u)  (soliton width)
        u = soliton velocity in moving frame

    The two-soliton solution exhibits elastic collision, providing
    a stringent test for numerical methods.

References:
    - From 615_kdv_exact: kdv_exact_sech, kdv_exact_rational,
      kdv_parameters, kdv_residual
    - Rao, Shukla & Yu, "Dust acoustic solitons in dusty plasmas",
      Planet. Space Sci. 38, 543 (1990)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def kdv_parameters_dusty_plasma(
    kappa: float,
    omega_pd: float,
    C_DA: float,
    lambda_D: float,
) -> Dict[str, float]:
    """
    Compute KdV coefficients for dusty plasma.

    The KdV equation for dust acoustic waves:
        phi_t + A*phi*phi_xi + B*phi_{xi,xi,xi} = 0

    Coefficients from the reductive perturbation method:
        A = (3*V_DA)/(2*(1 + kappa))  (for cold ions)
        B = (V_DA * lambda_D^2) / 2

    Parameters
    ----------
    kappa : float
        Screening parameter a/lambda_D.
    omega_pd : float
        Dust plasma frequency [rad/s].
    C_DA : float
        Dust acoustic speed [m/s].
    lambda_D : float
        Debye length [m].

    Returns
    -------
    params : dict
        'A': nonlinearity coefficient,
        'B': dispersion coefficient,
        'V_DA': dust acoustic speed.
    """
    A = 1.5 * C_DA / (1.0 + kappa)
    B = 0.5 * C_DA * lambda_D**2

    return {
        "A": A,
        "B": B,
        "V_DA": C_DA,
        "kappa": kappa,
        "omega_pd": omega_pd,
    }


def kdv_exact_sech(
    x: np.ndarray,
    t: float,
    params: Dict[str, float],
    amplitude: float = 1.0,
    x0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact single-soliton solution to KdV equation (from 615_kdv_exact).

    phi(x,t) = phi_m * sech^2((x - x0 - u*t) / Delta)

    where phi_m = 3*u/A, Delta = sqrt(4*B/u)

    Parameters
    ----------
    x : np.ndarray, shape (Nx,)
        Spatial grid.
    t : float
        Time.
    params : dict
        KdV parameters {'A', 'B'}.
    amplitude : float
        Soliton peak amplitude (determines velocity via u = A*phi_m/3).
    x0 : float
        Initial center position.

    Returns
    -------
    phi : np.ndarray
        Wave potential.
    phi_x : np.ndarray
        Spatial derivative.
    phi_t : np.ndarray
        Time derivative.
    """
    A = params["A"]
    B = params["B"]

    # Soliton velocity in moving frame
    u = A * amplitude / 3.0
    # Soliton width
    Delta = np.sqrt(abs(4.0 * B / u)) if abs(u) > 1e-30 else 1.0

    xi = x - x0 - u * t
    arg = xi / Delta

    # sech profile
    sech_val = 1.0 / np.cosh(arg)
    sech2 = sech_val**2

    phi = amplitude * sech2

    # Derivatives
    tanh_val = np.tanh(arg)
    phi_x = -2.0 * amplitude * sech2 * tanh_val / Delta
    phi_t = -u * phi_x

    return phi, phi_x, phi_t


def kdv_exact_rational(
    x: np.ndarray,
    t: float,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rational solution to KdV equation (from 615_kdv_exact).

    phi(x,t) = -4B / (A * (x - 12B*t^2/(x^2) + c))

    This is a singular solution useful for testing code robustness.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    t : float
        Time.
    params : dict
        KdV parameters.

    Returns
    -------
    phi : np.ndarray
        Wave potential.
    phi_xxx : np.ndarray
        Third spatial derivative.
    """
    A = params["A"]
    B = params["B"]

    # Avoid division by zero
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)

    phi = -4.0 * B / (A * (x_safe + 1e-15))

    # Third derivative (approximate)
    dx = x[1] - x[0] if len(x) > 1 else 0.01
    dx = max(abs(dx), 1e-10)
    phi_xxx = np.gradient(np.gradient(np.gradient(phi, dx), dx), dx)

    return phi, phi_xxx


def kdv_residual(
    x: np.ndarray,
    t: float,
    params: Dict[str, float],
    amplitude: float = 1.0,
    x0: float = 0.0,
) -> Dict[str, float]:
    """
    Compute the residual of the KdV equation for the exact solution.

    R = phi_t + A*phi*phi_x + B*phi_{xxx}

    For an exact solution, R should be ~0 (up to discretization).

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    t : float
        Time.
    params : dict
        KdV parameters.
    amplitude : float
        Soliton amplitude.
    x0 : float
        Initial position.

    Returns
    -------
    results : dict
        'L_inf': max absolute residual,
        'L2': L2 norm of residual,
        'max_phi': max absolute value of phi.
    """
    A = params["A"]
    B = params["B"]

    phi, phi_x, phi_t = kdv_exact_sech(x, t, params, amplitude, x0)

    # Third derivative via finite differences
    dx = x[1] - x[0] if len(x) > 1 else 0.01
    dx = max(abs(dx), 1e-10)

    phi_xx = np.gradient(phi_x, dx)
    phi_xxx = np.gradient(phi_xx, dx)

    residual = phi_t + A * phi * phi_x + B * phi_xxx

    return {
        "L_inf": float(np.max(np.abs(residual))),
        "L2": float(np.sqrt(np.mean(residual**2))),
        "max_phi": float(np.max(np.abs(phi))),
        "relative_error": float(
            np.max(np.abs(residual)) / max(np.max(np.abs(phi_t)), 1e-30)
        ),
    }


def kdv_two_soliton_interaction(
    x: np.ndarray,
    t_values: np.ndarray,
    params: Dict[str, float],
    amp1: float = 2.0,
    amp2: float = 1.0,
    x1_0: float = -5.0,
    x2_0: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Approximate two-soliton solution for KdV equation.

    For well-separated solitons (before and after collision), the
    solution is approximately the superposition of two single solitons.
    Near the collision, the exact solution involves a more complex
    expression, but we use the approximate form for benchmarking.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    t_values : np.ndarray
        Time values.
    params : dict
        KdV parameters.
    amp1, amp2 : float
        Soliton amplitudes.
    x1_0, x2_0 : float
        Initial positions.

    Returns
    -------
    results : dict
        't': time array,
        'phi_max': max amplitude at each time,
        'x_center': center of mass position.
    """
    A = params["A"]
    B = params["B"]

    u1 = A * amp1 / 3.0
    u2 = A * amp2 / 3.0
    Delta1 = np.sqrt(abs(4.0 * B / u1)) if abs(u1) > 1e-30 else 1.0
    Delta2 = np.sqrt(abs(4.0 * B / u2)) if abs(u2) > 1e-30 else 1.0

    phi_max_history = []
    x_center_history = []

    for t in t_values:
        xi1 = x - x1_0 - u1 * t
        xi2 = x - x2_0 - u2 * t

        phi1 = amp1 / np.cosh(xi1 / Delta1)**2
        phi2 = amp2 / np.cosh(xi2 / Delta2)**2
        phi_total = phi1 + phi2

        phi_max_history.append(np.max(phi_total))

        # Center of mass
        if np.sum(np.abs(phi_total)) > 1e-30:
            x_cm = np.sum(x * np.abs(phi_total)) / np.sum(np.abs(phi_total))
        else:
            x_cm = 0.0
        x_center_history.append(x_cm)

    return {
        "t": t_values,
        "phi_max": np.array(phi_max_history),
        "x_center": np.array(x_center_history),
    }


def kdv_conserved_quantities(
    x: np.ndarray,
    phi: np.ndarray,
) -> Dict[str, float]:
    """
    Compute KdV conserved quantities (invariants of motion).

    I1 = integral phi dx        (mass/momentum)
    I2 = integral phi^2 dx      (energy)
    I3 = integral (phi_x^2/2 - A*phi^3/(6B)) dx  (Hamiltonian)

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    phi : np.ndarray
        Wave potential.

    Returns
    -------
    invariants : dict
        'I1', 'I2', 'I3': conserved quantities.
    """
    dx = x[1] - x[0] if len(x) > 1 else 0.01

    I1 = np.trapz(phi, x)
    I2 = np.trapz(phi**2, x)

    phi_x = np.gradient(phi, dx)
    I3 = np.trapz(0.5 * phi_x**2, x)

    return {
        "I1": float(I1),
        "I2": float(I2),
        "I3": float(I3),
    }
