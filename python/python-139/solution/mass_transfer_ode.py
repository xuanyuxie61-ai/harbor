"""
Ordinary differential equation systems describing coupled mass transfer.

Adapted from:
  - reaction_ode (surface reaction kinetics)
  - kepler_ode (Hamiltonian / conservative formulation)
  - quasiperiodic_ode (higher-order oscillatory dynamics)
  - runge (function derivatives for stiff interpolation testing)

In the membrane context, these ODEs govern:
  1. Surface reaction-transient species balances
  2. Conservative trajectory of permeating molecules (analogy to Kepler orbits)
  3. Quasiperiodic forcing from oscillatory feed conditions
"""

import numpy as np


def reaction_deriv(t, y, k, K_co2, K_ch4, P_total):
    """
    Right-hand side of surface reaction ODEs for CO2/CH4 mixture.

    Langmuir-Hinshelwood kinetics on the membrane active layer:
        r_i = k * theta_i
        theta_i = K_i * P_i / (1 + sum_j K_j P_j)

    State vector y = [A, B, C] where:
        A = CO2 surface concentration
        B = CH4 surface concentration
        C = product / adsorbed complex
    """
    A = max(y[0], 0.0)
    B = max(y[1], 0.0)
    denom = 1.0 + K_co2 * A + K_ch4 * B
    if denom <= 0.0:
        denom = 1e-30
    theta_co2 = K_co2 * A / denom
    theta_ch4 = K_ch4 * B / denom

    # Reaction rates (first-order in surface coverage)
    r_co2 = k * theta_co2
    r_ch4 = k * theta_ch4

    dAdt = -r_co2
    dBdt = -r_ch4
    dCdt = r_co2 + r_ch4
    return np.array([dAdt, dBdt, dCdt], dtype=float)


def reaction_parameters():
    """
    Default parameters for surface reaction ODE.
    """
    return {
        "k": 4.2e-3,
        "K_co2": 1.2e-3,
        "K_ch4": 2.5e-4,
        "P_total": 5.0e6,
        "y0": np.array([150.0, 800.0, 0.0], dtype=float),
        "tspan": (0.0, 3600.0),
    }


def kepler_like_trajectory_deriv(t, y, mu=1.0):
    """
    Conservative trajectory equations for a permeating molecule in a cylindrical pore,
    modeled as a central-force (Kepler-like) problem in the radial-axial plane.

    State vector y = [q1, q2, p1, p2]:
        q1 = radial position r
        q2 = axial position z
        p1 = radial momentum
        p2 = axial momentum

    The potential represents a wall-interaction: V(r) = -mu / sqrt(r^2 + epsilon^2).
    """
    eps = 1e-12
    q1 = y[0]
    q2 = y[1]
    p1 = y[2]
    p2 = y[3]
    r2 = q1 * q1 + q2 * q2
    r_eff = np.sqrt(r2 + eps * eps)
    r_eff3 = r_eff ** 3

    dq1dt = p1
    dq2dt = p2
    dp1dt = -mu * q1 / r_eff3
    dp2dt = -mu * q2 / r_eff3
    return np.array([dq1dt, dq2dt, dp1dt, dp2dt], dtype=float)


def kepler_parameters():
    """
    Default parameters for Kepler-like pore trajectory.
    """
    return {
        "mu": 1.0e-20,
        "y0": np.array([1e-9, 0.0, 0.0, 1e-4], dtype=float),
        "tspan": (0.0, 1e-6),
    }


def quasiperiodic_forcing_deriv(t, y, omega1=np.pi, omega2=1.0):
    """
    Fourth-order quasiperiodic ODE modeling oscillatory feed-composition fluctuations.

    d^4 y / dt^4 + (omega1^2 + 1) d^2 y / dt^2 + omega1^2 y = 0

    State vector y = [u, du/dt, d^2u/dt^2, d^3u/dt^3].
    This governs the perturbation of feed concentration around a steady state.
    """
    dydt = np.zeros(4, dtype=float)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = y[3]
    dydt[3] = -(omega1 ** 2 + 1.0) * y[2] - (omega1 ** 2) * y[0]
    return dydt


def quasiperiodic_parameters():
    return {
        "omega1": np.pi,
        "omega2": 1.0,
        "y0": np.array([0.01, 0.0, -0.01 * np.pi ** 2, 0.0], dtype=float),
        "tspan": (0.0, 10.0),
    }


def runge_function(x):
    """
    Classic Runge function f(x) = 1 / (1 + 25 x^2).
    Used to test polynomial interpolation of steep concentration fronts.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_derivative(x):
    """
    Analytical derivative of Runge function:
        f'(x) = -50 x / (1 + 25 x^2)^2
    """
    x = np.asarray(x, dtype=float)
    denom = (1.0 + 25.0 * x * x) ** 2
    return -50.0 * x / denom


def runge_second_derivative(x):
    """
    Second derivative for curvature-based mesh adaptation.
    """
    x = np.asarray(x, dtype=float)
    x2 = x * x
    num = 50.0 * (75.0 * x2 - 1.0)
    denom = (1.0 + 25.0 * x2) ** 3
    return num / denom


def power_series_runge(x, n_terms=10):
    """
    Taylor power series expansion of Runge function about x=0.
    f(x) = sum_{k=0}^infty (-1)^k 5^{2k} x^{2k}
    """
    x = np.asarray(x, dtype=float)
    val = np.zeros_like(x, dtype=float)
    for k in range(n_terms):
        coeff = (-1.0) ** k * (5.0 ** (2 * k))
        val += coeff * (x ** (2 * k))
    return val


def coupled_membrane_reaction_ode(t, y, params):
    """
    Coupled ODE for membrane surface + bulk reaction with quasiperiodic forcing.

    y = [c_surf_co2, c_surf_ch4, c_prod, c_bulk_co2, c_bulk_ch4, u, ud, udd, uddd]
    where [u, ud, udd, uddd] is the quasiperiodic forcing state.
    """
    k = params["k_reaction"]
    K_co2 = params["K_ads_co2"]
    K_ch4 = params["K_ads_ch4"]
    omega1 = params.get("omega1", np.pi)
    h_mt = params.get("h_mt", 1e-4)  # mass transfer coefficient

    # Surface reaction block
    surf = y[:3]
    dsurf = reaction_deriv(t, surf, k, K_co2, K_ch4, 1.0)

    # Bulk depletion due to surface reaction
    c_bulk_co2 = max(y[3], 0.0)
    c_bulk_ch4 = max(y[4], 0.0)
    dc_bulk_co2 = -h_mt * (c_bulk_co2 - max(surf[0], 0.0))
    dc_bulk_ch4 = -h_mt * (c_bulk_ch4 - max(surf[1], 0.0))

    # Quasiperiodic forcing block
    qp = y[5:9]
    dqp = quasiperiodic_forcing_deriv(t, qp, omega1)

    return np.concatenate([dsurf, [dc_bulk_co2, dc_bulk_ch4], dqp])
