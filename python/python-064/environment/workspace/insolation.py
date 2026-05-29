"""
Insolation Computation Module
=============================
Computes top-of-atmosphere solar radiation distribution on Earth,
including seasonal and latitudinal variations driven by orbital forcing.

Key formulas:
-------------
1. Solar constant variation: S0(t) = S0 * (1 + 0.1 * sin(2*pi*t/11))  # 11-year cycle
2. Orbital forcing function: F_orb = e * sin(omega) + epsilon/epsilon_0
3. Meridional heat transport: H(phi) = -2*pi*R^2 * D(phi) * dT/dphi
4. Energy balance at surface: (1 - alpha) * Q = sigma * T^4 + C_p * dT/dt + div(H)
"""

import numpy as np
from milankovitch_orbits import (
    compute_orbital_elements, daily_insolation, annual_mean_insolation,
    S0, EPSILON_0
)

# Physical constants
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant W/m^2/K^4
EARTH_RADIUS = 6.371e6  # m
C_OCEAN = 4.2e6  # J/m^2/K, ocean heat capacity per unit area
C_LAND = 1.0e6  # J/m^2/K, land heat capacity per unit area
ALBEDO_ICE = 0.62  # Ice albedo
ALBEDO_OCEAN = 0.25  # Ocean albedo
ALBEDO_LAND = 0.3  # Land albedo
T_FREEZE = 273.15 - 10.0  # 263.15 K, critical temperature for ice formation


def orbital_forcing_index(time_kyr):
    """
    Compute a composite orbital forcing index combining all three
    Milankovitch parameters.

    Formula:
    F_orb(t) = w1 * e(t)/e_max + w2 * (epsilon(t) - epsilon_min)/(epsilon_max - epsilon_min)
               + w3 * (esin(omega) + 0.07) / 0.14

    Parameters
    ----------
    time_kyr : float or ndarray
        Time in kyr before present.

    Returns
    -------
    float or ndarray
        Normalized orbital forcing index in [0, 1].
    """
    e, eps, prec = compute_orbital_elements(time_kyr)
    e_norm = np.clip(e / 0.06, 0.0, 1.0)
    eps_norm = np.clip((eps - 0.38) / (0.45 - 0.38), 0.0, 1.0)
    prec_norm = np.clip((prec + 0.07) / 0.14, 0.0, 1.0)
    # Weighted combination emphasizing precession and obliquity
    F = 0.2 * e_norm + 0.4 * eps_norm + 0.4 * prec_norm
    return float(F) if np.isscalar(time_kyr) else F


def seasonality_index(time_kyr, latitudes_deg):
    """
    Compute seasonality index: ratio of summer insolation to annual mean.
    High seasonality favors ice sheet growth in cold summers.

    Parameters
    ----------
    time_kyr : float
        Time in kyr.
    latitudes_deg : ndarray
        Latitudes.

    Returns
    -------
    ndarray
        Seasonality index per latitude.
    """
    e, eps, prec = compute_orbital_elements(time_kyr)
    nlat = len(latitudes_deg)
    si = np.zeros(nlat)

    days = np.arange(0, 365, 5)
    for i, lat in enumerate(latitudes_deg):
        daily = daily_insolation(lat, days, e, eps, prec)
        annual = np.mean(daily)
        summer = np.max(daily)
        si[i] = summer / max(annual, 1.0)

    return si


def albedo_feedback(temperature_k, ice_line_lat=None):
    """
    Temperature-dependent albedo with ice-albedo feedback.

    Formula (Budyko-Sellers type):
    alpha(T) = alpha_ice + (alpha_ocean - alpha_ice) / (1 + exp((T - T_c)/w))
    where T_c = T_FREEZE, w = 5 K (transition width).

    Parameters
    ----------
    temperature_k : float or ndarray
        Surface temperature in Kelvin.
    ice_line_lat : float, optional
        Ice line latitude for more complex parameterization.

    Returns
    -------
    float or ndarray
        Albedo value.
    """
    # TODO: Implement Budyko-Sellers albedo feedback formula.
    # Hint: alpha(T) should transition smoothly from ALBEDO_ICE to ALBEDO_OCEAN
    # around T_FREEZE. Use the sigmoid or piecewise-linear model.
    # Must be consistent with the hardcoded values in ebm_fem_solver.py.
    raise NotImplementedError("Hole_1: albedo_feedback is not implemented.")


def outgoing_longwave_radiation(temperature_k):
    """
    Outgoing longwave radiation with greenhouse feedback.

    Formula (linearized):
    OLR(T) = A + B * T
    where A = 203.3 W/m^2, B = 2.09 W/m^2/K (Budyko 1969).
    With CO2 feedback: B_eff = B * (1 - gamma * log(CO2/C0))

    Parameters
    ----------
    temperature_k : float or ndarray
        Surface temperature in Kelvin.

    Returns
    -------
    float or ndarray
        OLR in W/m^2.
    """
    T = np.asarray(temperature_k, dtype=float)
    A = 203.3
    B = 2.09
    # Add weak greenhouse feedback
    gamma = 0.05
    co2_ratio = 1.0 + 0.5 * np.sin(T / 300.0 * np.pi)  # proxy
    B_eff = B * (1.0 - gamma * np.log(max(co2_ratio, 0.1)))
    B_eff = max(B_eff, 0.5)
    olr = A + B_eff * T
    return float(olr) if np.isscalar(temperature_k) else olr


def heat_transport_diffusion(latitudes_deg, temperature_k, D=0.3):
    """
    Meridional heat transport by atmospheric and oceanic diffusion.

    Formula:
    H(phi) = -2 * pi * R^2 * D * cos(phi) * dT/dphi
    where D is diffusivity in W/m^2/K.

    Parameters
    ----------
    latitudes_deg : ndarray
        Latitudes in degrees.
    temperature_k : ndarray
        Temperature at each latitude.
    D : float
        Diffusivity parameter.

    Returns
    -------
    ndarray
        Convergence of heat transport (W/m^2).
    """
    phi = np.deg2rad(latitudes_deg)
    n = len(phi)
    if n < 3:
        return np.zeros(n)

    # Compute dT/dphi using central differences
    dT_dphi = np.zeros(n)
    dT_dphi[0] = (temperature_k[1] - temperature_k[0]) / (phi[1] - phi[0])
    dT_dphi[-1] = (temperature_k[-1] - temperature_k[-2]) / (phi[-1] - phi[-2])
    for i in range(1, n - 1):
        dT_dphi[i] = (temperature_k[i + 1] - temperature_k[i - 1]) / (phi[i + 1] - phi[i - 1])

    # Heat transport convergence: div(H) = -1/(R^2 cos(phi)) * d/dphi(cos(phi) * H)
    H_flux = -2.0 * np.pi * (EARTH_RADIUS ** 2) * D * np.cos(phi) * dT_dphi

    # Divergence
    div_H = np.zeros(n)
    cos_phi = np.cos(phi)
    for i in range(1, n - 1):
        div_H[i] = -1.0 / ((EARTH_RADIUS ** 2) * cos_phi[i]) * \
                   ((H_flux[i + 1] * cos_phi[i + 1] - H_flux[i - 1] * cos_phi[i - 1]) /
                    (phi[i + 1] - phi[i - 1]))

    # Boundary conditions: no heat transport across poles
    div_H[0] = div_H[1]
    div_H[-1] = div_H[-2]

    return div_H


def energy_balance_residual(temperature_k, insolation, latitudes_deg, D=0.3):
    """
    Compute energy balance residual for zero-dimensional EBM.

    Formula:
    R(T) = (1 - alpha(T)) * Q - OLR(T) - div(H)

    Parameters
    ----------
    temperature_k : ndarray
        Temperature field.
    insolation : ndarray
        Insolation at each latitude.
    latitudes_deg : ndarray
        Latitudes.
    D : float
        Diffusivity.

    Returns
    -------
    ndarray
        Residual in W/m^2.
    """
    alpha = albedo_feedback(temperature_k)
    absorbed = (1.0 - alpha) * insolation
    olr = outgoing_longwave_radiation(temperature_k)
    heat_div = heat_transport_diffusion(latitudes_deg, temperature_k, D)
    residual = absorbed - olr - heat_div
    return residual


def equilibrium_temperature(insolation, latitudes_deg, T_guess=None, D=0.3, tol=1e-3, max_iter=100):
    """
    Solve for equilibrium temperature using fixed-point iteration.

    Parameters
    ----------
    insolation : ndarray
        Insolation array.
    latitudes_deg : ndarray
        Latitudes.
    T_guess : ndarray, optional
        Initial temperature guess.
    D : float
        Diffusivity.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    ndarray
        Equilibrium temperature in Kelvin.
    """
    n = len(latitudes_deg)
    if T_guess is None:
        T = 280.0 * np.ones(n)
    else:
        T = np.array(T_guess, dtype=float)

    for iteration in range(max_iter):
        alpha = albedo_feedback(T)
        absorbed = (1.0 - alpha) * insolation
        olr = outgoing_longwave_radiation(T)
        heat_div = heat_transport_diffusion(latitudes_deg, T, D)

        # Update: C * dT/dt = absorbed - olr - heat_div = 0 at equilibrium
        # Use under-relaxation for stability
        omega = 0.3
        new_T = T + omega * (absorbed - olr - heat_div) / 50.0
        new_T = np.clip(new_T, 200.0, 350.0)

        if np.max(np.abs(new_T - T)) < tol:
            return new_T
        T = new_T

    return T
