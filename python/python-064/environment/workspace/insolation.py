
import numpy as np
from milankovitch_orbits import (
    compute_orbital_elements, daily_insolation, annual_mean_insolation,
    S0, EPSILON_0
)


SIGMA = 5.670374419e-8
EARTH_RADIUS = 6.371e6
C_OCEAN = 4.2e6
C_LAND = 1.0e6
ALBEDO_ICE = 0.62
ALBEDO_OCEAN = 0.25
ALBEDO_LAND = 0.3
T_FREEZE = 273.15 - 10.0


def orbital_forcing_index(time_kyr):
    e, eps, prec = compute_orbital_elements(time_kyr)
    e_norm = np.clip(e / 0.06, 0.0, 1.0)
    eps_norm = np.clip((eps - 0.38) / (0.45 - 0.38), 0.0, 1.0)
    prec_norm = np.clip((prec + 0.07) / 0.14, 0.0, 1.0)

    F = 0.2 * e_norm + 0.4 * eps_norm + 0.4 * prec_norm
    return float(F) if np.isscalar(time_kyr) else F


def seasonality_index(time_kyr, latitudes_deg):
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




    raise NotImplementedError("Hole_1: albedo_feedback is not implemented.")


def outgoing_longwave_radiation(temperature_k):
    T = np.asarray(temperature_k, dtype=float)
    A = 203.3
    B = 2.09

    gamma = 0.05
    co2_ratio = 1.0 + 0.5 * np.sin(T / 300.0 * np.pi)
    B_eff = B * (1.0 - gamma * np.log(max(co2_ratio, 0.1)))
    B_eff = max(B_eff, 0.5)
    olr = A + B_eff * T
    return float(olr) if np.isscalar(temperature_k) else olr


def heat_transport_diffusion(latitudes_deg, temperature_k, D=0.3):
    phi = np.deg2rad(latitudes_deg)
    n = len(phi)
    if n < 3:
        return np.zeros(n)


    dT_dphi = np.zeros(n)
    dT_dphi[0] = (temperature_k[1] - temperature_k[0]) / (phi[1] - phi[0])
    dT_dphi[-1] = (temperature_k[-1] - temperature_k[-2]) / (phi[-1] - phi[-2])
    for i in range(1, n - 1):
        dT_dphi[i] = (temperature_k[i + 1] - temperature_k[i - 1]) / (phi[i + 1] - phi[i - 1])


    H_flux = -2.0 * np.pi * (EARTH_RADIUS ** 2) * D * np.cos(phi) * dT_dphi


    div_H = np.zeros(n)
    cos_phi = np.cos(phi)
    for i in range(1, n - 1):
        div_H[i] = -1.0 / ((EARTH_RADIUS ** 2) * cos_phi[i]) * \
                   ((H_flux[i + 1] * cos_phi[i + 1] - H_flux[i - 1] * cos_phi[i - 1]) /
                    (phi[i + 1] - phi[i - 1]))


    div_H[0] = div_H[1]
    div_H[-1] = div_H[-2]

    return div_H


def energy_balance_residual(temperature_k, insolation, latitudes_deg, D=0.3):
    alpha = albedo_feedback(temperature_k)
    absorbed = (1.0 - alpha) * insolation
    olr = outgoing_longwave_radiation(temperature_k)
    heat_div = heat_transport_diffusion(latitudes_deg, temperature_k, D)
    residual = absorbed - olr - heat_div
    return residual


def equilibrium_temperature(insolation, latitudes_deg, T_guess=None, D=0.3, tol=1e-3, max_iter=100):
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



        omega = 0.3
        new_T = T + omega * (absorbed - olr - heat_div) / 50.0
        new_T = np.clip(new_T, 200.0, 350.0)

        if np.max(np.abs(new_T - T)) < tol:
            return new_T
        T = new_T

    return T
