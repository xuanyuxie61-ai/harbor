
import numpy as np
from typing import Callable, Tuple

from ice_constitutive_model import (
    ICE_DENSITY, SPECIFIC_HEAT, THERMAL_CONDUCTIVITY, LATENT_HEAT, GLEN_N, GRAVITY
)


def build_thermal_diffusion_matrix(nz: int, dz: float,
                                   thermal_diffusivity: float) -> np.ndarray:
    if nz < 3:
        raise ValueError("nz must be at least 3 for finite difference.")
    if dz <= 0:
        raise ValueError("dz must be positive.")

    coef = thermal_diffusivity / (dz ** 2)
    L = np.zeros((nz, nz), dtype=np.float64)


    for i in range(1, nz - 1):
        L[i, i - 1] = coef
        L[i, i] = -2.0 * coef
        L[i, i + 1] = coef


    L[0, 0] = -coef
    L[0, 1] = coef
    L[-1, -2] = coef
    L[-1, -1] = -coef

    return L


def implicit_midpoint_step(y_old: np.ndarray,
                           dt: float,
                           f: Callable[[np.ndarray], np.ndarray],
                           it_max: int = 20,
                           tol: float = 1e-10) -> np.ndarray:
    y_old = np.asarray(y_old, dtype=np.float64)
    y_star = y_old.copy()

    for it in range(it_max):
        y_star_new = y_old + 0.5 * dt * f(y_star)
        diff = np.linalg.norm(y_star_new - y_star, ord=np.inf)
        y_star = y_star_new
        if diff < tol:
            break

    y_new = y_old + dt * f(y_star)
    return y_new


def solve_temperature_evolution(nz: int, z_max: float,
                                dt: float, nt: int,
                                surface_temperature: float,
                                basal_heat_flux: float,
                                velocity_vertical: np.ndarray,
                                dissipation: np.ndarray) -> np.ndarray:
    if nz < 3:
        raise ValueError("nz must be >= 3")
    if z_max <= 0 or dt <= 0 or nt < 0:
        raise ValueError("z_max, dt must be positive; nt >= 0")

    dz = z_max / (nz - 1)
    alpha = THERMAL_CONDUCTIVITY / (ICE_DENSITY * SPECIFIC_HEAT)


    T = np.linspace(surface_temperature,
                    surface_temperature + basal_heat_flux * z_max / THERMAL_CONDUCTIVITY,
                    nz)

    pressure_melting_point = 273.15 - 7.42e-8 * ICE_DENSITY * GRAVITY * np.linspace(0, z_max, nz)
    T = np.minimum(T, pressure_melting_point - 0.1)


    w = np.asarray(velocity_vertical, dtype=np.float64)
    phi = np.asarray(dissipation, dtype=np.float64)
    if w.shape != (nz,):
        raise ValueError(f"velocity_vertical must have shape ({nz},), got {w.shape}")
    if phi.shape != (nz,):
        raise ValueError(f"dissipation must have shape ({nz},), got {phi.shape}")

    T_history = np.zeros((nt + 1, nz), dtype=np.float64)
    T_history[0] = T


    L = build_thermal_diffusion_matrix(nz, dz, alpha)


    C = np.zeros((nz, nz), dtype=np.float64)
    for i in range(1, nz - 1):

        if w[i] >= 0:
            C[i, i] = w[i] / dz
            C[i, i - 1] = -w[i] / dz
        else:
            C[i, i] = -w[i] / dz
            C[i, i + 1] = w[i] / dz


    source = phi / (ICE_DENSITY * SPECIFIC_HEAT)

    def rhs(T_vec: np.ndarray) -> np.ndarray:
        dTdt = L @ T_vec - C @ T_vec + source

        dTdt[0] = 0.0

        return dTdt

    for n in range(nt):
        T = implicit_midpoint_step(T, dt, rhs, it_max=30, tol=1e-12)


        T = np.clip(T, 200.0, 273.15)
        T[0] = surface_temperature
        T_history[n + 1] = T

    return T_history


def solve_enthalpy_evolution(nz: int, z_max: float,
                             dt: float, nt: int,
                             surface_temperature: float,
                             basal_heat_flux: float,
                             velocity_vertical: np.ndarray,
                             dissipation: np.ndarray,
                             porosity_initial: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dz = z_max / (nz - 1)
    z = np.linspace(0, z_max, nz)


    pmp = 273.15 - 7.42e-8 * ICE_DENSITY * 9.81 * z
    H_cold = SPECIFIC_HEAT * (pmp - 200.0)


    T_init = np.minimum(np.linspace(surface_temperature, pmp[-1] + 5.0, nz), pmp - 0.1)
    H = SPECIFIC_HEAT * (T_init - 200.0)
    if porosity_initial is not None:
        H = H + np.asarray(porosity_initial) * LATENT_HEAT

    H = np.clip(H, 0.0, None)

    w = np.asarray(velocity_vertical, dtype=np.float64)
    phi = np.asarray(dissipation, dtype=np.float64)

    alpha_star = THERMAL_CONDUCTIVITY / (ICE_DENSITY * SPECIFIC_HEAT)
    L = build_thermal_diffusion_matrix(nz, dz, alpha_star)

    C = np.zeros((nz, nz), dtype=np.float64)
    for i in range(1, nz - 1):
        if w[i] >= 0:
            C[i, i] = w[i] / dz
            C[i, i - 1] = -w[i] / dz
        else:
            C[i, i] = -w[i] / dz
            C[i, i + 1] = w[i] / dz

    source = phi / ICE_DENSITY

    H_history = np.zeros((nt + 1, nz), dtype=np.float64)
    T_history = np.zeros((nt + 1, nz), dtype=np.float64)
    omega_history = np.zeros((nt + 1, nz), dtype=np.float64)

    H_history[0] = H
    T_history[0], omega_history[0] = enthalpy_to_temperature_water(H, pmp)

    def rhs(H_vec: np.ndarray) -> np.ndarray:
        dHdt = L @ H_vec - C @ H_vec + source
        dHdt[0] = 0.0
        return dHdt

    for n in range(nt):
        H = implicit_midpoint_step(H, dt, rhs, it_max=30, tol=1e-12)
        H = np.clip(H, 0.0, 1e7)

        H[0] = SPECIFIC_HEAT * (surface_temperature - 200.0)
        H[0] = max(H[0], 0.0)

        H_history[n + 1] = H
        T_history[n + 1], omega_history[n + 1] = enthalpy_to_temperature_water(H, pmp)

    return H_history, T_history, omega_history


def enthalpy_to_temperature_water(enthalpy: np.ndarray,
                                  pressure_melting_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = np.asarray(enthalpy, dtype=np.float64)
    Tm = np.asarray(pressure_melting_point, dtype=np.float64)

    T_ref = 200.0
    H_cold = SPECIFIC_HEAT * (Tm - T_ref)

    T = np.zeros_like(H)
    omega = np.zeros_like(H)

    cold_mask = H < H_cold
    temperate_mask = ~cold_mask

    T[cold_mask] = T_ref + H[cold_mask] / SPECIFIC_HEAT
    T[temperate_mask] = Tm[temperate_mask]
    omega[temperate_mask] = (H[temperate_mask] - H_cold[temperate_mask]) / LATENT_HEAT


    omega = np.clip(omega, 0.0, 0.1)
    T = np.clip(T, 200.0, 273.15)

    return T, omega
