"""
Physical and operational parameters for the membrane separation mass transfer model.

This module consolidates all thermodynamic, kinetic, and geometric parameters
required for the multi-scale simulation of gas separation in asymmetric hollow-fiber
membranes. All units are SI unless explicitly noted.
"""

import numpy as np


def get_membrane_parameters():
    """
    Return a dictionary of physical parameters for the membrane module.

    The default configuration models a polyimide hollow-fiber membrane
    for CO2/CH4 separation, a representative system in chemical engineering.
    """
    params = {
        # --- Geometric parameters ---
        "membrane_thickness": 1.5e-7,      # m (active layer thickness)
        "fiber_inner_radius": 1.0e-4,      # m
        "fiber_outer_radius": 2.0e-4,      # m
        "module_length": 0.5,              # m
        "porosity": 0.35,                  # void fraction of support layer
        "tortuosity": 2.8,                 # empirical tortuosity factor

        # --- Thermodynamic and transport properties ---
        "temperature": 308.15,             # K (35 C)
        "pressure_feed": 5.0e6,            # Pa (50 bar)
        "pressure_permeate": 1.0e5,        # Pa (1 bar)
        "R_gas": 8.314,                    # J/(mol*K)

        # --- Diffusion coefficients (m^2/s) ---
        "D_co2": 2.5e-10,                  # Fickian diffusion in polymer
        "D_ch4": 8.0e-11,
        "D_co2_support": 1.2e-6,           # Knudsen diffusion in porous support
        "D_ch4_support": 9.5e-7,

        # --- Solubility coefficients (mol/(m^3*Pa)) ---
        "S_co2": 1.8e-4,
        "S_ch4": 3.5e-5,

        # --- Surface reaction parameters (Langmuir-Hinshelwood) ---
        "k_reaction": 4.2e-3,              # 1/s (first-order surface reaction rate)
        "K_ads_co2": 1.2e-3,               # 1/Pa
        "K_ads_ch4": 2.5e-4,               # 1/Pa

        # --- Numerical discretization ---
        "Nx": 128,                         # FEM nodes across membrane thickness
        "Nt": 2000,                        # time steps for transient analysis
        "t_final": 3600.0,                 # s (simulation horizon)

        # --- Pore network model ---
        "pore_count": 256,                 # number of representative pores
        "pore_mean_radius": 5.0e-9,        # m
        "pore_std_radius": 1.5e-9,         # m

        # --- Cascade / network parameters ---
        "stages": 4,                       # number of membrane stages in cascade
        "stage_cut_nominal": 0.3,          # nominal stage cut
    }
    return params


def get_feed_composition():
    """
    Return feed-side mole fractions for a typical natural-gas stream.
    """
    return {
        "CO2": 0.15,
        "CH4": 0.80,
        "N2": 0.05,
    }


def compute_permeability(params):
    """
    Compute the permeability P_i = D_i * S_i for each species.

    Permeability is given in Barrer units for reporting, but returned
    here in SI (mol*m/(m^2*s*Pa)).
    """
    P_co2 = params["D_co2"] * params["S_co2"]
    P_ch4 = params["D_ch4"] * params["S_ch4"]
    return {"CO2": P_co2, "CH4": P_ch4}


def compute_sorption_heat():
    """
    Return representative sorption enthalpies (J/mol) for CO2 and CH4
    in polyimide membranes.  These values are used in the van't Hoff
    correction for temperature-dependent solubility.
    """
    return {"CO2": -24.0e3, "CH4": -18.5e3}


def get_critical_properties():
    """
    Critical properties (Tc in K, Pc in Pa) for equation-of-state calculations.
    """
    return {
        "CO2": {"Tc": 304.13, "Pc": 7.377e6, "omega": 0.225},
        "CH4": {"Tc": 190.56, "Pc": 4.599e6, "omega": 0.011},
        "N2":  {"Tc": 126.19, "Pc": 3.396e6, "omega": 0.037},
    }


def validate_parameters(params):
    """
    Validate parameter dictionary for physical consistency and numerical safety.
    Raises ValueError on inconsistency.
    """
    if params["membrane_thickness"] <= 0:
        raise ValueError("membrane_thickness must be positive.")
    if params["fiber_inner_radius"] <= 0:
        raise ValueError("fiber_inner_radius must be positive.")
    if params["fiber_outer_radius"] <= params["fiber_inner_radius"]:
        raise ValueError("fiber_outer_radius must exceed fiber_inner_radius.")
    if not (0.0 < params["porosity"] < 1.0):
        raise ValueError("porosity must lie in (0,1).")
    if params["tortuosity"] < 1.0:
        raise ValueError("tortuosity must be >= 1.")
    if params["temperature"] <= 0:
        raise ValueError("temperature must be positive (Kelvin).")
    if params["pressure_feed"] <= params["pressure_permeate"]:
        raise ValueError("feed pressure must exceed permeate pressure.")
    if params["Nx"] < 2:
        raise ValueError("Nx must be at least 2.")
    if params["Nt"] < 1:
        raise ValueError("Nt must be positive.")
    if params["t_final"] <= 0:
        raise ValueError("t_final must be positive.")


def compute_dimensionless_numbers(params, species="CO2"):
    """
    Compute key dimensionless groups for membrane mass transfer.

    Returns:
        dict with Biot, Damkohler, and Thiele modulus numbers.
    """
    D = params["D_co2"] if species == "CO2" else params["D_ch4"]
    S = params["S_co2"] if species == "CO2" else params["S_ch4"]
    L = params["membrane_thickness"]
    k = params["k_reaction"]

    # Biot number for mass transfer: Bi_m = k_c * L / D
    # Here k_c is approximated by D / L (film theory)
    k_c = D / L
    Bi_m = k_c * L / D  # => 1.0 in film theory, generalization kept for form

    # Damkohler number: Da = k * L^2 / D
    Da = k * (L ** 2) / D

    # Thiele modulus: phi = L * sqrt(k / D)
    phi = L * np.sqrt(k / D)

    return {
        "Biot_mass": Bi_m,
        "Damkohler": Da,
        "Thiele_modulus": phi,
    }
