
import numpy as np


def get_membrane_parameters():
    params = {

        "membrane_thickness": 1.5e-7,
        "fiber_inner_radius": 1.0e-4,
        "fiber_outer_radius": 2.0e-4,
        "module_length": 0.5,
        "porosity": 0.35,
        "tortuosity": 2.8,


        "temperature": 308.15,
        "pressure_feed": 5.0e6,
        "pressure_permeate": 1.0e5,
        "R_gas": 8.314,


        "D_co2": 2.5e-10,
        "D_ch4": 8.0e-11,
        "D_co2_support": 1.2e-6,
        "D_ch4_support": 9.5e-7,


        "S_co2": 1.8e-4,
        "S_ch4": 3.5e-5,


        "k_reaction": 4.2e-3,
        "K_ads_co2": 1.2e-3,
        "K_ads_ch4": 2.5e-4,


        "Nx": 128,
        "Nt": 2000,
        "t_final": 3600.0,


        "pore_count": 256,
        "pore_mean_radius": 5.0e-9,
        "pore_std_radius": 1.5e-9,


        "stages": 4,
        "stage_cut_nominal": 0.3,
    }
    return params


def get_feed_composition():
    return {
        "CO2": 0.15,
        "CH4": 0.80,
        "N2": 0.05,
    }


def compute_permeability(params):
    P_co2 = params["D_co2"] * params["S_co2"]
    P_ch4 = params["D_ch4"] * params["S_ch4"]
    return {"CO2": P_co2, "CH4": P_ch4}


def compute_sorption_heat():
    return {"CO2": -24.0e3, "CH4": -18.5e3}


def get_critical_properties():
    return {
        "CO2": {"Tc": 304.13, "Pc": 7.377e6, "omega": 0.225},
        "CH4": {"Tc": 190.56, "Pc": 4.599e6, "omega": 0.011},
        "N2":  {"Tc": 126.19, "Pc": 3.396e6, "omega": 0.037},
    }


def validate_parameters(params):
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
    D = params["D_co2"] if species == "CO2" else params["D_ch4"]
    S = params["S_co2"] if species == "CO2" else params["S_ch4"]
    L = params["membrane_thickness"]
    k = params["k_reaction"]



    k_c = D / L
    Bi_m = k_c * L / D


    Da = k * (L ** 2) / D


    phi = L * np.sqrt(k / D)

    return {
        "Biot_mass": Bi_m,
        "Damkohler": Da,
        "Thiele_modulus": phi,
    }
