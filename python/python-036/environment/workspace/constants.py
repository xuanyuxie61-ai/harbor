
import numpy as np






GF = 1.1663787e-5


EV_TO_JOULE = 1.602176634e-19


KM_TO_EV_INV = 5.067730889e9



EARTH_CRUST_NE = 6.02214076e23 * 2.6 * 0.5 / 14.0




CM3_TO_GEV3 = (1.973269804e-14) ** 3
MATTER_POTENTIAL_FACTOR = np.sqrt(2.0) * GF






THETA_12 = np.deg2rad(33.45)
THETA_23 = np.deg2rad(47.7)
THETA_13 = np.deg2rad(8.62)


DELTA_CP = np.deg2rad(234.0)



DELTA_M2_21 = 7.42e-5


DELTA_M2_31 = 2.510e-3
DELTA_M2_31_IH = -2.490e-3





MASS_ELECTRON = 0.5109989461e-3
MASS_MUON = 105.6583745e-3
MASS_TAU = 1776.86e-3





DEFAULT_QUAD_NX = 64
DEFAULT_QUAD_NY = 64
DEFAULT_MC_SAMPLES = 100000
DEFAULT_ODE_STEPS = 2000





EARTH_RADIUS_KM = 6371.0
CORE_RADIUS_KM = 3480.0


DENSITY_CRUST = 2.7

DENSITY_MANTLE = 5.5

DENSITY_OUTER_CORE = 11.0

DENSITY_INNER_CORE = 13.0


def get_prem_density(radius_ratio):
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")

    r_km = radius_ratio * EARTH_RADIUS_KM

    if r_km < 1221.0:

        return 13.0885 - 8.8381 * radius_ratio ** 2
    elif r_km < 3480.0:

        return 12.5815 - 1.2638 * radius_ratio - 3.6426 * radius_ratio ** 2 \
               - 5.5281 * radius_ratio ** 3
    elif r_km < 5701.0:

        return 7.9565 - 6.4761 * radius_ratio + 5.5283 * radius_ratio ** 2 \
               - 3.0807 * radius_ratio ** 3
    elif r_km < 5771.0:

        return 5.3197 - 1.4836 * radius_ratio
    elif r_km < 5971.0:

        return 11.2494 - 8.0298 * radius_ratio
    elif r_km < 6151.0:

        return 7.1089 - 3.8045 * radius_ratio
    else:

        return 2.6910 + 0.6924 * radius_ratio


def electron_fraction(radius_ratio):
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")

    return 0.465 + 0.04 * radius_ratio


def matter_potential_eV(radius_ratio, energy_gev=None):
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")

    rho = get_prem_density(radius_ratio)
    ye = electron_fraction(radius_ratio)







    avg_molar_mass = 20.0
    avogadro = 6.02214076e23
    ne_cm3 = rho * avogadro * ye / avg_molar_mass




    hbarc = 0.1973269804
    hbarc_cm = hbarc * 1e-13
    ne_gev3 = ne_cm3 * hbarc_cm ** 3


    v_gev = np.sqrt(2.0) * GF * ne_gev3


    v_ev = v_gev * 1e9

    return v_ev


def get_mass_squared_differences(hierarchy='normal'):
    hierarchy = hierarchy.lower()
    if hierarchy == 'normal':
        return np.array([DELTA_M2_21, DELTA_M2_31], dtype=np.float64)
    elif hierarchy == 'inverted':
        return np.array([DELTA_M2_21, DELTA_M2_31_IH], dtype=np.float64)
    else:
        raise ValueError("hierarchy must be 'normal' or 'inverted'")


def get_pmns_angles():
    return np.array([THETA_12, THETA_23, THETA_13], dtype=np.float64)


def get_cp_phase():
    return float(DELTA_CP)
