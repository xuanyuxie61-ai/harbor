"""
状态方程模块：Wuppertal-Budapest 格点 QCD 参数化
Equation of state: lattice-QCD inspired parameterization
(Wuppertal-Budapest collaboration, 2+1 flavor).

Energy density ε(T), pressure P(T), entropy density s(T),
sound speed c_s²(T), and their inverses.
"""
import numpy as np
from physics_constants import TC_QCD, stefan_boltzmann_constant_nb_flavors


# Wuppertal-Budapest fit parameters (2+1 flavor, physical quark masses)
# Parameterization: ε(T)/T^4, P(T)/T^4 fitted to lattice data
WB_A1 = 4.064
WB_A2 = 18.36
WB_T0 = 0.099  # GeV
WB_S1 = 3.978
WB_S2 = 0.8706
WB_S3 = 0.3308
WB_S4 = 0.0


def pressure_lattice(temperature):
    """
    Lattice-QCD pressure parameterization (Wuppertal-Budapest):
        P(T)/T^4 = A₁ exp(-A₂ T/T₀) + S₁ tanh((T - T₀)/S₂) + S₃

    For T → ∞: approaches Stefan-Boltzmann limit.
    For T → 0: exponentially suppressed.

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        Pressure P in GeV/fm³ (natural units)
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    x = T / WB_T0

    # Exponential + tanh parameterization
    p_over_t4 = (WB_A1 * np.exp(-WB_A2 * x)
                 + WB_S1 * np.tanh((T - WB_T0) / WB_S2)
                 + WB_S3)

    # Smooth match to SB limit at high T
    sigma_sb = stefan_boltzmann_constant_nb_flavors(3)
    p_ideal = sigma_sb * T**4 / 3.0
    p_latt = p_over_t4 * T**4

    # Blend: at very high T, approach ideal
    blend = np.exp(-((T - 2.0 * TC_QCD) / (0.5 * TC_QCD))**2)
    return np.where(T > 3.0 * TC_QCD, p_ideal,
                   p_latt * (1.0 - blend) + p_ideal * blend)


def energy_density_lattice(temperature):
    """
    Energy density from thermodynamic relation:
        ε = T dP/dT - P

    Computed via numerical differentiation of P(T).

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        Energy density ε in GeV/fm³
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    dT = 0.001 * TC_QCD  # step size for differentiation

    p_plus = pressure_lattice(T + dT)
    p_minus = pressure_lattice(T - dT)
    p_0 = pressure_lattice(T)

    dp_dt = (p_plus - p_minus) / (2.0 * dT)
    return T * dp_dt - p_0


def entropy_density_lattice(temperature):
    """
    Entropy density from thermodynamic relation:
        s = dP/dT

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        Entropy density s in fm^{-3}
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    dT = 0.001 * TC_QCD
    p_plus = pressure_lattice(T + dT)
    p_minus = pressure_lattice(T - dT)
    return (p_plus - p_minus) / (2.0 * dT)


def sound_speed_squared(temperature):
    """
    Speed of sound squared: c_s² = dP/dε = (dP/dT) / (dε/dT)

    For ideal gas: c_s² = 1/3
    Near Tc: dips to ~0.1 due to crossover softening

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        c_s² (dimensionless, in [0, 1/3])
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    dT = 0.001 * TC_QCD

    p_plus = pressure_lattice(T + dT)
    p_minus = pressure_lattice(T - dT)
    dp_dt = (p_plus - p_minus) / (2.0 * dT)

    eps_plus = energy_density_lattice(T + dT)
    eps_minus = energy_density_lattice(T - dT)
    deps_dt = (eps_plus - eps_minus) / (2.0 * dT)

    deps_dt = np.maximum(deps_dt, 1e-14)
    cs2 = dp_dt / deps_dt
    # Bound c_s² to physical range
    return np.clip(cs2, 0.0, 1.0 / 3.0 + 0.05)


def temperature_from_energy_density(eps):
    """
    Invert ε(T) → T using bisection.
    For each energy density value, find T such that ε(T) = eps.

    Parameters
    ----------
    eps : float or ndarray
        Energy density in GeV/fm³

    Returns
    -------
    float or ndarray
        Temperature T in GeV
    """
    eps = np.asarray(eps, dtype=float)
    scalar_input = (eps.ndim == 0)
    eps = np.atleast_1d(eps)

    T_result = np.zeros_like(eps)

    for idx, eps_val in np.ndenumerate(eps):
        if eps_val <= 0:
            T_result[idx] = 0.0
            continue

        # Bisection in [0.05, 2.0] GeV
        T_lo, T_hi = 0.05, 2.0
        for _ in range(60):
            T_mid = 0.5 * (T_lo + T_hi)
            eps_mid = energy_density_lattice(T_mid)
            if eps_mid < eps_val:
                T_lo = T_mid
            else:
                T_hi = T_mid

        T_result[idx] = 0.5 * (T_lo + T_hi)

    if scalar_input:
        return float(T_result[0])
    return T_result


def shear_viscosity_eta_over_s(temperature):
    """
    Temperature-dependent shear viscosity η/s.

    Parameterization (minimal near Tc):
        η/s(T) = (η/s)_min + a * ((T - Tc) / Tc)^2 for T > Tc
        η/s(T) = (η/s)_min for T ≈ Tc
        η/s(T) = (η/s)_min + b * ((Tc - T) / Tc)^2 for T < Tc

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        η/s (dimensionless, ≥ 1/(4π))
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    eta_s_min = 1.0 / (4.0 * np.pi)  # KSS bound
    a_coeff = 0.3
    b_coeff = 0.15

    x = (T - TC_QCD) / TC_QCD
    eta_s = eta_s_min + np.where(x >= 0, a_coeff * x**2, b_coeff * x**2)
    return np.maximum(eta_s, eta_s_min)


def bulk_viscosity_zeta_over_s(temperature):
    """
    Bulk viscosity ζ/s parameterization.

    Peak near Tc (from lattice), Gaussian shape:
        ζ/s(T) = ζ_max exp(-((T - Tc) / σ)²)

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        ζ/s (dimensionless)
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    zeta_max = 0.04
    sigma_width = 0.05  # GeV
    return zeta_max * np.exp(-((T - TC_QCD) / sigma_width)**2)


def relaxation_time_tau_pi(temperature, eta_over_s):
    """
    Israel-Stewart relaxation time for shear stress:
        τ_π = 5 η / (ε + P) = 5 (η/s) / s

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV
    eta_over_s : float or ndarray
        Shear viscosity ratio

    Returns
    -------
    float or ndarray
        Relaxation time τ_π in fm/c
    """
    s = entropy_density_lattice(temperature)
    s = np.maximum(s, 1e-14)
    tau = 5.0 * eta_over_s / s

    # Convert from GeV^{-1} to fm: multiply by ħc
    return tau * 0.1973


def interaction_measure_I_T(temperature):
    """
    Interaction measure (trace anomaly):
        I(T) = (ε - 3P) / T^4

    Zero for ideal gas; peaks near Tc for interacting QGP.

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        (ε - 3P)/T^4 in GeV^{-4}
    """
    T = np.maximum(np.asarray(temperature, dtype=float), 1e-10)
    eps = energy_density_lattice(T)
    p = pressure_lattice(T)
    return (eps - 3.0 * p) / T**4


def enthalpy_density(temperature):
    """
    Enthalpy density w = ε + P

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in GeV

    Returns
    -------
    float or ndarray
        Enthalpy density in GeV/fm³
    """
    eps = energy_density_lattice(temperature)
    p = pressure_lattice(temperature)
    return eps + p
