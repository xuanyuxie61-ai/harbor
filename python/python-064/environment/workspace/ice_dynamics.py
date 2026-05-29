"""
Ice Sheet Dynamics Module
=========================
Models the evolution of continental ice sheet volume using coupled
ODE systems that combine glaciological mass balance with nonlinear
climate feedbacks.

Incorporates:
- Anishchenko chaotic oscillator (006_anishchenko_ode) for internal climate variability
- Nonlinear ODE integration with adaptive time-stepping

Scientific Background:
----------------------
Ice sheet volume V(t) evolves according to:
    dV/dt = B_m(V, T) - D(V)
where B_m is mass balance (accumulation minus ablation) and D is discharge.

Mass balance follows the positive-degree-day (PDD) model:
    B_m = max(0, P - mu * max(0, T - T_m))
where P is precipitation, mu is melt factor, T_m is melting threshold.

Coupled with Anishchenko-type nonlinear oscillator for climate noise:
    dx/dt = mu * x + y - x * z
    dy/dt = -x
    dz/dt = -eta * z + eta * H(x) * x^2
where H(x) is Heaviside step function.

The coupled system introduces chaotic sensitivity to orbital forcing,
mimicking the stochastic resonance observed in paleoclimate records.
"""

import numpy as np


def heaviside(x):
    """
    Heaviside step function.
    H(x) = 1 if x >= 0, else 0.
    """
    return np.where(x >= 0, 1.0, 0.0)


def anishchenko_deriv(t, xyz, mu=1.2, eta=0.5):
    """
    Right-hand side of Anishchenko oscillator.
    From 006_anishchenko_ode.

    Parameters
    ----------
    t : float
        Time (unused but kept for API consistency).
    xyz : array_like, shape (3,)
        State vector [x, y, z].
    mu : float
        Nonlinearity parameter.
    eta : float
        Damping parameter.

    Returns
    -------
    ndarray
        Derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = xyz
    dxdt = mu * x + y - x * z
    dydt = -x
    dzdt = -eta * z + eta * heaviside(x) * x ** 2
    return np.array([dxdt, dydt, dzdt])


def ice_mass_balance(T_surface, orbital_forcing, P_base=0.3, mu_melt=0.008, T_melt=273.15):
    """
    Compute ice sheet mass balance using positive-degree-day model.

    Formula:
    B_m = max(0, P(T) - mu * max(0, T - T_m))
    P(T) = P_base * (1 + beta_p * (T - T_ref))  # precipitation increases with temperature

    Parameters
    ----------
    T_surface : float
        Surface temperature in K.
    orbital_forcing : float
        Normalized orbital forcing index [0, 1].
    P_base : float
        Base precipitation rate (m/yr water equivalent).
    mu_melt : float
        Degree-day melt factor (m/yr/K).
    T_melt : float
        Melting temperature threshold (K).

    Returns
    -------
    float
        Mass balance in m/yr (positive = accumulation).
    """
    beta_p = 0.05  # precipitation sensitivity to temperature
    T_ref = 273.15
    P = P_base * (1.0 + beta_p * (T_surface - T_ref))
    P = max(P, 0.05)  # Minimum precipitation

    melt = mu_melt * max(0.0, T_surface - T_melt)
    melt = min(melt, P * 2.0)  # Cap melt rate

    # Orbital forcing modulates accumulation season
    accumulation_boost = 1.0 + 0.3 * orbital_forcing
    B_m = P * accumulation_boost - melt
    return B_m


def ice_discharge(volume, V_max=50e6, k_discharge=0.001):
    """
    Ice sheet discharge (calving + basal melting) proportional to volume excess.

    Formula:
    D = k_discharge * max(0, V - V_eq)
    where V_eq decreases with sea level.

    Parameters
    ----------
    volume : float
        Ice volume in km^3.
    V_max : float
        Maximum stable ice volume.
    k_discharge : float
        Discharge rate constant.

    Returns
    -------
    float
        Discharge rate in km^3/yr.
    """
    D = k_discharge * max(0.0, volume - 0.7 * V_max)
    return D


def sea_level(volume, V_full_glacial=50e6, dSL_dV=7.4e-6):
    """
    Compute sea level change from ice volume.

    Formula:
    Delta_SL = -dSL_dV * V
    where dSL_dV ~ 7.4e-6 m/km^3 (for Antarctic + Greenland ice).

    Parameters
    ----------
    volume : float
        Ice volume in km^3.
    V_full_glacial : float
        Reference full glacial volume.
    dSL_dV : float
        Sea level sensitivity to ice volume.

    Returns
    -------
    float
        Sea level change in meters (negative = lower sea level).
    """
    return -dSL_dV * volume


def bedrock_depression(volume, tau_relax=5000.0, rho_ice=917.0, rho_mantle=3300.0):
    """
    Compute isostatic bedrock depression under ice load.

    Formula:
    h_bedrock = - (rho_ice / rho_mantle) * h_ice * (1 - exp(-t/tau))

    Parameters
    ----------
    volume : float
        Ice volume.
    tau_relax : float
        Relaxation time in years.
    rho_ice, rho_mantle : float
        Densities in kg/m^3.

    Returns
    -------
    float
        Bedrock depression in meters.
    """
    # Approximate average ice thickness from volume
    A_ice = 14e6  # km^2, approximate ice-covered area
    h_ice = volume / A_ice * 1e3  # Convert to meters
    depression = -(rho_ice / rho_mantle) * h_ice
    return depression


def coupled_climate_ice_deriv(t, state, orbital_forcing_func, mu=1.2, eta=0.5):
    """
    Coupled climate-ice sheet ODE system.

    State vector:
    [x, y, z, V_ice, T_global, h_bedrock]

    where:
    - (x, y, z): Anishchenko oscillator (climate internal variability)
    - V_ice: Ice volume (km^3)
    - T_global: Global mean temperature (K)
    - h_bedrock: Bedrock depression (m)

    Parameters
    ----------
    t : float
        Time in years.
    state : ndarray, shape (6,)
        State vector.
    orbital_forcing_func : callable
        Function f(t) returning orbital forcing index.
    mu, eta : float
        Anishchenko parameters.

    Returns
    -------
    ndarray
        Derivatives.
    """
    x, y, z, V_ice, T_global, h_bedrock = state

    # Orbital forcing
    F_orb = orbital_forcing_func(t)
    F_orb = np.clip(F_orb, 0.0, 1.0)

    # Climate oscillator
    climate_deriv = anishchenko_deriv(t, [x, y, z], mu, eta)

    # Coupling: orbital + chaotic -> temperature
    # T_eq = T_base + delta_T_orbital + delta_T_chaotic
    T_base = 288.0  # K
    delta_T_orbital = 5.0 * (F_orb - 0.5)  # +/- 2.5 K orbital modulation
    delta_T_chaotic = 1.0 * x  # Chaotic component
    T_target = T_base + delta_T_orbital + delta_T_chaotic

    # Temperature relaxation
    tau_temp = 10.0  # years
    dTdt = (T_target - T_global) / tau_temp
    T_global = max(200.0, min(350.0, T_global))

    # Ice mass balance
    B_m = ice_mass_balance(T_global, F_orb)
    D = ice_discharge(V_ice)

    # Ice volume change (convert m/yr to km^3/yr)
    A_ice_sheet = 14e6  # km^2
    dVdt = (B_m * 1e-3 * A_ice_sheet) - D

    # Bedrock relaxation
    tau_bed = 5000.0
    h_target = bedrock_depression(V_ice)
    dhdt = (h_target - h_bedrock) / tau_bed

    return np.array([
        climate_deriv[0],
        climate_deriv[1],
        climate_deriv[2],
        dVdt,
        dTdt,
        dhdt
    ])


def rk4_step(f, t, y, h, *args):
    """
    Single 4th-order Runge-Kutta step.

    Parameters
    ----------
    f : callable
        Derivative function f(t, y, *args).
    t : float
        Current time.
    y : ndarray
        Current state.
    h : float
        Step size.
    *args : extra arguments for f.

    Returns
    -------
    ndarray
        Next state.
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ice_climate(t_span, y0, orbital_forcing_func, dt=1.0, mu=1.2, eta=0.5):
    """
    Integrate coupled climate-ice sheet system using RK4.

    Parameters
    ----------
    t_span : tuple
        (t_start, t_end) in years.
    y0 : ndarray
        Initial state [x, y, z, V, T, h].
    orbital_forcing_func : callable
        f(t) -> orbital forcing.
    dt : float
        Time step in years.
    mu, eta : float
        Oscillator parameters.

    Returns
    -------
    t_array : ndarray
        Time points.
    sol : ndarray
        Solution array (n_steps, 6).
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)
    sol = np.zeros((n_steps, len(y0)))
    sol[0] = y0

    y = np.array(y0, dtype=float)
    for i in range(1, n_steps):
        t = t_array[i - 1]
        y = rk4_step(coupled_climate_ice_deriv, t, y, dt, orbital_forcing_func, mu, eta)
        # Boundary checks
        y[3] = max(0.0, y[3])  # V_ice >= 0
        y[4] = np.clip(y[4], 200.0, 350.0)  # T in range
        sol[i] = y

    return t_array, sol


def compute_ice_line_latitude(T_profile, latitudes_deg):
    """
    Find the ice line latitude where T = T_freeze.

    Parameters
    ----------
    T_profile : ndarray
        Temperature profile across latitudes.
    latitudes_deg : ndarray
        Latitude array.

    Returns
    -------
    float
        Ice line latitude (positive = Northern Hemisphere).
    """
    T_freeze = 263.15  # -10 C
    # Find where temperature crosses T_freeze
    for i in range(len(latitudes_deg) - 1):
        if (T_profile[i] - T_freeze) * (T_profile[i + 1] - T_freeze) < 0:
            # Linear interpolation
            frac = (T_freeze - T_profile[i]) / (T_profile[i + 1] - T_profile[i])
            return latitudes_deg[i] + frac * (latitudes_deg[i + 1] - latitudes_deg[i])
    # If no crossing, return polewardmost latitude below freezing
    below = latitudes_deg[T_profile < T_freeze]
    if len(below) > 0:
        return float(np.max(np.abs(below)))
    return 0.0
