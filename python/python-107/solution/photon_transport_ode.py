"""
photon_transport_ode.py

ODE-based photon transport models for light propagation in layered tissue.
Combines:
- sensitive_ode: sensitivity analysis of photon density to boundary conditions
- ode_midpoint: midpoint method for stable integration of transport ODEs
- fitzhugh_nagumo & glycolysis: biological dynamics affecting optical properties
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sensitive ODE system (from sensitive_ode)
# y' = [ y2; y1 ]  =>  y1'' = y1
# This models exponential photon amplification/attenuation in a simplified
# homogenized medium where the growth/decay rate depends on the net
# scattering-absorption balance.
# ---------------------------------------------------------------------------

def sensitive_photon_deriv(t, y, growth_rate=1.0):
    """
    Derivative of photon density with sensitive dependence on initial conditions.

    dy1/dt = y2
    dy2/dt = lambda^2 * y1

    where lambda^2 = mu_a * (mu_a + 2 mu_s') / 3  in diffusion limit.

    Parameters
    ----------
    t : float
        Propagation distance (optical depth).
    y : array_like, shape (2,)
        [photon_density, flux]
    growth_rate : float
        Effective growth rate lambda.

    Returns
    -------
    dydt : ndarray, shape (2,)
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    dydt = np.zeros(2, dtype=float)
    dydt[0] = y[1]
    dydt[1] = growth_rate * growth_rate * y[0]
    return dydt


def sensitive_photon_exact(t, y0, growth_rate=1.0):
    """
    Exact solution of the sensitive photon ODE.

    y1(t) = (1 - eps/2) exp(-lambda t) + (eps/2) exp(lambda t)
    where eps = y0[0] - 1.

    Parameters
    ----------
    t : array_like
        Propagation distances.
    y0 : array_like, shape (2,)
        Initial condition [phi0, j0].
    growth_rate : float
        Lambda.

    Returns
    -------
    y : ndarray, shape (len(t), 2)
    """
    t = np.asarray(t, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    eps = y0[0] - 1.0
    n = t.size
    y = np.zeros((n, 2), dtype=float)
    for i in range(n):
        y[i, 0] = (1.0 - eps / 2.0) * np.exp(-growth_rate * t[i]) + (eps / 2.0) * np.exp(growth_rate * t[i])
        y[i, 1] = -(1.0 - eps / 2.0) * np.exp(-growth_rate * t[i]) + (eps / 2.0) * np.exp(growth_rate * t[i])
        y[i, 1] *= growth_rate
    return y


# ---------------------------------------------------------------------------
# Midpoint method for general photon transport ODEs (from ode_midpoint)
# ---------------------------------------------------------------------------

def ode_midpoint_solve(f, a, b, ya, n_steps, theta=0.5, it_max=10):
    """
    Midpoint method for y' = f(t, y) with fixed-point iteration.

    Step:
      x_m = x_i + theta * h
      y_m^{(0)} = y_i
      y_m^{(j+1)} = y_i + theta * h * f(x_m, y_m^{(j)})
      y_{i+1} = (1/theta) * y_m + (1 - 1/theta) * y_i

    Parameters
    ----------
    f : callable
        f(t, y) -> dydt.
    a, b : float
        Interval.
    ya : ndarray
        Initial condition y(a).
    n_steps : int
        Number of steps.
    theta : float
        Midpoint parameter (0.5 for standard midpoint).
    it_max : int
        Fixed-point iterations per step.

    Returns
    -------
    t : ndarray
        Time nodes.
    y : ndarray
        Solution array, shape (n_steps+1, len(ya)).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if a >= b:
        raise ValueError("Require a < b.")
    ya = np.atleast_1d(np.asarray(ya, dtype=float))
    dim = ya.size

    t = np.linspace(a, b, n_steps + 1)
    y = np.zeros((n_steps + 1, dim), dtype=float)
    y[0, :] = ya
    h = (b - a) / n_steps

    for i in range(n_steps):
        xm = t[i] + theta * h
        ym = y[i, :].copy()
        for _ in range(it_max):
            ym = y[i, :] + theta * h * np.atleast_1d(f(xm, ym))
        y[i + 1, :] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i, :]
    return t, y


# ---------------------------------------------------------------------------
# Layered photon transport: piecewise ODE across tissue layers
# ---------------------------------------------------------------------------

def layered_photon_transport(layer_boundaries, layer_properties, y0, n_steps_per_layer=50):
    """
    Solve photon transport across layered tissue using midpoint method.

    Each layer has optical properties (mu_a, mu_s, g).
    The ODE in each layer is:
      dphi/dz = -J
      dJ/dz   = - (3 mu_a (mu_a + mu_s')) phi
    where J is the net flux and phi is the photon density (fluence rate).

    Parameters
    ----------
    layer_boundaries : array_like
        Sorted z-coordinates of layer interfaces, including start and end.
    layer_properties : list of dict
        Each dict has keys 'mu_a', 'mu_s', 'g'.
    y0 : array_like, shape (2,)
        Initial [phi, J] at first boundary.
    n_steps_per_layer : int
        Steps per layer.

    Returns
    -------
    z_all : ndarray
        Concatenated depth array.
    y_all : ndarray
        Concatenated solution array.
    """
    layer_boundaries = np.asarray(layer_boundaries, dtype=float)
    if len(layer_boundaries) < 2:
        raise ValueError("Need at least 2 boundaries.")
    n_layers = len(layer_boundaries) - 1
    if len(layer_properties) != n_layers:
        raise ValueError("layer_properties length must match number of layers.")

    z_all = []
    y_all = []
    y_current = np.asarray(y0, dtype=float)

    for i in range(n_layers):
        a = layer_boundaries[i]
        b = layer_boundaries[i + 1]
        props = layer_properties[i]
        mu_a = props['mu_a']
        mu_s = props['mu_s']
        g = props['g']
        mu_s_prime = (1.0 - g) * mu_s
        # Effective diffusion coefficient gives growth rate
        D = 1.0 / (3.0 * (mu_s_prime + mu_a))
        lambda_eff = np.sqrt(mu_a / D) if D > 0 else 0.0

        def f_layer(z, y):
            return sensitive_photon_deriv(z, y, growth_rate=lambda_eff)

        t, y = ode_midpoint_solve(f_layer, a, b, y_current, n_steps_per_layer)
        if i == 0:
            z_all.extend(t)
            y_all.extend(y)
        else:
            z_all.extend(t[1:])
            y_all.extend(y[1:])
        y_current = y[-1, :].copy()

    return np.array(z_all), np.array(y_all)


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo dynamics for excitable tissue (from fitzhugh_nagumo_ode)
# ---------------------------------------------------------------------------

def fitzhugh_nagumo_deriv(t, y, a=0.7, b=0.8, c=12.5, d=0.5):
    """
    FitzHugh-Nagumo model for excitable membrane dynamics.

    dv/dt = v - v^3/3 - w + d
    dw/dt = (v + a - b w) / c

    In OCT context, the membrane potential v modulates local refractive index
    through electro-optic (Pockels) effect: delta_n ~ r_eo * E ~ k * v.

    Parameters
    ----------
    t : float
        Time.
    y : array_like, shape (2,)
        [v, w].
    a, b, c, d : float
        Model parameters.

    Returns
    -------
    dydt : ndarray, shape (2,)
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    v = y[0]
    w = y[1]
    dvdt = v - (v ** 3) / 3.0 - w + d
    dwdt = (v + a - b * w) / c
    return np.array([dvdt, dwdt])


# ---------------------------------------------------------------------------
# Glycolysis oscillator (from glycolysis_ode)
# ---------------------------------------------------------------------------

def glycolysis_deriv(t, y, a=0.08, b=0.6):
    """
    Sel'kov glycolysis model for metabolic oscillations in tissue.

    du/dt = -u + a v + u^2 v
    dv/dt = b - a v - u^2 v

    Metabolic activity changes local temperature and refractive index,
    providing OCT contrast in functional imaging.

    Parameters
    ----------
    t : float
        Time.
    y : array_like, shape (2,)
        [u, v] (substrate concentrations).
    a, b : float
        Model parameters.

    Returns
    -------
    dydt : ndarray, shape (2,)
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    u = y[0]
    v = y[1]
    dudt = -u + a * v + u * u * v
    dvdt = b - a * v - u * u * v
    return np.array([dudt, dvdt])


def glycolysis_equilibrium(a=0.08, b=0.6):
    """
    Equilibrium point of Sel'kov glycolysis model.

    u* = b,  v* = b / (a + b^2)

    Returns
    -------
    y_eq : ndarray, shape (2,)
    """
    denom = a + b * b
    if abs(denom) < 1e-14:
        raise ValueError("Denominator too small in equilibrium calculation.")
    return np.array([b, b / denom])


# ---------------------------------------------------------------------------
# Coupled biological-optical dynamics
# ---------------------------------------------------------------------------

def refractive_index_from_bio_state(v_membrane, u_metabolite,
                                     n0=1.33, alpha_eo=1e-4, alpha_thermo=1e-3):
    """
    Compute local refractive index modulation from biological state.

    n_local = n0 + alpha_eo * v_membrane + alpha_thermo * u_metabolite

    Parameters
    ----------
    v_membrane : float
        Membrane potential (FHN v).
    u_metabolite : float
        Metabolite concentration (glycolysis u).
    n0 : float
        Baseline refractive index.
    alpha_eo, alpha_thermo : float
        Coupling coefficients.

    Returns
    -------
    n : float
    """
    n = n0 + alpha_eo * v_membrane + alpha_thermo * u_metabolite
    return max(n, 1.0)  # physical bound
