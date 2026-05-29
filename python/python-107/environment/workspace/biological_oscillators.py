"""
biological_oscillators.py

Models of active biological dynamics that modulate tissue optical properties
for functional OCT imaging.

Incorporates:
- fitzhugh_nagumo_ode: excitable membrane dynamics (cardiac/neural tissue)
- glycolysis_ode: metabolic oscillations (Sel'kov model)

These models provide time-varying refractive index perturbations that can be
detected by phase-sensitive OCT, enabling functional imaging of tissue
physiology without exogenous contrast agents.
"""

import numpy as np
from photon_transport_ode import ode_midpoint_solve, fitzhugh_nagumo_deriv, glycolysis_deriv


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo integration
# ---------------------------------------------------------------------------

def integrate_fitzhugh_nagumo(y0, t_span, n_steps=1000, a=0.7, b=0.8, c=12.5, d=0.5):
    """
    Integrate FitzHugh-Nagumo equations over t_span using midpoint method.

    Parameters
    ----------
    y0 : array_like, shape (2,)
        Initial [v, w].
    t_span : tuple
        (t0, t1).
    n_steps : int
    a, b, c, d : float
        Model parameters.

    Returns
    -------
    t : ndarray
    y : ndarray, shape (n_steps+1, 2)
    """
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    def f(t, y):
        return fitzhugh_nagumo_deriv(t, y, a, b, c, d)

    t, y = ode_midpoint_solve(f, t0, t1, y0, n_steps, theta=0.5, it_max=5)
    return t, y


# ---------------------------------------------------------------------------
# Glycolysis integration
# ---------------------------------------------------------------------------

def integrate_glycolysis(y0, t_span, n_steps=1000, a=0.08, b=0.6):
    """
    Integrate Sel'kov glycolysis model over t_span.

    Parameters
    ----------
    y0 : array_like, shape (2,)
        Initial [u, v].
    t_span : tuple
        (t0, t1).
    n_steps : int
    a, b : float
        Model parameters.

    Returns
    -------
    t : ndarray
    y : ndarray, shape (n_steps+1, 2)
    """
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    def f(t, y):
        return glycolysis_deriv(t, y, a, b)

    t, y = ode_midpoint_solve(f, t0, t1, y0, n_steps, theta=0.5, it_max=5)
    return t, y


# ---------------------------------------------------------------------------
# Refractive index time series from biological dynamics
# ---------------------------------------------------------------------------

def compute_refractive_index_timeseries(t, v_trace, u_trace,
                                        n0=1.33, alpha_eo=5e-4, alpha_thermo=2e-3):
    """
    Compute time-varying refractive index from biological state traces.

    n(t) = n0 + alpha_eo * v(t) + alpha_thermo * u(t)

    Physical basis:
    - Electro-optic (Pockels) effect: delta_n = r_ijk E_j E_k ~ alpha_eo v
    - Thermo-optic effect: dn/dT ~ -1e-4 K^-1, metabolic heat raises T ~ u

    Parameters
    ----------
    t : ndarray
        Time array.
    v_trace : ndarray
        Membrane potential trace (FHN v).
    u_trace : ndarray
        Metabolite concentration trace (glycolysis u).
    n0 : float
        Baseline refractive index.
    alpha_eo, alpha_thermo : float
        Coupling coefficients.

    Returns
    -------
    n_t : ndarray
        Refractive index time series.
    """
    v_trace = np.asarray(v_trace, dtype=float)
    u_trace = np.asarray(u_trace, dtype=float)
    n_t = n0 + alpha_eo * v_trace + alpha_thermo * u_trace
    # Physical bound: refractive index of biological tissue ~ 1.33-1.45
    n_t = np.clip(n_t, 1.30, 1.50)
    return n_t


# ---------------------------------------------------------------------------
# OCT phase shift from refractive index changes
# ---------------------------------------------------------------------------

def phase_shift_from_dn(dn, optical_path_length, lambda0=0.84):
    """
    Compute OCT phase shift from refractive index change.

    Delta_phi = (4 pi / lambda0) * OPL * dn

    where OPL = integral n(z) dz along the beam path.

    Parameters
    ----------
    dn : float or ndarray
        Refractive index change.
    optical_path_length : float
        Physical path length (micron).
    lambda0 : float
        Central wavelength (micron).

    Returns
    -------
    delta_phi : float or ndarray
        Phase shift in radians.
    """
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive.")
    delta_phi = (4.0 * np.pi / lambda0) * optical_path_length * dn
    return delta_phi


# ---------------------------------------------------------------------------
# Combined biological-OCT functional imaging model
# ---------------------------------------------------------------------------

def simulate_functional_oct_signal(t_bio, bio_params, oct_params):
    """
    Simulate functional OCT signal that captures biological dynamics.

    Parameters
    ----------
    t_bio : ndarray
        Biological time array.
    bio_params : dict
        {'type': 'FHN' or 'glycolysis', 'y0': [...], 'model_params': {...}}
    oct_params : dict
        {'lambda0': ..., 'path_length': ..., 'alpha_eo': ..., 'alpha_thermo': ...}

    Returns
    -------
    result : dict
        {'t': ..., 'n_t': ..., 'phase_shift': ..., 'intensity_modulation': ...}
    """
    n0 = oct_params.get('n0', 1.33)
    alpha_eo = oct_params.get('alpha_eo', 5e-4)
    alpha_thermo = oct_params.get('alpha_thermo', 2e-3)
    lambda0 = oct_params.get('lambda0', 0.84)
    path_length = oct_params.get('path_length', 100.0)

    bio_type = bio_params.get('type', 'FHN')
    y0 = np.asarray(bio_params.get('y0', [0.0, 0.0]), dtype=float)
    t0 = float(t_bio[0])
    t1 = float(t_bio[-1])
    n_steps = min(len(t_bio) - 1, 5000)

    if bio_type.upper() == 'FHN':
        mp = bio_params.get('model_params', {})
        t, y = integrate_fitzhugh_nagumo(
            y0, (t0, t1), n_steps,
            a=mp.get('a', 0.7), b=mp.get('b', 0.8),
            c=mp.get('c', 12.5), d=mp.get('d', 0.5)
        )
        v_trace = y[:, 0]
        u_trace = np.zeros_like(v_trace)
    elif bio_type.upper() == 'GLYCOLYSIS':
        mp = bio_params.get('model_params', {})
        t, y = integrate_glycolysis(
            y0, (t0, t1), n_steps,
            a=mp.get('a', 0.08), b=mp.get('b', 0.6)
        )
        v_trace = np.zeros_like(y[:, 0])
        u_trace = y[:, 0]
    else:
        raise ValueError(f"Unknown bio_type: {bio_type}")

    n_t = compute_refractive_index_timeseries(t, v_trace, u_trace, n0, alpha_eo, alpha_thermo)
    dn = n_t - n0
    delta_phi = phase_shift_from_dn(dn, path_length, lambda0)

    # Intensity modulation via interference: I ~ I0 (1 + cos(delta_phi))
    intensity_mod = 0.5 * (1.0 + np.cos(delta_phi))

    return {
        't': t,
        'n_t': n_t,
        'phase_shift': delta_phi,
        'intensity_modulation': intensity_mod,
        'v_trace': v_trace,
        'u_trace': u_trace
    }
