"""
黄金分割优化模块
Golden section search for optimizing smearing width σ and η/s ratio.

Algorithms sourced from:
  - 834_opt_golden (golden section search for 1D minimization)
"""
import numpy as np
from physics_constants import GOLDEN_RATIO


def golden_section_search(f, a, b, tol=1e-5, max_iter=100):
    """
    Golden section search for minimizing f(x) on [a, b].

    From 834_opt_golden: assumes f is unimodal on [a, b].

    Parameters
    ----------
    f : callable
        Function to minimize
    a, b : float
        Search interval
    tol : float
        Tolerance on interval width
    max_iter : int
        Maximum iterations

    Returns
    -------
    x_opt : float
        Optimal x
    f_opt : float
        Minimum function value
    n_iter : int
        Number of iterations
    """
    gr = GOLDEN_RATIO
    x1 = b - (b - a) / gr
    x2 = a + (b - a) / gr

    f1 = f(x1)
    f2 = f(x2)

    for i in range(max_iter):
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / gr
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / gr
            f2 = f(x2)

    x_opt = 0.5 * (a + b)
    f_opt = f(x_opt)
    return x_opt, f_opt, i + 1


def optimize_smearing_width(eps_target, initial_conditions_func, n_grid=64,
                            extent=12.0, sigma_range=(0.2, 1.5)):
    """
    Optimize Gaussian smearing width σ to match target energy density profile.

    Objective: minimize ||ε_computed(σ) - ε_target||²

    Parameters
    ----------
    eps_target : ndarray
        Target energy density
    initial_conditions_func : callable
        Function(σ) → ε(σ)
    n_grid : int
        Grid size
    extent : float
        Domain extent
    sigma_range : tuple
        (σ_min, σ_max) search range

    Returns
    -------
    sigma_opt : float
        Optimal smearing width
    error_opt : float
        Minimum error
    """
    def objective(sigma):
        eps_computed = initial_conditions_func(sigma)
        return np.sum((eps_computed - eps_target)**2)

    sigma_opt, f_opt, n_iter = golden_section_search(
        objective, sigma_range[0], sigma_range[1], tol=0.01, max_iter=50)

    return sigma_opt, f_opt


def optimize_eta_over_s(v2_data, v2_model_func, eta_s_range=(0.05, 0.3)):
    """
    Optimize η/s to match experimental v2(pT) data.

    Objective: minimize χ² = Σ (v2_model - v2_data)² / σ_data²

    Parameters
    ----------
    v2_data : tuple (pt, v2, v2_err)
        Experimental data
    v2_model_func : callable
        Function(η/s) → v2_model array
    eta_s_range : tuple
        Search range for η/s

    Returns
    -------
    eta_s_opt : float
        Optimal η/s
    chi2_min : float
        Minimum χ²
    """
    pt_data, v2_exp, v2_err = v2_data

    def chi_squared(eta_s):
        v2_model = v2_model_func(eta_s)
        return np.sum(((v2_model - v2_exp) / v2_err)**2)

    eta_s_opt, chi2_min, n_iter = golden_section_search(
        chi_squared, eta_s_range[0], eta_s_range[1], tol=0.001, max_iter=50)

    return eta_s_opt, chi2_min


def optimize_freezeout_temperature(spectrum_data, cooper_frye_func, T_range=(0.10, 0.18)):
    """
    Optimize freezeout temperature to match hadron spectra.

    Parameters
    ----------
    spectrum_data : tuple (pt, dN_dpt)
        Experimental spectrum
    cooper_frye_func : callable
        Function(T_freeze) → spectrum
    T_range : tuple
        Search range

    Returns
    -------
    T_opt : float
        Optimal freezeout temperature
    error : float
    """
    pt_data, spec_exp = spectrum_data

    def objective(T_freeze):
        spec_model = cooper_frye_func(T_freeze)
        return np.sum((spec_model - spec_exp)**2)

    T_opt, err, n_iter = golden_section_search(
        objective, T_range[0], T_range[1], tol=0.001, max_iter=50)

    return T_opt, err


def grid_search_2d(f, x_range, y_range, n_points=20):
    """
    2D grid search followed by golden section refinement.

    Parameters
    ----------
    f : callable
        Function f(x, y) to minimize
    x_range, y_range : tuple
        (min, max) for each dimension
    n_points : int
        Grid resolution

    Returns
    -------
    x_opt, y_opt : float
        Optimal point
    f_opt : float
    """
    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.linspace(y_range[0], y_range[1], n_points)

    best_val = np.inf
    best_xy = (0, 0)

    for x in xs:
        for y in ys:
            val = f(x, y)
            if val < best_val:
                best_val = val
                best_xy = (x, y)

    # Refine with golden section along each dimension
    def f_x(x):
        return f(x, best_xy[1])

    def f_y(y):
        return f(best_xy[0], y)

    x_opt, _, _ = golden_section_search(f_x, x_range[0], x_range[1])
    y_opt, f_opt, _ = golden_section_search(f_y, y_range[0], y_range[1])

    return x_opt, y_opt, f_opt
