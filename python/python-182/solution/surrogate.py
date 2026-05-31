"""
surrogate.py

Polynomial surrogate model construction via least-squares approximation
with SVD-based regularization, reimplemented from the seed project.

Provides a fast 1D surrogate for the FitzHugh-Nagumo stationary voltage
as a function of the stimulus parameter d, used to accelerate likelihood
evaluations inside the MCMC sampler.
"""
import numpy as np
from forward_models import fhn_stationary_voltage


def least_squares_approximant_coef(nd: int, xd: np.ndarray, yd: np.ndarray, m: int):
    """
    Compute coefficients of the least-squares polynomial approximant
    of degree at most m-1 to data (xd, yd).

    Minimizes  || V c - yd ||_2  where V is the Vandermonde matrix.
    Uses both lstsq and SVD-thresholded pseudoinverse for robustness.

    Parameters:
        nd: number of data points
        xd: data abscissas, shape (nd,)
        yd: data ordinates, shape (nd,)
        m: desired polynomial order (number of coefficients)

    Returns:
        c: coefficients [c0, c1, ..., c_{m-1}] where c0 is constant term
    """
    if nd < 1:
        raise ValueError("least_squares_approximant_coef: nd must be >= 1")
    xd = np.asarray(xd, dtype=float).ravel()
    yd = np.asarray(yd, dtype=float).ravel()
    A = np.vander(xd, m, increasing=True)
    c1 = np.linalg.lstsq(A, yd, rcond=None)[0]
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    eps_sqrt = np.sqrt(np.finfo(float).eps)
    s_inv = np.where(s < eps_sqrt, 0.0, 1.0 / s)
    c2 = vh.T @ np.diag(s_inv) @ u.T @ yd
    if nd < m:
        c = np.zeros(m, dtype=float)
        c[:nd] = c1[:nd]
    else:
        c = c1
    return c


def poly_value(c: np.ndarray, x: np.ndarray):
    """
    Evaluate polynomial with coefficients c (constant term first)
    at points x using Horner's method.
    """
    x = np.asarray(x, dtype=float)
    # np.polyval expects highest degree first
    return np.polyval(c[::-1], x)


def build_fhn_surrogate(a_fixed: float = 0.7, b_fixed: float = 0.8,
                        c_fixed: float = 3.0, degree: int = 6,
                        n_train: int = 15, d_min: float = -0.3, d_max: float = 0.3):
    """
    Build a 1D polynomial surrogate for v_end(d) = FHN stationary voltage
    as a function of stimulus d, with (a,b,c) fixed.

    Returns:
        surrogate dict with keys 'c', 'd_min', 'd_max', 'a_fixed', 'b_fixed'
    """
    xd = np.linspace(d_min, d_max, n_train)
    yd = np.empty(n_train, dtype=float)
    for i, d in enumerate(xd):
        yd[i] = fhn_stationary_voltage(a_fixed, b_fixed, c_fixed, d)
    c = least_squares_approximant_coef(n_train, xd, yd, degree)
    return {
        'c': c,
        'd_min': d_min,
        'd_max': d_max,
        'a_fixed': a_fixed,
        'b_fixed': b_fixed,
    }


def surrogate_predict(surrogate: dict, d: float):
    """
    Predict v_end using the polynomial surrogate.
    Clamps d to [d_min, d_max] for numerical safety.
    """
    d_clamped = max(surrogate['d_min'], min(surrogate['d_max'], d))
    return float(poly_value(surrogate['c'], np.array([d_clamped]))[0])
