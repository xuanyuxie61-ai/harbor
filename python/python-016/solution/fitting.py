"""
Tight-Binding Parameter Fitting via Least-Squares Approximation
================================================================
Fits the hopping parameters of the twisted bilayer graphene tight-binding
model to reference data (e.g., from first-principles DFT calculations)
using Chebyshev-spaced sampling and Lagrange-basis least-squares.

Scientific Background
---------------------
Given a set of reference energies {E_ref(k_i)} computed by DFT and the
corresponding tight-binding predictions {E_TB(k_i; {t})}, we minimize

    χ²({t}) = Σ_i w_i [E_ref(k_i) − E_TB(k_i; {t})]²

where {t} are the hopping parameters (t₀, t′, w₀, ξ, etc.).

For a single parameter t that controls the energy scale, the problem
reduces to fitting a function f(t) to data.  We use Chebyshev nodes
on [a, b]:

    x_j = (a+b)/2 + (b−a)/2 · cos(π(2j+1)/(2m)),   j = 0,…,m−1

which minimize Runge phenomena and give exponentially convergent
interpolation for analytic functions.

The least-squares polynomial of degree n (n < m) in Lagrange form is

    p(x) = Σ_{k=0}^{n} c_k L_k(x)

where L_k are Lagrange basis polynomials.  The coefficients c are found
by solving the normal equations

    (Aᵀ A) c = Aᵀ y

with A_{jk} = L_k(x_j).

For multi-parameter fitting we use a sequential approach: fit the
intralayer hopping t₀ first, then the interlayer w₀, then the angular
anisotropy α.
"""

import numpy as np
from typing import Tuple, Optional, Callable


def chebyshev_nodes(a: float, b: float, m: int) -> np.ndarray:
    """
    Generate m Chebyshev nodes of the first kind on [a, b].

        x_j = (a+b)/2 + (b-a)/2 · cos(π(2j+1)/(2m))

    Parameters
    ----------
    a, b : float
        Interval endpoints.
    m : int
        Number of nodes.

    Returns
    -------
    np.ndarray of shape (m,)
    """
    if m < 1:
        raise ValueError("m must be positive.")
    if b <= a:
        raise ValueError("b must be greater than a.")
    j = np.arange(m)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (2.0 * j + 1.0) / (2.0 * m))
    return x


def lagrange_basis(
    nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Evaluate Lagrange basis polynomials at points x_eval.

    L_k(x) = Π_{j≠k} (x − x_j) / (x_k − x_j)

    Parameters
    ----------
    nodes : np.ndarray of shape (n,)
    x_eval : np.ndarray of shape (m,)

    Returns
    -------
    np.ndarray of shape (m, n)
        L[k, i] = L_i(x_k).
    """
    n = nodes.size
    m = x_eval.size
    L = np.ones((m, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = nodes[i] - nodes[j]
                if abs(denom) < 1e-14:
                    raise ValueError("Duplicate nodes in Lagrange basis.")
                L[:, i] *= (x_eval - nodes[j]) / denom
    return L


def least_squares_lagrange_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a degree-n polynomial to data (x_i, y_i) using least squares
    in the Lagrange basis on Chebyshev nodes.

    Parameters
    ----------
    x_data : np.ndarray of shape (m,)
    y_data : np.ndarray of shape (m,)
    degree : int
        Polynomial degree (n < m).
    weights : np.ndarray, optional
        Weights w_i for weighted least squares.

    Returns
    -------
    coeffs : np.ndarray of shape (degree+1,)
        Coefficients in the Lagrange basis.
    cheb_nodes : np.ndarray of shape (degree+1,)
        Chebyshev nodes used as basis points.
    """
    m = x_data.size
    if y_data.size != m:
        raise ValueError("x_data and y_data must have the same length.")
    if degree < 0 or degree >= m:
        raise ValueError("degree must satisfy 0 <= degree < m.")

    # Chebyshev nodes for the basis
    a = float(np.min(x_data))
    b = float(np.max(x_data))
    cheb_nodes = chebyshev_nodes(a, b, degree + 1)

    # Lagrange basis evaluated at data points
    L = lagrange_basis(cheb_nodes, x_data)

    if weights is not None:
        W = np.sqrt(weights)
        L = L * W[:, None]
        y = y_data * W
    else:
        y = y_data

    # Solve normal equations via SVD for stability
    coeffs, residuals, rank, s = np.linalg.lstsq(L, y, rcond=None)

    return coeffs, cheb_nodes


def evaluate_lagrange_polynomial(
    x_eval: np.ndarray,
    coeffs: np.ndarray,
    cheb_nodes: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the fitted Lagrange polynomial at new points.

    Parameters
    ----------
    x_eval : np.ndarray
    coeffs : np.ndarray
    cheb_nodes : np.ndarray

    Returns
    -------
    np.ndarray
    """
    L = lagrange_basis(cheb_nodes, x_eval)
    return L @ coeffs


def fit_tight_binding_parameters(
    kpoints_ref: np.ndarray,
    energies_ref: np.ndarray,
    param_ranges: dict,
    tb_calculator: Callable,
    n_samples_per_param: int = 10,
) -> dict:
    """
    Fit tight-binding parameters to reference band-structure data.

    We perform a sequential 1D least-squares fit for each parameter
    while holding the others fixed at their current best values.

    Parameters
    ----------
    kpoints_ref : np.ndarray of shape (N_k, 2)
    energies_ref : np.ndarray of shape (N_k, n_bands)
    param_ranges : dict
        Mapping parameter_name → (min, max).
    tb_calculator : callable
        Function(params_dict, kpoints) → energies.
    n_samples_per_param : int

    Returns
    -------
    dict
        Fitted parameter values.
    """
    best_params = {k: 0.5 * (v[0] + v[1]) for k, v in param_ranges.items()}

    for param_name, (p_min, p_max) in param_ranges.items():
        # Sample parameter values
        p_values = np.linspace(p_min, p_max, n_samples_per_param)
        chi2_values = np.zeros(n_samples_per_param)

        for i, p_val in enumerate(p_values):
            trial_params = best_params.copy()
            trial_params[param_name] = p_val
            try:
                energies_tb = tb_calculator(trial_params, kpoints_ref)
                diff = energies_tb - energies_ref
                chi2_values[i] = np.mean(diff ** 2)
            except Exception:
                chi2_values[i] = 1e10

        # Fit a quadratic through the chi² landscape
        # Use Chebyshev-spaced subset if many samples
        if n_samples_per_param >= 5:
            subset_idx = np.linspace(0, n_samples_per_param - 1, 5, dtype=int)
            p_sub = p_values[subset_idx]
            c_sub = chi2_values[subset_idx]
        else:
            p_sub = p_values
            c_sub = chi2_values

        # Fit degree-2 Lagrange polynomial and find minimum
        try:
            coeffs, nodes = least_squares_lagrange_fit(p_sub, c_sub, degree=min(2, len(p_sub) - 1))
            # Fine grid search for minimum
            p_fine = np.linspace(p_min, p_max, 200)
            c_fine = evaluate_lagrange_polynomial(p_fine, coeffs, nodes)
            best_idx = np.argmin(c_fine)
            best_params[param_name] = float(p_fine[best_idx])
        except Exception:
            # Fallback: direct minimum of samples
            best_params[param_name] = float(p_values[np.argmin(chi2_values)])

    return best_params


def cross_validation_error(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    n_folds: int = 5,
) -> float:
    """
    Compute cross-validation error for polynomial fitting to guard
    against overfitting.

    Parameters
    ----------
    x_data, y_data : np.ndarray
    degree : int
    n_folds : int

    Returns
    -------
    float
        Average validation MSE across folds.
    """
    N = x_data.size
    if N < n_folds:
        n_folds = max(1, N)
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N // n_folds
    errors = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else N
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        if train_idx.size <= degree:
            continue

        try:
            coeffs, nodes = least_squares_lagrange_fit(
                x_data[train_idx], y_data[train_idx], degree
            )
            y_pred = evaluate_lagrange_polynomial(x_data[val_idx], coeffs, nodes)
            mse = np.mean((y_pred - y_data[val_idx]) ** 2)
            errors.append(mse)
        except Exception:
            errors.append(1e10)

    return float(np.mean(errors)) if errors else 1e10
