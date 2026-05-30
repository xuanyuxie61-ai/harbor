
import numpy as np
from typing import Tuple, Optional, Callable


def chebyshev_nodes(a: float, b: float, m: int) -> np.ndarray:
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
    m = x_data.size
    if y_data.size != m:
        raise ValueError("x_data and y_data must have the same length.")
    if degree < 0 or degree >= m:
        raise ValueError("degree must satisfy 0 <= degree < m.")


    a = float(np.min(x_data))
    b = float(np.max(x_data))
    cheb_nodes = chebyshev_nodes(a, b, degree + 1)


    L = lagrange_basis(cheb_nodes, x_data)

    if weights is not None:
        W = np.sqrt(weights)
        L = L * W[:, None]
        y = y_data * W
    else:
        y = y_data


    coeffs, residuals, rank, s = np.linalg.lstsq(L, y, rcond=None)

    return coeffs, cheb_nodes


def evaluate_lagrange_polynomial(
    x_eval: np.ndarray,
    coeffs: np.ndarray,
    cheb_nodes: np.ndarray,
) -> np.ndarray:
    L = lagrange_basis(cheb_nodes, x_eval)
    return L @ coeffs


def fit_tight_binding_parameters(
    kpoints_ref: np.ndarray,
    energies_ref: np.ndarray,
    param_ranges: dict,
    tb_calculator: Callable,
    n_samples_per_param: int = 10,
) -> dict:
    best_params = {k: 0.5 * (v[0] + v[1]) for k, v in param_ranges.items()}

    for param_name, (p_min, p_max) in param_ranges.items():

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



        if n_samples_per_param >= 5:
            subset_idx = np.linspace(0, n_samples_per_param - 1, 5, dtype=int)
            p_sub = p_values[subset_idx]
            c_sub = chi2_values[subset_idx]
        else:
            p_sub = p_values
            c_sub = chi2_values


        try:
            coeffs, nodes = least_squares_lagrange_fit(p_sub, c_sub, degree=min(2, len(p_sub) - 1))

            p_fine = np.linspace(p_min, p_max, 200)
            c_fine = evaluate_lagrange_polynomial(p_fine, coeffs, nodes)
            best_idx = np.argmin(c_fine)
            best_params[param_name] = float(p_fine[best_idx])
        except Exception:

            best_params[param_name] = float(p_values[np.argmin(chi2_values)])

    return best_params


def cross_validation_error(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    n_folds: int = 5,
) -> float:
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
