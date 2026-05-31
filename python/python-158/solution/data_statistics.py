"""
data_statistics.py
==================
Statistical post-processing of multi-condition combustion simulation results.

Incorporates:
- brc_naive (118): grouped aggregation (city statistics) applied to
  combustion condition groups.
- hand_data (502): discrete data contour fitting via polynomial approximation.

Scientific application: after running many burner simulations with varying
operating conditions, aggregate results by condition groups (e.g., by
excess air ratio bins) and compute statistics: mean NOx, min, max, std.
Also fit response surfaces to the data.

Key formulas:
    Grouped mean:
        mu_g = (1/N_g) * sum_{i in group g} y_i
    
    Response surface (2D polynomial):
        R(x1, x2) = sum_{i=0}^p sum_{j=0}^{p-i} c_{ij} * x1^i * x2^j
    
    Coefficients via least squares:
        c = (Phi^T Phi)^{-1} Phi^T y
    where Phi is the Vandermonde matrix.
"""

import numpy as np
from typing import List, Dict


# ======================================================================
# 1. Grouped aggregation (from brc_naive)
# ======================================================================

def group_statistics(
    conditions: np.ndarray,
    nox_values: np.ndarray,
    burnout_values: np.ndarray,
    n_bins: int = 5
) -> dict:
    """
    Aggregate simulation results by condition bins.
    Analogous to brc_naive city grouping.
    
    Args:
        conditions: array of condition values (e.g., excess air ratio)
        nox_values: corresponding NOx emissions [ppm]
        burnout_values: corresponding burnout efficiencies [-]
        n_bins: number of bins for grouping
    
    Returns:
        dict with bin edges and statistics per bin.
    """
    if len(conditions) == 0:
        return {}
    
    cmin, cmax = np.min(conditions), np.max(conditions)
    if cmax <= cmin:
        cmax = cmin + 1.0
    
    bin_edges = np.linspace(cmin, cmax, n_bins + 1)
    
    groups = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < n_bins - 1:
            mask = (conditions >= lo) & (conditions < hi)
        else:
            mask = (conditions >= lo) & (conditions <= hi)
        
        if np.any(mask):
            groups.append({
                "bin_index": b,
                "bin_range": (lo, hi),
                "count": int(np.sum(mask)),
                "NOx_mean": float(np.mean(nox_values[mask])),
                "NOx_min": float(np.min(nox_values[mask])),
                "NOx_max": float(np.max(nox_values[mask])),
                "NOx_std": float(np.std(nox_values[mask])),
                "burnout_mean": float(np.mean(burnout_values[mask])),
                "burnout_min": float(np.min(burnout_values[mask])),
                "burnout_max": float(np.max(burnout_values[mask])),
            })
    
    return {
        "bin_edges": bin_edges,
        "groups": groups,
        "global_NOx_mean": float(np.mean(nox_values)),
        "global_NOx_min": float(np.min(nox_values)),
        "global_NOx_max": float(np.max(nox_values)),
        "global_burnout_mean": float(np.mean(burnout_values)),
    }


# ======================================================================
# 2. Response surface fitting (from hand_data contour fitting)
# ======================================================================

def polynomial_features_2d(x1: np.ndarray, x2: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Build Vandermonde matrix for 2D polynomial of given degree.
    Basis: {x1^i * x2^j | i+j <= degree}
    """
    n = len(x1)
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x1 ** i) * (x2 ** j))
    return np.column_stack(cols)


def fit_response_surface(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, degree: int = 3
) -> dict:
    """
    Fit a 2D polynomial response surface via least squares.
    
    Model:
        y = Phi(x1, x2) * c + epsilon
    
    Solution:
        c = (Phi^T Phi + lambda*I)^{-1} Phi^T y   (ridge regression for stability)
    """
    Phi = polynomial_features_2d(x1, x2, degree)
    
    # Ridge regression for numerical stability
    lam = 1e-6
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.T @ y
    
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    
    y_pred = Phi @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    return {
        "coefficients": coeffs,
        "degree": degree,
        "R2": float(r2),
        "RMSE": float(rmse),
        "y_pred": y_pred,
    }


def evaluate_response_surface(coeffs: np.ndarray, x1: float, x2: float,
                              degree: int = 3) -> float:
    """Evaluate fitted response surface at a single point."""
    idx = 0
    y = 0.0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            y += coeffs[idx] * (x1 ** i) * (x2 ** j)
            idx += 1
    return y


# ======================================================================
# 3. Uncertainty quantification
# ======================================================================

def monte_carlo_uncertainty(
    model_func, param_means: np.ndarray, param_stds: np.ndarray,
    n_samples: int = 1000, seed: int = 42
) -> dict:
    """
    Propagate parameter uncertainties through the combustion model
    using Monte Carlo sampling.
    
    Returns mean, std, and 95% confidence interval of model output.
    """
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(n_samples):
        sample = rng.normal(param_means, param_stds)
        try:
            out = model_func(sample)
            if np.isfinite(out):
                outputs.append(out)
        except Exception:
            pass
    
    outputs = np.array(outputs)
    if len(outputs) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_95": (0.0, 0.0)}
    
    return {
        "mean": float(np.mean(outputs)),
        "std": float(np.std(outputs)),
        "ci_95": (float(np.percentile(outputs, 2.5)),
                   float(np.percentile(outputs, 97.5))),
        "n_valid": len(outputs),
    }
