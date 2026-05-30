
import numpy as np
from typing import List, Dict






def group_statistics(
    conditions: np.ndarray,
    nox_values: np.ndarray,
    burnout_values: np.ndarray,
    n_bins: int = 5
) -> dict:
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






def polynomial_features_2d(x1: np.ndarray, x2: np.ndarray, degree: int = 3) -> np.ndarray:
    n = len(x1)
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x1 ** i) * (x2 ** j))
    return np.column_stack(cols)


def fit_response_surface(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, degree: int = 3
) -> dict:
    Phi = polynomial_features_2d(x1, x2, degree)
    

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
    idx = 0
    y = 0.0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            y += coeffs[idx] * (x1 ** i) * (x2 ** j)
            idx += 1
    return y






def monte_carlo_uncertainty(
    model_func, param_means: np.ndarray, param_stds: np.ndarray,
    n_samples: int = 1000, seed: int = 42
) -> dict:
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
