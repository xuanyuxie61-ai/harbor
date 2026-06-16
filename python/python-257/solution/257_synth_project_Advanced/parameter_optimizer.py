"""
parameter_optimizer.py
======================
Fitting cosmological parameters to a CMB power spectrum by minimising
    chi^2(theta) = sum_{l=2}^{l_max}  (C_l^obs - C_l^th(theta))^2 / sigma_l^2

Two derivative-free optimisers are provided:

  1. Nelder-Mead simplex  (from nelder_mead.m)
     - Reflection, expansion, contraction, shrink
     - Parameters: rho=1, xi=2, gamma=0.5, sigma=0.5

  2. Coordinate search  (from coordinate_search.m)
     - Stencil search along coordinate axes
     - Parameters: delta=initial step, tolerance, max_feval

Both are appropriate for non-smooth or noisy objective functions
arising from Monte-Carlo estimators of C_l.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Dict, Optional

from cosmology_params import PLANCK_2018, cmb_cl_theory


# ---------------------------------------------------------------------------
# chi-squared objective
# ---------------------------------------------------------------------------
def build_chisq(cl_obs: List[float],
                 sigma_l: List[float],
                 l_min: int = 2,
                 l_max: int = 20) -> Callable[[Dict[str, float]], float]:
    """
    Build the chi-squared objective function  chi^2(theta).
    """
    def chisq(params: Dict[str, float]) -> float:
        s = 0.0
        for ell in range(l_min, l_max + 1):
            cl_th = cmb_cl_theory(ell, params)
            if sigma_l[ell] > 0.0:
                s += ((cl_obs[ell] - cl_th) / sigma_l[ell]) ** 2
        return s
    return chisq


# ---------------------------------------------------------------------------
# Nelder-Mead simplex optimiser
# ---------------------------------------------------------------------------
def nelder_mead(x0: List[float],
                  func: Callable[[List[float]], float],
                  rho: float = 1.0, xi: float = 2.0,
                  gamma: float = 0.5, sig: float = 0.5,
                  tolerance: float = 1.0e-6,
                  max_feval: int = 500) -> Tuple[List[float], float, int]:
    """
    Nelder-Mead simplex algorithm.
    x0 : list of n initial vertices (must be n+1 rows in (n+1) x n matrix,
          but for simplicity we accept a single point and build a simplex).
    Returns (x_opt, f_opt, n_feval).
    """
    n = len(x0)
    # Build initial simplex
    x = [x0[:]]
    for i in range(n):
        p = x0[:]
        p[i] += max(abs(x0[i]) * 0.05, 0.01)
        x.append(p)
    f = [func(xi) for xi in x]
    n_feval = n + 1
    while True:
        # Sort ascending
        order = sorted(range(n + 1), key=lambda k: f[k])
        x = [x[k] for k in order]
        f = [f[k] for k in order]
        # Check convergence
        if f[-1] - f[0] < tolerance:
            break
        if n_feval >= max_feval:
            break
        # Centroid (excluding worst)
        x_bar = [sum(x[i][d] for i in range(n)) / n for d in range(n)]
        # Reflection
        x_r = [(1.0 + rho) * x_bar[d] - rho * x[-1][d] for d in range(n)]
        f_r = func(x_r)
        n_feval += 1
        if f[0] <= f_r < f[-2]:
            x[-1] = x_r
            f[-1] = f_r
            continue
        if f_r < f[0]:
            # Expansion
            x_e = [(1.0 + rho * xi) * x_bar[d] - rho * xi * x[-1][d] for d in range(n)]
            f_e = func(x_e)
            n_feval += 1
            if f_e < f_r:
                x[-1] = x_e
                f[-1] = f_e
            else:
                x[-1] = x_r
                f[-1] = f_r
            continue
        # Contraction
        if f_r < f[-1]:
            x_c = [(1.0 + rho * gamma) * x_bar[d] - rho * gamma * x[-1][d] for d in range(n)]
        else:
            x_c = [(1.0 - gamma) * x_bar[d] + gamma * x[-1][d] for d in range(n)]
        f_c = func(x_c)
        n_feval += 1
        if f_c < f[-1]:
            x[-1] = x_c
            f[-1] = f_c
            continue
        # Shrink
        for i in range(1, n + 1):
            x[i] = [sig * x[i][d] + (1.0 - sig) * x[0][d] for d in range(n)]
            f[i] = func(x[i])
            n_feval += 1
    order = sorted(range(n + 1), key=lambda k: f[k])
    return x[order[0]], f[order[0]], n_feval


# ---------------------------------------------------------------------------
# Coordinate search (pattern search)
# ---------------------------------------------------------------------------
def coordinate_search(x0: List[float],
                        func: Callable[[List[float]], float],
                        delta: float = 1.0,
                        tolerance: float = 1.0e-6,
                        max_feval: int = 500) -> Tuple[List[float], float, int]:
    """
    Coordinate search (Hooke-Jeeves style stencil search).
    Returns (x_opt, f_opt, n_feval).
    """
    n = len(x0)
    x = x0[:]
    f = func(x)
    n_feval = 1
    while delta > tolerance and n_feval < max_feval:
        improved = False
        for i in range(n):
            for direction in [+1.0, -1.0]:
                x_trial = x[:]
                x_trial[i] += direction * delta
                f_trial = func(x_trial)
                n_feval += 1
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            delta *= 0.5
    return x, f, n_feval


# ---------------------------------------------------------------------------
# Fit cosmological parameters
# ---------------------------------------------------------------------------
def fit_cosmology(cl_obs: List[float], sigma_l: List[float],
                    l_min: int = 2, l_max: int = 20,
                    method: str = "nelder-mead",
                    initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Fit the base cosmological parameters (ns, ln10As, ombh2, omch2, H0)
    to the observed C_l.
    """
    if initial_params is None:
        initial_params = PLANCK_2018.copy()

    chisq = build_chisq(cl_obs, sigma_l, l_min, l_max)
    # Reduce to 5-parameter fit:  [H0, ombh2, omch2, ns, ln10As]
    free_keys = ["H0", "ombh2", "omch2", "ns", "ln10As"]
    x0 = [initial_params[k] for k in free_keys]

    def obj(x: List[float]) -> float:
        p = initial_params.copy()
        for k, v in zip(free_keys, x):
            p[k] = v
        return chisq(p)

    if method == "nelder-mead":
        x_opt, f_opt, nfev = nelder_mead(x0, obj)
    elif method == "coordinate":
        x_opt, f_opt, nfev = coordinate_search(x0, obj, delta=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = initial_params.copy()
    for k, v in zip(free_keys, x_opt):
        result[k] = v
    result["chi2"] = f_opt
    result["nfev"] = nfev
    return result


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test: fit to a synthetic observation
    true_params = PLANCK_2018.copy()
    cl_obs = [0.0] * 21
    sigma_l = [0.0] * 21
    for ell in range(2, 21):
        cl_obs[ell] = cmb_cl_theory(ell, true_params) * 1.0
        sigma_l[ell] = abs(cl_obs[ell]) * 0.1 + 1e-20
    # Perturb initial guess
    init = PLANCK_2018.copy()
    init["ns"] = 1.0
    init["H0"] = 70.0
    result = fit_cosmology(cl_obs, sigma_l, method="nelder-mead", initial_params=init)
    print("Fit result:")
    print(f"  ns     = {result['ns']:.4f}  (true = {true_params['ns']})")
    print(f"  H0     = {result['H0']:.2f}  (true = {true_params['H0']})")
    print(f"  chi^2  = {result['chi2']:.4e}")
