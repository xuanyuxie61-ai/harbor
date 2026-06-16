# -*- coding: utf-8 -*-
"""
L-BFGS-B optimiser for the effective low-energy Hamiltonian of the TFIM.

Given a target ground-state-energy curve E_target(L, lambda) (say,
from DMRG, from a tensor-network calculation, or from experiment) we
wish to fit an *effective* Hamiltonian of the form

    H_eff = - J_eff sum ZZ  -  h_eff sum X  -  V_eff sum ZZZZ  - ...

by minimising the nonlinear least-squares residual

    R(theta) = sum_{L, lambda} w_{L, lambda}
                 [ E_theta(L, lambda) - E_target(L, lambda) ]^2.

The ``V_eff ZZZZ`` term is a nearest-neighbour four-spin interaction
that captures leading irrelevant perturbations to the QCP and enters
the finite-size scaling corrections as the confluent exponent omega.

The optimiser wraps ``scipy.optimize.minimize`` with method L-BFGS-B
so that we can impose physical bounds  (J > 0, V >= 0, etc.)
inspired by the site-wise parameter optimisation of the GPP vegetation
code (de-ranit, 1119).
"""

from __future__ import annotations
from typing import Tuple, Callable, Dict
import numpy as np
from scipy.optimize import minimize
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Effective Hamiltonian
# ---------------------------------------------------------------------------
def effective_hamiltonian_energy(theta: Dict[str, float],
                                   L: int, lam: float) -> float:
    """Evaluate the per-site effective ground-state energy.

    Parameters
    ----------
    theta : dict
        Keys: 'J', 'V' (four-spin coupling), 'K' (next-nearest ZZ).
        The transverse field is derived from ``lam`` as h = J * lam.
    L : int
        System size.
    lam : float
        Dimensionless transverse-field ratio h / J.

    The energy is evaluated in mean-field decoupling (valid for
    capturing the gross lambda-dependence near the QCP):
        e_0 ~ - J (1 + V/2) - J * lam * sqrt(1 - (lam/lam_c_eff)^2)
              - K * (1 - lam/lam_c_eff)^2
    where lam_c_eff is a function of (J, V, K).  This closed form
    keeps the optimisation smooth and differentiable.
    """
    J = max(float(theta.get("J", 1.0)), C.EPS_NUM)
    V = float(theta.get("V", 0.0))
    K = float(theta.get("K", 0.0))
    # Effective critical coupling: perturbative shift from V, K
    lam_c_eff = 1.0 + 0.25 * V + 0.1 * K / J
    ratio = lam / lam_c_eff
    disc = max(1.0 - ratio * ratio, 0.0)
    e0 = -J * (1.0 + 0.5 * V) - J * lam * np.sqrt(disc + C.EPS_NUM) \
         - K * disc
    return float(e0)


def residual(theta_vec: np.ndarray,
              Ls: np.ndarray, lams: np.ndarray,
              E_target: np.ndarray,
              weights: np.ndarray) -> float:
    """Weighted least-squares residual."""
    theta = {"J": float(theta_vec[0]),
             "V": float(theta_vec[1]),
             "K": float(theta_vec[2])}
    r = 0.0
    for k in range(len(Ls)):
        e_model = effective_hamiltonian_energy(theta, int(Ls[k]), float(lams[k]))
        r += float(weights[k]) * (e_model - float(E_target[k])) ** 2
    return r


def residual_jacobian_fd(theta_vec: np.ndarray,
                           Ls: np.ndarray, lams: np.ndarray,
                           E_target: np.ndarray,
                           weights: np.ndarray,
                           h: float = 1.0e-5) -> np.ndarray:
    """Finite-difference Jacobian of the residual (used only if the
    user-supplied L-BFGS-B routine cannot compute gradients."""
    n = len(theta_vec)
    jac = np.zeros(n)
    f0 = residual(theta_vec, Ls, lams, E_target, weights)
    for i in range(n):
        tvp = theta_vec.copy()
        tvp[i] += h
        fp = residual(tvp, Ls, lams, E_target, weights)
        jac[i] = (fp - f0) / h
    return jac


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fit_effective_hamiltonian(Ls: np.ndarray, lams: np.ndarray,
                                E_target: np.ndarray,
                                weights: np.ndarray = None,
                                theta0: np.ndarray = None,
                                bounds=None,
                                max_iter: int = 300,
                                tol: float = 1.0e-8) -> Dict[str, object]:
    """Run L-BFGS-B to fit (J, V, K).

    Returns a dict with keys 'theta', 'fun', 'nfev', 'success', 'jac'.
    """
    n_data = len(lams)
    if weights is None:
        weights = np.ones(n_data) / max(n_data, 1)
    if theta0 is None:
        theta0 = np.array([1.0, 0.0, 0.0])
    if bounds is None:
        # J > 0, V in [-2, 2], K in [-2, 2]
        bounds = [(1.0e-3, 10.0), (-2.0, 2.0), (-2.0, 2.0)]

    def fun(x):
        return residual(x, Ls, lams, E_target, weights)

    def jac(x):
        return residual_jacobian_fd(x, Ls, lams, E_target, weights)

    res = minimize(fun, theta0, jac=jac, method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter, "ftol": tol,
                              "gtol": tol * 10.0})
    out = {
        "theta": np.asarray(res.x, dtype=float),
        "theta_dict": {"J": float(res.x[0]),
                        "V": float(res.x[1]),
                        "K": float(res.x[2])},
        "fun": float(res.fun),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "jac": jac(np.asarray(res.x)),
    }
    return out


# ---------------------------------------------------------------------------
# Cross-validation for model selection
# ---------------------------------------------------------------------------
def kfold_score(Ls: np.ndarray, lams: np.ndarray,
                  E_target: np.ndarray,
                  k_folds: int = 4,
                  seed: int = 0) -> float:
    """Run k-fold cross-validation for the (J, V, K) effective model.
    Returns the mean squared test residual."""
    rng = np.random.default_rng(seed)
    n = len(lams)
    idx = rng.permutation(n)
    fold_size = n // k_folds
    scores = []
    for kf in range(k_folds):
        test = idx[kf * fold_size: (kf + 1) * fold_size]
        train = np.concatenate([idx[:kf * fold_size],
                                  idx[(kf + 1) * fold_size:]])
        Ls_tr, lams_tr, E_tr = Ls[train], lams[train], E_target[train]
        Ls_te, lams_te, E_te = Ls[test], lams[test], E_target[test]
        fit = fit_effective_hamiltonian(Ls_tr, lams_tr, E_tr, max_iter=100)
        theta = fit["theta_dict"]
        pred = np.array([effective_hamiltonian_energy(theta, int(L), float(lam))
                           for L, lam in zip(Ls_te, lams_te)])
        scores.append(float(np.mean((pred - E_te) ** 2)))
    return float(np.mean(scores))
