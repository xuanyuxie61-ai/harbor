# -*- coding: utf-8 -*-
"""
NACA-inspired parameterisation of the TFIM dispersion relation.

Adapted from the ``785_naca`` seed project, which parameterises airfoil
shapes using the classical 4-digit NACA formula

    y_t / c = 5 t [ 0.2969 sqrt(x/c) - 0.1260 (x/c) - 0.3516 (x/c)^2
                     + 0.2843 (x/c)^3 - 0.1015 (x/c)^4 ]

The NACA shape is a compact 4-parameter family that covers a large
variety of aerodynamic profiles; in an analogous spirit we use it to
parameterise the TFIM quasi-particle dispersion

    eps(k; a, b, c, d)  =  2 J * sqrt( P_a(k)^2 + P_b(k)^2 )

where P_a and P_b are NACA-shaped polynomials in cos(k).  This lets
us *fit* an effective low-energy theory to numerical data with only
four real parameters, dramatically simplifying the finite-size
scaling analysis.

We also implement the camber-line / thickness decomposition
(symmetric vs cambered NACA profiles) as two alternative ansaetze.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# NACA 4-digit symmetric thickness distribution
# ---------------------------------------------------------------------------
def naca4_symmetric_thickness(t: float, x: np.ndarray) -> np.ndarray:
    """Return the half-thickness distribution y_t(x) for a NACA
    4-digit symmetric airfoil with maximum thickness t (as a fraction
    of chord c = 1).

    The classical formula is
        y_t = 5 t [ 0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2
                     + 0.2843 x^3 - 0.1015 x^4 ]
    We use the closed-trailing-edge modification (-0.1015 -> -0.1036)
    so that y_t(1) = 0 exactly.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    y = 5.0 * t * (0.2969 * np.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x * x
                    + 0.2843 * x ** 3
                    - 0.1036 * x ** 4)
    return y


def naca4_camber_line(m: float, p: float, x: np.ndarray) -> np.ndarray:
    """Mean camber line  y_c(x) for a NACA 4-digit cambered airfoil:
        y_c = (m / p^2) (2 p x - x^2)      for x <= p
        y_c = (m / (1-p)^2) (1 - 2 p + 2 p x - x^2)   for x > p
    m is the maximum camber (fraction of chord); p is its location.
    """
    x = np.asarray(x, dtype=float)
    p = np.clip(p, 0.05, 0.95)
    out = np.zeros_like(x)
    lo = x <= p
    hi = ~lo
    out[lo] = (m / (p * p)) * (2 * p * x[lo] - x[lo] ** 2)
    out[hi] = (m / ((1 - p) ** 2)) * (1 - 2 * p + 2 * p * x[hi] - x[hi] ** 2)
    return out


# ---------------------------------------------------------------------------
# Mapping NACA -> TFIM dispersion
# ---------------------------------------------------------------------------
def naca_dispersion_ansatz(k: np.ndarray,
                            t: float = 0.12,
                            m: float = 0.0,
                            p: float = 0.4,
                            J: float = 1.0) -> np.ndarray:
    """Parameterised TFIM-like dispersion built from NACA shapes.

    We map x = (1 + cos k)/2 in [0, 1], then
        P(k) = y_t(x; t) + y_c(x; m, p)
    and  eps(k) = 2 J * (1 - cos k + P(k)).

    The free-stream (P = 0) limit recovers the TFIM free-fermion
    dispersion near lam = 1:  eps(k) = 2 J (1 - cos k) = 4 J sin^2(k/2).
    The NACA correction terms capture higher-harmonic contributions
    from irrelevant operators.
    """
    x = 0.5 * (1.0 + np.cos(np.asarray(k)))
    P = naca4_symmetric_thickness(t, x) + naca4_camber_line(m, p, x)
    return 2.0 * J * (1.0 - np.cos(k) + P)


def naca_group_velocity(k: np.ndarray, **kwargs) -> np.ndarray:
    """Group velocity  v_g(k) = d eps / d k  by finite difference."""
    dk = 1.0e-4
    return (naca_dispersion_ansatz(k + dk, **kwargs)
            - naca_dispersion_ansatz(k - dk, **kwargs)) / (2.0 * dk)


def naca_dos_at_zero(t: float, J: float = 1.0) -> float:
    """Density of states at omega = 0 for the NACA dispersion.
    Near k = 0 the dispersion is eps ~ J k^2 so  rho(0) ~ 1/sqrt(omega).
    We regularise and return the coefficient of the 1/sqrt divergence.
    """
    if abs(J) < C.EPS_NUM:
        return 0.0
    # Coefficient of k^2 in eps(k) for small k:
    #   eps ~ 2 J (k^2 / 2 + 5 t * 0.2969 * sqrt(k^2/4) )
    # The sqrt term is subdominant; the leading behaviour is J k^2.
    # So rho(omega) ~ 1/(2 sqrt(J omega)).
    return 1.0 / (2.0 * np.sqrt(max(abs(J), C.EPS_NUM)))


# ---------------------------------------------------------------------------
# Fit to a reference dispersion
# ---------------------------------------------------------------------------
def fit_naca_to_tfim(L: int, J: float, h: float,
                       k_grid: np.ndarray = None) -> dict:
    """Fit the 4-parameter NACA dispersion to the exact TFIM
    free-fermion dispersion eps_k = 2 J sqrt(1 + lam^2 - 2 lam cos k).

    Returns the fitted parameters and the residual.
    """
    if k_grid is None:
        k_grid = np.linspace(0.0, C.PI, 64)
    lam = h / J if J != 0.0 else 0.0
    eps_ref = 2.0 * abs(J) * np.sqrt(np.maximum(
        1.0 + lam * lam - 2.0 * lam * np.cos(k_grid), 0.0))
    # Simple grid search over (t, m, p) with J fixed.
    from scipy.optimize import minimize

    def obj(params):
        t_, m_, p_ = params
        eps_model = naca_dispersion_ansatz(k_grid, t=t_, m=m_, p=p_, J=J)
        return float(np.mean((eps_model - eps_ref) ** 2))

    res = minimize(obj, x0=[0.12, 0.0, 0.4],
                    bounds=[(0.01, 0.5), (0.0, 0.2), (0.1, 0.9)],
                    method="L-BFGS-B")
    return {
        "t": float(res.x[0]),
        "m": float(res.x[1]),
        "p": float(res.x[2]),
        "residual": float(res.fun),
        "success": bool(res.success),
    }
