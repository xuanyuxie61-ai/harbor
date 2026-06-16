# -*- coding: utf-8 -*-
"""
Finite-size scaling analysis for the 1D TFIM quantum phase transition.

At the QCP lambda_c = 1 the correlation length diverges as
    xi ~ |lambda - lambda_c|^{-nu},     nu = 1.
On a finite lattice of linear size L the only relevant length is L
itself, so any singular observable O takes the FSS form
    O(L, lambda) = L^{kappa/nu} * f_O( (lambda - lambda_c) L^{1/nu} ).

We implement the following classical analyses (see Cardy, Privman,
and the modern reviews by Vojta 2003 and LO 2019):

  1. Binder cumulant crossing  ->  lambda_c
  2. Correlation-length ratio xi/L  ->  lambda_c
  3. Peak-shift method           ->  lambda_c(L) = lambda_c + a L^{-1/nu - omega}
  4. Data collapse (nu fit)
  5. Critical-exponent extraction  nu, beta, gamma  via log-log slopes

All routines accept the *raw* finite-L data from ``tfim_hamiltonian``
or ``mc_path_integral`` and are completely agnostic to the backend.
"""

from __future__ import annotations
from typing import Tuple, Callable, List
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Raw observables
# ---------------------------------------------------------------------------
def binder_cumulant(m_samples: np.ndarray) -> float:
    """Binder cumulant  U_L = 1 - <m^4> / (3 <m^4>_Gauss)
                     = 1 - <m^4> / (3 <m^2>^2).
    For the Ising class U* = 0.6107 at criticality in 2D; for the
    quantum 1D model with z = 1 the effective dimensionality is d+z = 2
    so the same universal value applies.
    """
    m2 = float(np.mean(m_samples ** 2))
    m4 = float(np.mean(m_samples ** 4))
    denom = 3.0 * m2 * m2
    if abs(denom) < C.EPS_NUM:
        return 0.0
    return 1.0 - m4 / denom


def binder_u_from_moments(m2: float, m4: float) -> float:
    """Deterministic variant of ``binder_cumulant`` given raw moments."""
    denom = 3.0 * m2 * m2
    if abs(denom) < C.EPS_NUM:
        return 0.0
    return 1.0 - m4 / denom


def correlation_length_from_corr(corr_r: np.ndarray, rs: np.ndarray) -> float:
    """Estimate xi from the long-distance exponential decay of the
    correlator C(r).  We fit
        log C(r) = a - r / xi
    by weighted linear regression over the tail half of the data.
    """
    n = len(corr_r)
    if n < 4:
        return np.nan
    half = n // 2
    # Use only the positive, not-too-small tail to avoid log of noise.
    mask = corr_r[half:] > C.SAFE_LOG_FLOOR
    if mask.sum() < 2:
        return np.nan
    r_tail = rs[half:][mask]
    logc = np.log(corr_r[half:][mask])
    # Linear regression log c = a - r / xi  ->  slope = -1/xi
    A = np.column_stack([np.ones_like(r_tail), r_tail])
    coeff, *_ = np.linalg.lstsq(A, logc, rcond=None)
    slope = float(coeff[1])
    if slope >= 0.0:
        return np.nan
    return -1.0 / slope


# ---------------------------------------------------------------------------
# Crossing-point analysis
# ---------------------------------------------------------------------------
def crossing_point(lams: np.ndarray, ul_L1: np.ndarray,
                    ul_L2: np.ndarray) -> float:
    """Linear-interpolation estimate of the crossing of U_{L1}(lambda)
    and U_{L2}(lambda).  Returns np.nan if no crossing is found.
    """
    diff = ul_L1 - ul_L2
    # Look for a sign change
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    # linear interpolation
    d0, d1 = diff[i], diff[i + 1]
    t = d0 / (d0 - d1 + C.EPS_NUM)
    return float(lams[i] + t * (lams[i + 1] - lams[i]))


# ---------------------------------------------------------------------------
# Peak-shift method
# ---------------------------------------------------------------------------
def peak_location(lams: np.ndarray, ys: np.ndarray) -> float:
    """Return the location of the maximum of y(lambda) by cubic
    interpolation around the discrete maximum."""
    i = int(np.argmax(ys))
    if i == 0 or i == len(ys) - 1:
        return float(lams[i])
    # Fit a cubic to (i-1, i, i+1, i+2) if available else 3 points
    js = [i - 1, i, i + 1]
    ls = lams[js]
    vs = ys[js]
    # Lagrange cubic on a non-uniform grid:
    lam_star = ls[1]
    num = 0.0
    den = 0.0
    for a in range(3):
        prod_num = 1.0
        prod_den = 1.0
        for b in range(3):
            if b == a:
                continue
            prod_num *= (ls[a] - ls[b])
            prod_den *= (ls[a] - ls[b])
        # derivative of Lagrange basis at ls[1]:
        other = [c for c in range(3) if c != a]
        d = 0.0
        for o in other:
            p = 1.0
            for c in other:
                if c != o:
                    p *= (ls[1] - ls[c])
            d += p
        num += vs[a] * d / prod_den
        den += 0  # not used
    # Use numpy polyfit for robustness
    p = np.polyfit(ls, vs, 2)
    if p[0] >= 0:
        return float(ls[1])
    return float(-p[1] / (2 * p[0]))


def peak_shift_fit(Ls: np.ndarray, lam_peaks: np.ndarray,
                    nu_guess: float = 1.0) -> Tuple[float, float]:
    """Fit  lambda_peak(L) = lambda_c + a * L^{-1/nu - omega}
    with omega = 1 (correction-to-scaling exponent for 1D TFIM).
    Returns (lambda_c, a).
    """
    if len(Ls) < 3:
        return float("nan"), float("nan")
    x = np.log(Ls.astype(float))
    y = lam_peaks
    # Linearise by a first-order Taylor around the guess.
    exp = -1.0 / nu_guess - 1.0
    A = np.column_stack([np.ones_like(x), Ls.astype(float) ** exp])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    lam_c = float(coeff[0])
    a = float(coeff[1])
    return lam_c, a


# ---------------------------------------------------------------------------
# Data collapse
# ---------------------------------------------------------------------------
def data_collapse_residual(params: np.ndarray, Ls: np.ndarray,
                            lams: np.ndarray,
                            obs_by_L: List[np.ndarray],
                            kappa_nu: float) -> float:
    """Compute the variance of a polynomial fit to the rescaled curves
    O_i(L) vs x_i = (lambda - lambda_c) L^{1/nu} after vertical
    rescaling L^{-kappa/nu}.  Smaller -> better collapse.
    """
    lam_c, inv_nu = float(params[0]), float(params[1])
    xs_all: List[np.ndarray] = []
    ys_all: List[np.ndarray] = []
    for L, obs in zip(Ls, obs_by_L):
        x = (lams - lam_c) * (L ** inv_nu)
        y = obs * (L ** (-kappa_nu))
        xs_all.append(x)
        ys_all.append(y)
    x_all = np.concatenate(xs_all)
    y_all = np.concatenate(ys_all)
    # Polynomial of degree 4 reference curve
    try:
        p = np.polyfit(x_all, y_all, 4)
    except np.linalg.LinAlgError:
        return 1.0e30
    residual = 0.0
    for x, y in zip(xs_all, ys_all):
        residual += float(np.sum((y - np.polyval(p, x)) ** 2))
    return residual


# ---------------------------------------------------------------------------
# Critical-exponent extraction
# ---------------------------------------------------------------------------
def exponent_from_slope(Ls: np.ndarray, obs_at_lc: np.ndarray) -> float:
    """If an observable scales as O(L, lambda_c) ~ L^{kappa/nu}, then
        log O = (kappa/nu) log L + const.
    Returns the estimated ratio kappa/nu by linear regression.
    """
    mask = obs_at_lc > C.SAFE_LOG_FLOOR
    if mask.sum() < 2:
        return float("nan")
    x = np.log(Ls.astype(float)[mask])
    y = np.log(obs_at_lc[mask])
    p = np.polyfit(x, y, 1)
    return float(p[0])


def fidelity_susceptibility_log_coeff(Ls: np.ndarray,
                                        chi_fs: np.ndarray) -> float:
    """At the QCP the fidelity susceptibility diverges as
        chi_F(L) = g * log(L) + const.
    Returns the coefficient g (should be proportional to the central
    charge c = 1/2 for the Ising QCP via chi_F = (c/8) log L + ...).
    """
    p = np.polyfit(np.log(Ls.astype(float)), chi_fs, 1)
    return float(p[0])


# ---------------------------------------------------------------------------
# Entanglement entropy (finite-size log scaling)
# ---------------------------------------------------------------------------
def entanglement_entropy_cft(L: int, x_cut: int, c: float = 0.5) -> float:
    """Calabrese-Cardy formula for the ground-state entanglement entropy
    of a contiguous block of length ``x_cut`` inside a chain of length L
    with open boundary conditions:

        S = (c/3) log( (L / pi) sin(pi x_cut / L) ) + c1

    We return the scale-dependent part (dropping the non-universal c1).
    """
    if x_cut <= 0 or x_cut >= L:
        return 0.0
    arg = (L / C.PI) * np.sin(C.PI * x_cut / L)
    if arg <= 0.0:
        return 0.0
    return (c / 3.0) * float(np.log(arg))
