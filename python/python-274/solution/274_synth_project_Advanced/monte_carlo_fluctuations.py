# -*- coding: utf-8 -*-
"""
monte_carlo_fluctuations.py
---------------------------
Monte-Carlo sampling of thermal and quantum fluctuations in the electron-
phonon coupling constant lambda and in the Coulomb pseudo-potential mu*.
Used to estimate the uncertainty on the predicted Tc.

Scientific origin of the fused algorithms
-----------------------------------------
* Dice simulation   (seed project 277_dice_simulation)
    -> repeated sampling of discrete random variables and construction of
       the resulting distribution / histogram.
    -> adapted: each "throw" samples a (lambda, mu*) pair from a joint
       distribution and evaluates Tc through the McMillan / Allen-Dynes
       formula.  The histogram of Tc values gives the prediction interval.

Core physics / mathematics
--------------------------
* McMillan formula (modified by Allen-Dynes):
        Tc = (omega_log / 1.2) * exp(
              -1.04 (1 + lambda) / (lambda - mu* (1 + 0.62 lambda))
        )
* Allen-Dynes correction for strong coupling:
        f_1 = [1 + (lambda / lam_1)^{3/2}]^{1/3}
        f_2 = 1 + (omega_2 / omega_log - 1) lambda^2 /
                  (lambda^2 + lam_2^2)
        Tc_AD = (omega_log / 1.37) * f_1 * f_2 *
                exp(-1.04 (1 + lambda) / (lambda - mu* (1 + 0.62 lambda)))
* Each Monte-Carlo sample draws
        lambda ~ N(lambda_0, sigma_lambda^2)   (truncated at lambda > 0)
        mu*    ~ N(mu0, sigma_mu^2)            (truncated at 0 < mu* < 0.5)
  and evaluates Tc.

Stability / boundary notes
--------------------------
* Samples with  lambda - mu* (1 + 0.62 lambda) <= 0  are rejected (unphysical
  denominator in McMillan formula).
* Random seed is explicit so the histogram is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  McMillan / Allen-Dynes formulas
# ---------------------------------------------------------------------------
def mcmillan_tc(
    lambda_ep: float,
    mu_star: float,
    omega_log: float,
) -> float:
    """McMillan formula for Tc (in the same units as omega_log).

    Returns 0.0 if the denominator is non-positive (unphysical regime).
    """
    if lambda_ep <= 0.0:
        return 0.0
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if denom <= 0.0:
        return 0.0
    arg = -1.04 * (1.0 + lambda_ep) / denom
    return (omega_log / 1.2) * math.exp(arg)


def allen_dynes_tc(
    lambda_ep: float,
    mu_star: float,
    omega_log: float,
    omega_2: float,
    lam_1: float = 2.46,
    lam_2: float = 0.62,
) -> float:
    """Allen-Dynes modification of the McMillan formula."""
    if lambda_ep <= 0.0:
        return 0.0
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if denom <= 0.0:
        return 0.0
    f1 = (1.0 + (lambda_ep / lam_1) ** 1.5) ** (1.0 / 3.0)
    ratio = omega_2 / max(omega_log, 1e-300)
    f2 = 1.0 + (ratio - 1.0) * lambda_ep ** 2 / (lambda_ep ** 2 + lam_2 ** 2)
    arg = -1.04 * (1.0 + lambda_ep) / denom
    return (omega_log / 1.37) * f1 * f2 * math.exp(arg)


# ---------------------------------------------------------------------------
# 2.  Monte-Carlo engine  (adapted from seed project 277)
# ---------------------------------------------------------------------------
@dataclass
class McTcResult:
    """Result of a Monte-Carlo Tc sampling campaign."""
    lambda_samples: np.ndarray    # (N,) sampled lambda values
    mu_samples: np.ndarray        # (N,) sampled mu* values
    tc_samples: np.ndarray        # (N,) resulting Tc values
    tc_mean: float
    tc_std: float
    tc_median: float
    tc_5pct: float                # 5-th percentile
    tc_95pct: float               # 95-th percentile
    n_rejected: int               # samples rejected (unphysical denominator)
    histogram_bins: np.ndarray    # (n_bins+1,) bin edges
    histogram_counts: np.ndarray  # (n_bins,) counts


def monte_carlo_tc(
    lambda_0: float,
    sigma_lambda: float,
    mu_0: float,
    sigma_mu: float,
    omega_log: float,
    omega_2: float,
    n_samples: int = 50000,
    n_bins: int = 80,
    seed: int = 274,
    use_allen_dynes: bool = True,
) -> McTcResult:
    """Sample the Tc distribution by Monte-Carlo.

    Each "throw" draws (lambda, mu*) from truncated Gaussians and evaluates
    the McMillan / Allen-Dynes formula.
    """
    if n_samples <= 0:
        raise ValueError("monte_carlo_tc: n_samples must be positive")
    if sigma_lambda <= 0.0 or sigma_mu <= 0.0:
        raise ValueError("monte_carlo_tc: sigmas must be positive")
    if omega_log <= 0.0:
        raise ValueError("monte_carlo_tc: omega_log must be positive")

    rng = np.random.default_rng(seed)

    def tc_eval(lam_val, mu_val):
        if use_allen_dynes:
            return allen_dynes_tc(lam_val, mu_val, omega_log, omega_2)
        return mcmillan_tc(lam_val, mu_val, omega_log)

    lam_s = np.zeros(n_samples)
    mu_s = np.zeros(n_samples)
    tc_s = np.zeros(n_samples)
    n_rejected = 0
    i = 0
    while i < n_samples:
        # Rejection sampling: truncate to lambda > 0, 0 < mu* < 0.5
        l_cand = rng.normal(lambda_0, sigma_lambda)
        m_cand = rng.normal(mu_0, sigma_mu)
        if l_cand <= 0.0 or m_cand <= 0.0 or m_cand >= 0.5:
            n_rejected += 1
            continue
        denom = l_cand - m_cand * (1.0 + 0.62 * l_cand)
        if denom <= 0.0:
            n_rejected += 1
            continue
        tc_val = tc_eval(l_cand, m_cand)
        if tc_val <= 0.0 or not math.isfinite(tc_val):
            n_rejected += 1
            continue
        lam_s[i] = l_cand
        mu_s[i] = m_cand
        tc_s[i] = tc_val
        i += 1

    tc_finite = tc_s[tc_s > 0]
    if tc_finite.size == 0:
        raise RuntimeError("monte_carlo_tc: all samples were rejected")

    counts, edges = np.histogram(tc_finite, bins=n_bins)
    return McTcResult(
        lambda_samples=lam_s,
        mu_samples=mu_s,
        tc_samples=tc_s,
        tc_mean=float(tc_finite.mean()),
        tc_std=float(tc_finite.std()),
        tc_median=float(np.median(tc_finite)),
        tc_5pct=float(np.percentile(tc_finite, 5)),
        tc_95pct=float(np.percentile(tc_finite, 95)),
        n_rejected=n_rejected,
        histogram_bins=edges,
        histogram_counts=counts,
    )


# ---------------------------------------------------------------------------
# 3.  Histogram formatting (textual, no plotting)
# ---------------------------------------------------------------------------
def histogram_text(counts: np.ndarray, edges: np.ndarray, width: int = 60) -> str:
    """Return a textual representation of the histogram."""
    mx = counts.max() if counts.size > 0 else 1
    mx = max(mx, 1)
    lines = []
    for i in range(counts.size):
        bar = "#" * int(width * counts[i] / mx)
        lines.append(f"[{edges[i]:8.3f}, {edges[i+1]:8.3f})  {counts[i]:5d}  {bar}")
    return "\n".join(lines)
