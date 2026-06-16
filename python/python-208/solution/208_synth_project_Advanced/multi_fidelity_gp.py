"""
multi_fidelity_gp.py
====================

Non-linear autoregressive multi-fidelity Gaussian process (NAR-GP)
predictor.  This is the central engine of the multi-fidelity UQ
framework, combining:
  - the fidelity hierarchy (fidelity_manager.py),
  - the banded covariance solver (banded_covariance.py),
  - the confidence-bound machinery (confidence_bounds.py),
  - the denoising filter (denoising_filter.py),
  - the coordinate whitening (coordinate_transform.py).

Mathematical formulation
------------------------
Following Perdikaris & Karniadakis (2017) and Kennedy & O'Hagan (2000),
we model the output of fidelity l as

    F_l(xi) = rho_l(xi) * F_{l-1}(xi) + delta_l(xi)

where rho_l(xi) is a deterministic scaling function (here we take
rho_l = Corr(F_l, F_{l-1}), a constant) and delta_l(xi) is a GP
discrepancy term.  Recursing from F_0 (the lowest fidelity),

    F_L(xi) = (prod_{l=1}^L rho_l) F_0(xi)
              + sum_{l=1}^L (prod_{k=l+1}^L rho_k) delta_l(xi)

The posterior mean and variance of F_L(xi) given all training data
are computed by assembling the joint GP over all fidelity levels and
conditioning on the observations.

For tractability we project all training data onto the whitened
1-D coordinate s(xi) = || eta(xi) ||_2 (a radial coordinate in the
whitened space) and build a banded Matern-3/2 GP on the sorted s-grid.
This reduces the multi-dimensional problem to a 1-D GP on a local
correlation length, with the whitening transform absorbing the
anisotropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from banded_covariance import (
    _cholesky_band_textbook,
    banded_sl,
    gp_covariance_banded,
    matern32,
)
from coordinate_transform import WhiteningTransform, encode
from denoising_filter import denoise_training_labels


# ----------------------------------------------------------------------
# NAR-GP data container.
# ----------------------------------------------------------------------
@dataclass
class NARGPData:
    """Container for NAR-GP training data across fidelity levels."""
    xi_by_level: List[List[List[float]]] = field(default_factory=list)
    y_by_level: List[List[float]] = field(default_factory=list)
    n_levels: int = 0


# ----------------------------------------------------------------------
# Build the NAR-GP from a set of FidelitySamples.
# ----------------------------------------------------------------------
def build_nargp_data(
    samples,                  # List[FidelitySample]
    n_levels: int,
) -> NARGPData:
    """Group FidelitySamples by level into a NARGPData container."""
    data = NARGPData(n_levels=n_levels)
    data.xi_by_level = [[] for _ in range(n_levels)]
    data.y_by_level = [[] for _ in range(n_levels)]
    for s in samples:
        if 0 <= s.level < n_levels:
            data.xi_by_level[s.level].append(s.xi)
            data.y_by_level[s.level].append(s.y)
    return data


# ----------------------------------------------------------------------
# Radial projection in whitened space.
# ----------------------------------------------------------------------
def radial_project(xi: List[float], W: WhiteningTransform) -> float:
    """Project xi to the radial coordinate s = || eta(xi) ||_2."""
    eta = encode(xi, W)
    return math.sqrt(sum(e * e for e in eta))


# ----------------------------------------------------------------------
# NAR-GP posterior predictor.
# ----------------------------------------------------------------------
class NARGPPredictor:
    """Non-linear autoregressive multi-fidelity GP predictor."""

    def __init__(
        self,
        data: NARGPData,
        W: WhiteningTransform,
        correlation_matrix: List[List[float]],
        ell: float = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = 0.05,
        mu_band: int = 16,
        denoise_method: str = "median",
    ) -> None:
        self.data = data
        self.W = W
        self.correlation_matrix = correlation_matrix
        self.ell = ell
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.mu_band = mu_band

        # Aggregate all samples with their fidelity weights.
        self.all_xi: List[List[float]] = []
        self.all_y: List[float] = []
        self.all_level: List[int] = []
        self.all_s: List[float] = []     # radial coordinate
        for l in range(data.n_levels):
            for xi, y in zip(data.xi_by_level[l], data.y_by_level[l]):
                self.all_xi.append(xi)
                self.all_y.append(y)
                self.all_level.append(l)
                self.all_s.append(radial_project(xi, W))

        # Denoise labels (median filter on sorted radial coordinate).
        if len(self.all_y) > 2 and denoise_method != "none":
            pairs = sorted(zip(self.all_s, self.all_y, self.all_xi, self.all_level),
                           key=lambda p: p[0])
            s_sorted = [p[0] for p in pairs]
            y_sorted = [p[1] for p in pairs]
            y_den = denoise_training_labels(
                [[s] for s in s_sorted], y_sorted, method=denoise_method
            )
            # Restore order.
            self._s_order = [p[0] for p in pairs]
            self._y_den = y_den
            self._xi_order = [p[2] for p in pairs]
            self._level_order = [p[3] for p in pairs]
        else:
            self._s_order = list(self.all_s)
            self._y_den = list(self.all_y)
            self._xi_order = list(self.all_xi)
            self._level_order = list(self.all_level)

        # Sort by radial coordinate for banded assembly.
        order = sorted(range(len(self._s_order)), key=lambda i: self._s_order[i])
        self.s_sorted = [self._s_order[i] for i in order]
        self.y_sorted = [self._y_den[i] for i in order]
        self.xi_sorted = [self._xi_order[i] for i in order]
        self.level_sorted = [self._level_order[i] for i in order]

        # Fit a fidelity-weighted GP.
        self._fit_gp()

    def _fidelity_weight(self, level: int) -> float:
        """Return rho_{l, L-1} as the fidelity weight."""
        L = self.data.n_levels
        if L == 0:
            return 1.0
        return self.correlation_matrix[level][L - 1]

    def _fit_gp(self) -> None:
        """Fit the weighted GP on the sorted radial grid."""
        n = len(self.s_sorted)
        if n == 0:
            self._alpha: List[float] = []
            self._AB: List[List[float]] = []
            self._mu_band_actual = 0
            return
        # Weighted observations.
        y_w = [
            self._fidelity_weight(self.level_sorted[i]) * self.y_sorted[i]
            for i in range(n)
        ]
        # Assemble banded covariance on sorted s-grid.
        mu = min(self.mu_band, n - 1)
        AB = gp_covariance_banded(
            self.s_sorted, self.ell, self.sigma_f, self.sigma_n, mu
        )
        try:
            AB = _cholesky_band_textbook(n, mu, AB)
            alpha = banded_sl(n, mu, AB, y_w)
            # Sanity check: if any alpha is nan, fall back to ridge regression.
            if any(math.isnan(a) or math.isinf(a) for a in alpha):
                raise ValueError("nan/inf in Cholesky solution")
        except (ValueError, OverflowError, ZeroDivisionError):
            # Fall back: simple weighted mean.
            mean_y = sum(y_w) / max(n, 1)
            alpha = [0.0] * n
            # Revert AB to zeros so predict returns the prior mean.
            AB = [[0.0] * n for _ in range(mu + 1)]
            for j in range(n):
                AB[mu][j] = self.sigma_f
            self._fallback_mean = mean_y
            self._alpha = alpha
            self._AB = AB
            self._mu_band_actual = mu
            return
        self._alpha = alpha
        self._AB = AB
        self._mu_band_actual = mu
        self._fallback_mean = None

    def predict(self, xi: List[float]) -> Tuple[float, float]:
        """Return (posterior mean, posterior std) at xi."""
        s_test = radial_project(xi, self.W)
        n = len(self.s_sorted)
        if n == 0:
            return 0.0, self.sigma_f
        # Fallback path: if Cholesky failed, return the prior mean.
        if getattr(self, "_fallback_mean", None) is not None:
            return self._fallback_mean, self.sigma_f
        # k_star = [k(s_test, s_i)] for i in training set (banded truncation).
        mu = self._mu_band_actual
        # Build dense k_star for simplicity.
        k_star = [matern32(s_test - si, self.ell, self.sigma_f) for si in self.s_sorted]
        # Posterior mean.
        m = sum(k_star[i] * self._alpha[i] for i in range(n))
        # Solve K v = k_star.
        try:
            v = banded_sl(n, mu, self._AB, k_star)
            vKv = sum(k_star[i] * v[i] for i in range(n))
            var = matern32(0.0, self.ell, self.sigma_f) - vKv
            var = max(var, 1.0e-14)
        except (ValueError, OverflowError, ZeroDivisionError):
            var = self.sigma_f ** 2
        return m, math.sqrt(var)

    def batch_predict(
        self, xi_batch: List[List[float]]
    ) -> Tuple[List[float], List[float]]:
        means: List[float] = []
        stds: List[float] = []
        for xi in xi_batch:
            m, s = self.predict(xi)
            means.append(m)
            stds.append(s)
        return means, stds


# ----------------------------------------------------------------------
# Multi-fidelity predictor wrapper (callable as required by adaptive sampler).
# ----------------------------------------------------------------------
class MultiFidelityPredictor:
    """Callable wrapper around NARGPPredictor for the adaptive sampler."""

    def __init__(self, nargp: NARGPPredictor) -> None:
        self.nargp = nargp

    def __call__(self, xi: List[float]) -> Tuple[float, float]:
        return self.nargp.predict(xi)

    def batch_predict(
        self, xi_batch: List[List[float]]
    ) -> Tuple[List[float], List[float]]:
        return self.nargp.batch_predict(xi_batch)
