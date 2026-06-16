# -*- coding: utf-8 -*-
"""
turbulence_diagnostics.py
=========================

Post-processing and diagnostic suite for the gyrokinetic turbulence
simulation.  This module is the "glue" that ties together the numerical
fields produced by the solver and turns them into physically meaningful
quantities:

    * Savitzky-Golay smoothing of 1-D and 2-D turbulent fields
      (port of ``SavitzkyGolay.sgolay2d`` from Richards et al.)
    * stochastic noise injection for synthetic diagnostic modelling
      (port of ``brc_data`` -- random sampling with Gaussian noise)
    * flux PDF and histogram analysis for intermittency characterisation
      (port of ``histogram_display``)
    * zonal / flux-surface averaging (port of ``compute_regional_mean``
      from ``Eddien826_SA``)
    * turbulent heat-flux time series and flux balance diagnostics

The turbulent heat flux  Q  is the moment

    Q = < integral d^3 v  (m v^2 / 2) v_E . grad x  delta f >

where  v_E = (c/B) b0 x grad phi  is the  E x B  drift and  <...>  denotes
an average over the flux surface.  In the slab reduction with
phi(x, y, z) = sum_k phi_k(x) exp(i k_y y + i k_z z)  this reduces to

    Q = sum_k  k_y  Im( phi_k*  Gamma_k )

with  Gamma_k  the gyrokinetic density moment of mode  k.

Intermittency
-------------
The PDF of  Q(t)  exhibits heavy tails that are well-modelled by a
log-normal or a stretched-exponential distribution

    P(|Q|) ~ exp(-(|Q| / Q_0)^alpha)      alpha in (0.4, 0.8)

characteristic of avaloid transport events.  We estimate alpha by a
least-squares fit in log-log space.

References:
    [1] Richards et al., ``SavitzkyGolay.py`` -- 2-D SG filter.
    [2] Burkardt, ``brc_data`` / ``histogram_display``.
    [3] Chen et al. (Eddien826), regional mean operator.
    [4] Terry, Rev. Mod. Phys. 88, 021003 (2016) -- turbulence review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ============================================================================
# Savitzky-Golay filter -- port of SavitzkyGolay.sgolay2d
# ============================================================================
def sgolay1d(y: np.ndarray, window_size: int = 7, order: int = 3,
             derivative: int = 0) -> np.ndarray:
    """1-D Savitzky-Golay smoothing (or differentiation).

    For each window of length ``window_size`` a polynomial of degree
    ``order`` is fitted by least squares.  The smoothed value is the
    polynomial evaluated at the window centre.

    Parameters
    ----------
    y : 1-D array
    window_size : odd positive integer
    order : polynomial degree < window_size
    derivative : 0 = smoothed values, 1 = first derivative, ...
    """
    if window_size % 2 == 0:
        raise ValueError("sgolay1d: window_size must be odd")
    if window_size < order + 1:
        raise ValueError("sgolay1d: window too small for order")
    half = window_size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    # Vandermonde
    A = np.vander(x, N=order + 1, increasing=True)
    # Moore-Penrose pseudo-inverse
    coef = np.linalg.pinv(A)        # (order+1, window_size)
    # extract the row corresponding to the requested derivative
    # d^k/dx^k x^m = m! / (m-k)! x^{m-k}
    dcoef = np.zeros(order + 1, dtype=np.float64)
    if derivative <= order:
        dcoef[derivative] = math.factorial(derivative)
    # convolution weights
    weights = dcoef @ coef
    # pad with reflection
    y_pad = np.pad(y, (half, half), mode="reflect")
    out = np.zeros_like(y)
    for i in range(y.size):
        out[i] = np.dot(weights, y_pad[i:i + window_size])
    return out


def sgolay2d(z: np.ndarray, window_size: int = 5, order: int = 2,
             derivative: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """2-D Savitzky-Golay filter.  Direct port of ``SavitzkyGolay.sgolay2d``.

    Parameters
    ----------
    z : 2-D array
    derivative : (dx, dy) -- if None returns smoothed field,
                 (1, 0) returns dz/dx, (0, 1) returns dz/dy,
                 (1, 1) returns d^2 z / dx dy, etc.
    """
    if window_size % 2 == 0:
        raise ValueError("sgolay2d: window_size must be odd")
    n_terms = (order + 1) * (order + 2) // 2
    if window_size ** 2 < n_terms:
        raise ValueError("sgolay2d: order too high for window size")
    half = window_size // 2
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]
    ind = np.arange(-half, half + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, (window_size, 1)).reshape(window_size ** 2)
    A = np.empty((window_size ** 2, len(exps)), dtype=np.float64)
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])
    # pad with reflection
    new_shape = (z.shape[0] + 2 * half, z.shape[1] + 2 * half)
    Z = np.zeros(new_shape, dtype=np.float64)
    Z[half:-half, half:-half] = z
    # top/bottom/left/right bands
    for i in range(half):
        Z[i, half:-half] = z[half - i - 1, :]
        Z[-(i + 1), half:-half] = z[-(half - i), :]
    for j in range(half):
        Z[:, j] = Z[:, 2 * half - j]
        Z[:, -(j + 1)] = Z[:, -(2 * half - j + 1)]
    # corners: average of the two adjacent edges
    Z[:half, :half] = Z[half, half]
    Z[:half, -half:] = Z[half, -half - 1]
    Z[-half:, :half] = Z[-half - 1, half]
    Z[-half:, -half:] = Z[-half - 1, -half - 1]

    coef = np.linalg.pinv(A)
    out = np.zeros_like(z, dtype=np.float64)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            patch = Z[i:i + window_size, j:j + window_size].ravel()
            coeffs = coef @ patch
            # derivative selection
            if derivative is None:
                out[i, j] = coeffs[0]
            else:
                dxo, dyo = derivative
                # coefficient index for x^dxo y^dyo
                idx = exps.index((dxo, dyo))
                val = coeffs[idx] * math.factorial(dxo) * math.factorial(dyo)
                out[i, j] = val
    return out


# ============================================================================
# Stochastic noise injection -- port of brc_data
# ============================================================================
@dataclass
class SyntheticDiagnostic:
    """Generate noisy synthetic measurements of a clean signal.

    Port of ``brc_data``: we treat each "flux-surface label" like a city
    and each "measurement" as a noisy sample  T = T_eq + sigma * N(0, 1).

    This models a suite of turbulent-heat-flux probes distributed around
    the torus, each with its own calibration offset (the "city mean")
    and Gaussian random noise.
    """

    n_channels: int = 16
    noise_level: float = 0.05
    seed: int = 42
    channel_means: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.channel_means is None:
            rng = np.random.default_rng(self.seed)
            self.channel_means = rng.normal(0.0, 0.1, self.n_channels)

    def sample(self, clean_signal: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Return a noisy measurement of the given 1-D clean signal.

        clean_signal : (N,) array -- the "true" heat-flux time series.
        returns        : (N, n_channels) array of channel measurements.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        N = clean_signal.size
        out = np.zeros((N, self.n_channels), dtype=np.float64)
        for c in range(self.n_channels):
            out[:, c] = (clean_signal + self.channel_means[c]
                         + self.noise_level * rng.standard_normal(N))
        return out


# ============================================================================
# Regional / flux-surface averaging -- port of Eddien826_SA
# ============================================================================
def regional_mean_2d(field: np.ndarray,
                     axis: int = 0,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted average of a 2-D periodic field along one axis.

    In toroidal geometry the flux-surface average is

        <Q>(r) = (1 / 2 pi) int_0^{2 pi} Q(r, theta) d theta

    with an optional cos(theta) or sin(theta) weighting for the odd
    moments.  This is the analogue of the latitudinal mean used in the
    climate analysis of ``Eddien826_SA``.

    Parameters
    ----------
    field   : (N_r, N_theta) array
    axis    : axis to average over (default 0 = flux surfaces)
    weights : (N_theta,) optional weights
    """
    if field.ndim != 2:
        raise ValueError("regional_mean_2d: field must be 2-D")
    if weights is None:
        return field.mean(axis=axis)
    if weights.size != field.shape[1 if axis == 0 else 0]:
        raise ValueError("regional_mean_2d: weights length mismatch")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if axis == 0:
        return np.tensordot(w, field, axes=([0], [1]))
    else:
        return np.tensordot(field, w, axes=([1], [0]))


def zonal_average(field: np.ndarray, Lx: float = 1.0) -> np.ndarray:
    """Zonal (y-averaged) component of a 2-D field phi(x, y).

    The zonal flow is  phi_zonal(x) = (1 / Ly) int_0^Ly phi(x, y) dy.
    """
    if field.ndim != 2:
        raise ValueError("zonal_average: field must be 2-D")
    return field.mean(axis=1)


# ============================================================================
# Turbulent heat-flux PDF and intermittency
# ============================================================================
@dataclass
class FluxPDF:
    """Statistical characterisation of the turbulent heat-flux time series."""
    Q: np.ndarray            # 1-D time series
    n_bins: int = 50

    @property
    def mean(self) -> float:
        return float(self.Q.mean())

    @property
    def std(self) -> float:
        return float(self.Q.std())

    @property
    def skewness(self) -> float:
        s = self.std
        if s <= 0:
            return 0.0
        return float(((self.Q - self.mean) ** 3).mean() / s ** 3)

    @property
    def kurtosis(self) -> float:
        s = self.std
        if s <= 0:
            return 0.0
        return float(((self.Q - self.mean) ** 4).mean() / s ** 4)

    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bin_edges, counts) of |Q| (positive tail)."""
        q = np.abs(self.Q)
        q = q[q > 0]
        if q.size == 0:
            return np.zeros(1), np.zeros(1)
        counts, edges = np.histogram(q, bins=self.n_bins, density=True)
        return edges, counts

    def stretched_exponential_fit(self) -> Tuple[float, float]:
        """Fit  log P(Q) = a - b |Q|^alpha  in the tail.

        We use a log-log fit: log(-log P) = log b + alpha log|Q|.
        Returns (alpha, b).  Uses the upper-decile tail of |Q|.
        """
        q = np.abs(self.Q)
        q = q[q > 0]
        if q.size < 30:
            return (1.0, 1.0)
        # Empirical CDF on a log grid
        edges, counts = self.histogram()
        mid = 0.5 * (edges[:-1] + edges[1:])
        # Survival function
        Q_sorted = np.sort(q)
        N = q.size
        # evaluate P(|Q| > x) at each mid-point
        P = np.array([float(np.sum(q > m)) / N for m in mid])
        mask = (P > 0) & (P < 1.0) & (mid > 0)
        if mask.sum() < 4:
            return (1.0, 1.0)
        # log-log fit: log(-log P) vs log(Q)
        x = np.log(mid[mask])
        y = np.log(np.maximum(-np.log(np.maximum(P[mask], 1.0e-30)), 1.0e-30))
        # remove any non-finite
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 3:
            return (1.0, 1.0)
        A = np.column_stack([np.ones_like(x[good]), x[good]])
        coef, _, _, _ = np.linalg.lstsq(A, y[good], rcond=None)
        alpha = float(coef[1])
        b = float(np.exp(coef[0]))
        # sanity clamp
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 1.0
        if not np.isfinite(b) or b <= 0:
            b = 1.0
        return alpha, b


# ============================================================================
# Full-diagnostics entry point
# ============================================================================
def run_diagnostics(
    phi_time_series: np.ndarray,
    Q_time_series: np.ndarray,
    Lx: float = 1.0,
    smooth_window: int = 7,
    smooth_order: int = 3,
) -> dict:
    """Apply the full diagnostic suite to a simulation snapshot.

    Parameters
    ----------
    phi_time_series : (Nt, Nx) zonal-flow potential history
    Q_time_series   : (Nt,) turbulent heat-flux time series

    Returns a dict with
        - smoothed_phi          : (Nt, Nx) SG-filtered potential
        - noisy_Q               : (Nt, Nch) synthetic diagnostic
        - zonal_phi             : (Nt,) zonal average
        - flux_pdf              : FluxPDF instance
        - stretched_exp_fit     : (alpha, b)
        - heat_flux_stats       : dict with mean, std, skew, kurt
    """
    # 2-D SG on phi
    if phi_time_series.ndim == 2 and phi_time_series.shape[1] > smooth_window:
        try:
            smoothed_phi = sgolay2d(phi_time_series, window_size=smooth_window,
                                     order=smooth_order)
        except Exception:
            smoothed_phi = phi_time_series
    else:
        smoothed_phi = phi_time_series
    # zonal average
    zonal = zonal_average(phi_time_series) if phi_time_series.ndim == 2 else phi_time_series
    # synthetic noise
    sd = SyntheticDiagnostic(n_channels=8, noise_level=0.03, seed=0)
    noisy = sd.sample(Q_time_series)
    # flux PDF
    pdf = FluxPDF(Q_time_series)
    alpha, b = pdf.stretched_exponential_fit()
    stats = {
        "mean": pdf.mean,
        "std": pdf.std,
        "skewness": pdf.skewness,
        "kurtosis": pdf.kurtosis,
    }
    return {
        "smoothed_phi": smoothed_phi,
        "noisy_Q": noisy,
        "zonal_phi": zonal,
        "flux_pdf": pdf,
        "stretched_exp_fit": (alpha, b),
        "heat_flux_stats": stats,
    }


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 61)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    ys = sgolay1d(y, window_size=7, order=3)
    print("SG1D: max |y_smooth - sin(x)|:", np.max(np.abs(ys - np.sin(x))))
    # 2-D
    z = np.outer(np.sin(x), np.cos(x)) + 0.05 * rng.standard_normal((x.size, x.size))
    zs = sgolay2d(z, window_size=5, order=2)
    print("SG2D: max |z_smooth - z_exact|:", np.max(np.abs(zs - np.outer(np.sin(x), np.cos(x)))))
    # synthetic
    sd = SyntheticDiagnostic(n_channels=4, noise_level=0.02)
    clean = np.sin(np.linspace(0, 5, 100))
    noisy = sd.sample(clean)
    print("synthetic: shape =", noisy.shape, " channel rms =",
          np.std(noisy - clean[:, None], axis=0).mean())
    # regional mean
    F = rng.standard_normal((10, 20))
    print("regional mean along axis 0:", regional_mean_2d(F, axis=0).shape)
    # PDF
    Q = rng.standard_normal(2000)
    pdf = FluxPDF(Q, n_bins=30)
    print("kurtosis of Gaussian ~", round(pdf.kurtosis, 3))
    # diagnostics
    phi = np.outer(np.exp(-np.linspace(0, 5, 20)), np.sin(np.linspace(0, 2 * np.pi, 40)))
    Q_ts = np.sin(np.linspace(0, 10, 500)) + 0.3 * rng.standard_normal(500)
    d = run_diagnostics(phi, Q_ts)
    print("diag smoothed_phi shape:", d["smoothed_phi"].shape)
    print("heat_flux_stats:", {k: round(v, 3) for k, v in d["heat_flux_stats"].items()})
