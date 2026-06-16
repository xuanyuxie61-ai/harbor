"""
denoise_filter.py — 3×3 median filter for stabilising real-space fields
========================================================================

High-order finite-difference operators applied to noisy charge densities
or potentials can produce Gibbs-like oscillations near the defect core. We
use a 3×3 *median filter* as a non-linear regulariser: for each grid point
the value is replaced by the median of its 9-point neighbourhood (itself
plus the 8 nearest neighbours with periodic wrapping).

This is a direct Python port of
    `576_image_denoise/image_denoise_gray_3x3.m`
by John Burkardt. We adapt it to the scientific context: instead of image
pixels we filter a 2-D scalar field (charge density, potential, or wave
function) on a periodic real-space grid. The filter:

  * preserves sharp features (unlike a Gaussian blur),
  * removes salt-and-pepper noise caused by numerical outliers,
  * commutes with the normalisation constraint on the charge density
    (∫ n d²r = N_e is approximately preserved).

We also provide a *selective* version that only filters points where the
local Laplacian exceeds a threshold (i.e. only near the defect core).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------------
# (1) Full-grid 3x3 median filter (port of image_denoise_gray_3x3.m)
# -------------------------------------------------------------------------
def median_filter_3x3(field: np.ndarray) -> np.ndarray:
    """Apply a 3x3 median filter with periodic boundary conditions.

    Port of `image_denoise_gray_3x3.m`. We use `np.roll` in place of
    Burkardt's `circshift`: the two are identical for 2-D arrays.

    Parameters
    ----------
    field : (Ny, Nx) ndarray — the noisy scalar field.

    Returns
    -------
    filtered : (Ny, Nx) ndarray — the filtered field.
    """
    m, n = field.shape
    # collect the 9 neighbourhood layers (Burkardt's gray3d)
    layers = np.empty((9, m, n), dtype=field.dtype)
    shifts = [(-1, -1), (-1, 0), (-1, +1),
              (0, -1), (0, 0), (0, +1),
              (+1, -1), (+1, 0), (+1, +1)]
    for k, (sy, sx) in enumerate(shifts):
        layers[k] = np.roll(np.roll(field, -sy, axis=0), -sx, axis=1)
    # vectorised median along axis 0
    return np.median(layers, axis=0)


# -------------------------------------------------------------------------
# (2) Selective filter: only apply where |∇²field| > threshold
# -------------------------------------------------------------------------
def selective_median_filter(field: np.ndarray, h: float,
                            threshold: float, n_iter: int = 1
                            ) -> np.ndarray:
    """Apply the median filter selectively near sharp features.

    A grid point (i, j) is flagged if
        |∇² field(i, j)| > threshold
    where ∇² is the standard 2nd-order FD Laplacian. Only flagged points
    are replaced by the median of their neighbourhood; unflagged points
    are left unchanged. Iterated `n_iter` times.
    """
    out = field.copy()
    for _ in range(n_iter):
        lap = (np.roll(out, -1, axis=0) + np.roll(out, +1, axis=0)
               + np.roll(out, -1, axis=1) + np.roll(out, +1, axis=1)
               - 4.0 * out) / (h * h)
        flagged = np.abs(lap) > threshold
        if not flagged.any():
            break
        med = median_filter_3x3(out)
        out = np.where(flagged, med, out)
    return out


# -------------------------------------------------------------------------
# (3) Charge-conserving variant: renormalise after filtering
# -------------------------------------------------------------------------
def charge_conserving_median(field: np.ndarray, h: float,
                             target_charge: float) -> np.ndarray:
    """Apply median filter and rescale so that ∫ n d²r = target_charge."""
    filt = median_filter_3x3(field)
    filt = np.maximum(filt, 0.0)     # density must be non-negative
    current = float(np.sum(filt)) * h * h
    if current < 1e-30:
        return field
    return filt * (target_charge / current)


# -------------------------------------------------------------------------
# (4) Quality metrics (to assess the effect of denoising)
# -------------------------------------------------------------------------
def field_roughness(field: np.ndarray, h: float) -> float:
    """Compute the roughness  R = ∫ |∇ field|² d²r as a quality metric."""
    gx = (np.roll(field, -1, axis=1) - np.roll(field, +1, axis=1)) / (2 * h)
    gy = (np.roll(field, -1, axis=0) - np.roll(field, +1, axis=0)) / (2 * h)
    return float(np.sum(gx ** 2 + gy ** 2)) * h * h


def denoising_report(field: np.ndarray, h: float,
                     target_charge: float) -> Tuple[float, float, float]:
    """Return (roughness_before, roughness_after, charge_error)."""
    r_before = field_roughness(field, h)
    filt = charge_conserving_median(field, h, target_charge)
    r_after = field_roughness(filt, h)
    q_err = abs(float(np.sum(filt)) * h * h - target_charge)
    return r_before, r_after, q_err
