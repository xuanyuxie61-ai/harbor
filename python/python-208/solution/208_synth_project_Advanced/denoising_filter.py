"""
denoising_filter.py
===================

Spatial and temporal denoising kernels for the multi-fidelity UQ
framework.  Adapted from the 3x3 and Newsam image-denoising filters
of project 576.

In multi-fidelity UQ, the "image" being denoised is the d-dimensional
parameter-space surface F(xi) of simulator outputs.  Observational
noise (from measurement, from stochastic simulation estimators, or from
deliberate coarse-grid bias) corrupts the training labels.  We apply
two denoising strategies:

1. Local 3x3 median filter (from `image_denoise_gray_3x3`):
   - Each training label is replaced by the median of its 3^d neighbors
     in parameter space.  This suppresses outlier labels without blurring
     sharp features (unlike Gaussian smoothing).

2. Non-local means filter (from `image_denoise_gray_news`):
   - A patch-based method: for each training point xi, we average the
     labels of all other training points, weighted by the similarity
     of their local neighborhoods.

Both are adapted from 2-D image processing to d-dimensional parameter
spaces; the circular shift used in the original image-denoise code
becomes a periodic wrap-around in the parameter-space grid.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# d-dimensional neighborhood enumeration.
# ----------------------------------------------------------------------
def neighborhood_offsets(d: int, radius: int = 1) -> List[List[int]]:
    """Enumerate all offsets in [-radius, radius]^d (including [0,...,0])."""
    grid = [[]]
    for _ in range(d):
        new_grid = []
        for g in grid:
            for k in range(-radius, radius + 1):
                new_grid.append(g + [k])
        grid = new_grid
    return grid


# ----------------------------------------------------------------------
# 3x3 median filter adapted to d dimensions.
# ----------------------------------------------------------------------
def median_filter_1d(values: List[float], kernel: int = 3) -> List[float]:
    """1-D median filter with periodic boundary conditions."""
    n = len(values)
    if n == 0:
        return []
    half = kernel // 2
    out = [0.0] * n
    for i in range(n):
        window: List[float] = []
        for k in range(-half, half + 1):
            idx = (i + k) % n
            window.append(values[idx])
        window.sort()
        out[i] = window[len(window) // 2]
    return out


def median_filter_d(
    grid_values: List[float],
    shape: List[int],
    kernel: int = 3,
) -> List[float]:
    """d-dimensional median filter on a flattened tensor.

    Parameters
    ----------
    grid_values : list of n = prod(shape) values.
    shape       : list of d side-lengths.
    kernel      : side length of the cubic kernel (odd).

    Returns
    -------
    flattened tensor of denoised values.
    """
    d = len(shape)
    n = len(grid_values)
    if n != _prod(shape):
        raise ValueError("median_filter_d: shape/value length mismatch.")
    half = kernel // 2
    offsets = neighborhood_offsets(d, half)
    # Precompute strides.
    strides = [1] * d
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    def unravel(idx: int) -> List[int]:
        coords = [0] * d
        rem = idx
        for k in range(d):
            coords[k] = rem // strides[k]
            rem %= strides[k]
        return coords

    def ravel(coords: List[int]) -> int:
        return sum(coords[k] * strides[k] for k in range(d))

    out = [0.0] * n
    for i in range(n):
        coords = unravel(i)
        window: List[float] = []
        for off in offsets:
            nbr = [(coords[k] + off[k]) % shape[k] for k in range(d)]
            j = ravel(nbr)
            window.append(grid_values[j])
        window.sort()
        out[i] = window[len(window) // 2]
    return out


def _prod(x: List[int]) -> int:
    out = 1
    for v in x:
        out *= v
    return out


# ----------------------------------------------------------------------
# Non-local means filter (adapted from image_denoise_gray_news).
# ----------------------------------------------------------------------
def nonlocal_means_1d(
    values: List[float],
    h: float = 0.1,
    patch_radius: int = 2,
    search_radius: int = 5,
) -> List[float]:
    """Non-local means filter in 1-D.

    For each index i, the denoised value is:
        y_i = sum_j w_{ij} v_j  /  sum_j w_{ij}
    where
        w_{ij} = exp( - || P_i - P_j ||^2 / h^2 )
    and P_i is the patch of values centered at i with radius patch_radius.
    """
    n = len(values)
    if n == 0:
        return []
    out = [0.0] * n
    for i in range(n):
        # Patch around i.
        patch_i = [values[(i + k) % n] for k in range(-patch_radius, patch_radius + 1)]
        wsum = 0.0
        vsum = 0.0
        for j in range(max(0, i - search_radius),
                       min(n, i + search_radius + 1)):
            patch_j = [values[(j + k) % n] for k in range(-patch_radius, patch_radius + 1)]
            d2 = sum((patch_i[m] - patch_j[m]) ** 2 for m in range(len(patch_i)))
            w = math.exp(-d2 / max(h * h, 1.0e-30))
            wsum += w
            vsum += w * values[j]
        out[i] = vsum / max(wsum, 1.0e-30)
    return out


def nonlocal_means_d(
    grid_values: List[float],
    shape: List[int],
    h: float = 0.1,
    patch_radius: int = 1,
    search_radius: int = 2,
) -> List[float]:
    """Non-local means on a d-dimensional grid (flattened).

    For tractability this uses a restricted neighborhood search.
    """
    d = len(shape)
    n = len(grid_values)
    if n != _prod(shape):
        raise ValueError("nonlocal_means_d: shape/value length mismatch.")
    strides = [1] * d
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    def unravel(idx: int) -> List[int]:
        coords = [0] * d
        rem = idx
        for k in range(d):
            coords[k] = rem // strides[k]
            rem %= strides[k]
        return coords

    def ravel(coords: List[int]) -> int:
        return sum(coords[k] * strides[k] for k in range(d))

    patch_offsets = neighborhood_offsets(d, patch_radius)
    search_offsets = neighborhood_offsets(d, search_radius)

    out = [0.0] * n
    for i in range(n):
        ci = unravel(i)
        patch_i = [
            grid_values[ravel([(ci[k] + off[k]) % shape[k] for k in range(d)])]
            for off in patch_offsets
        ]
        wsum = 0.0
        vsum = 0.0
        for off_s in search_offsets:
            j_coords = [(ci[k] + off_s[k]) % shape[k] for k in range(d)]
            j = ravel(j_coords)
            patch_j = [
                grid_values[ravel([(j_coords[k] + off[k]) % shape[k] for k in range(d)])]
                for off in patch_offsets
            ]
            d2 = sum((patch_i[m] - patch_j[m]) ** 2 for m in range(len(patch_i)))
            w = math.exp(-d2 / max(h * h, 1.0e-30))
            wsum += w
            vsum += w * grid_values[j]
        out[i] = vsum / max(wsum, 1.0e-30)
    return out


# ----------------------------------------------------------------------
# Denoising a list of (xi, y) training samples.
# ----------------------------------------------------------------------
def denoise_training_labels(
    xi_samples: List[List[float]],
    y_values: List[float],
    method: str = "median",
    h: float = 0.1,
) -> List[float]:
    """Apply denoising to the training labels y_values.

    The samples are sorted lexicographically by xi, treated as a 1-D
    sequence, and denoised by the chosen method.  This is a practical
    compromise that works well when the xi samples roughly lie on a
    low-dimensional manifold.
    """
    if len(xi_samples) != len(y_values):
        raise ValueError("denoise_training_labels: length mismatch.")
    # Sort lexicographically.
    paired = sorted(zip(xi_samples, y_values), key=lambda p: p[0])
    y_sorted = [p[1] for p in paired]
    if method == "median":
        y_den = median_filter_1d(y_sorted, kernel=3)
    elif method == "nlm":
        y_den = nonlocal_means_1d(y_sorted, h=h)
    elif method == "none":
        y_den = list(y_sorted)
    else:
        raise ValueError(f"denoise_training_labels: unknown method '{method}'.")
    # Restore original order.
    index_map = sorted(range(len(paired)), key=lambda i: paired[i][0])
    y_out = [0.0] * len(y_sorted)
    for out_idx, orig_idx in enumerate(index_map):
        y_out[orig_idx] = y_den[out_idx]
    return y_out
