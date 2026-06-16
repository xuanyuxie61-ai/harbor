"""
latin_sampler.py - Latin Hypercube Sampling for parameter sensitivity.

Adapted from 649_latin_center (centered Latin hypercube) and 1358_trinity
(tile/grid decomposition).  Generates quasi-random parameter samples for
r-process sensitivity studies.

For a d-dimensional unit cube, divide each dimension into n strata and
place one sample per stratum row/column.  The centered variant places
samples at (i + 0.5) / n within each stratum.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

def latin_center(n: int, d: int, seed: int = 1) -> np.ndarray:
    """
    Generate n samples in d dimensions using centered Latin hypercube.
    Returns array of shape (n, d) with values in [0, 1).
    Adapted from 649_latin_center.m.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            samples[i, j] = (perm[i] + 0.5) / n
    return samples


def latin_hypercube_scale(samples: np.ndarray,
                          lower: List[float],
                          upper: List[float]) -> np.ndarray:
    """Scale [0,1] samples to [lower, upper] for each dimension."""
    scaled = samples.copy()
    for j in range(samples.shape[1]):
        scaled[:, j] = lower[j] + samples[:, j] * (upper[j] - lower[j])
    return scaled


def trinity_tile_id(word_index: int, words_per_tile: int) -> int:
    """Return tile ID for a given word index.  Adapted from 1358_trinity."""
    return word_index // words_per_tile


def trinity_grid(nuclides: List[Tuple[int, int]],
                 grid_A: int, grid_Z: int) -> np.ndarray:
    """
    Map nuclides (A, Z) to a 2D grid (A_grid, Z_grid).
    Returns grid of shape (grid_A, grid_Z) with 1 where nuclide present.
    """
    grid = np.zeros((grid_A, grid_Z), dtype=np.int32)
    for A, Z in nuclides:
        if 0 <= A < grid_A and 0 <= Z < grid_Z:
            grid[A, Z] = 1
    return grid


def sensitivity_samples(param_names: List[str],
                        param_ranges: List[Tuple[float, float]],
                        n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin hypercube samples for sensitivity analysis.
    param_ranges: list of (lower, upper) for each parameter.
    Returns array of shape (n_samples, n_params).
    """
    d = len(param_names)
    lower = [r[0] for r in param_ranges]
    upper = [r[1] for r in param_ranges]
    samples_unit = latin_center(n_samples, d, seed)
    return latin_hypercube_scale(samples_unit, lower, upper)


__all__ = [
    "latin_center", "latin_hypercube_scale",
    "trinity_tile_id", "trinity_grid",
    "sensitivity_samples",
]
