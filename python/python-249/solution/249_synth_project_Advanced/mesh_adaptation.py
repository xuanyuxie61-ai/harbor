# -*- coding: utf-8 -*-
"""
mesh_adaptation.py
==================
Adaptive mass-grid generation for the Lagrangian stellar structure
equations via a 1D Centroidal Voronoi Tessellation (CVT).

Background
----------
A Lagrangian stellar model places grid points at fixed enclosed-mass
coordinates m_i.  To resolve the sharp gradients near the centre and
at burning shells, the mass grid is adapted iteratively using an
equidistribution principle:

    rho(m) * h(m) = const

where rho(m) is a monitor function (typically the absolute value of a
second derivative) and h(m) = m_{i+1} - m_i the local spacing.

This is exactly a 1D CVT with non-uniform density, generalising the
2D algorithm in seed 253_cvt_circle_nonuniform to a bounded 1D
interval [0, M_total] with a user-specified density function.

Algorithm
---------
Given a density function w(m) > 0 on [0, M] and N generators
0 = g_0 < g_1 < ... < g_{N-1} < g_N = M, iterate:
  1. For a dense Monte Carlo sample {s_j} drawn from w, assign each
     sample to the nearest generator.
  2. Replace each generator by the weighted centroid of its Voronoi
     cell (the mean of its assigned samples).
  3. Enforce g_0 = 0, g_N = M.
  4. Repeat until the maximum generator displacement is < tol.

The resulting grid concentrates points where w(m) is large, exactly
like the non-uniform CVT on a circle concentrates Voronoi cells where
the sampling density is high.
"""

from __future__ import annotations
from typing import List, Callable, Tuple
import math
import random


# =====================================================================
# Monitor functions
# =====================================================================

def monitor_equidistribute(N: int) -> Callable[[float], float]:
    """Uniform monitor w(m) = 1: recovers a uniform grid."""
    def w(m: float) -> float:
        return 1.0
    return w


def monitor_derivative_second(d2f: Callable[[float], float],
                              floor: float = 1.0e-6) -> Callable[[float], float]:
    """Monitor w(m) = floor + |d2f/dm2(m)|  (arc-length equidistribution)."""
    def w(m: float) -> float:
        return floor + abs(d2f(m))
    return w


def monitor_shell_shells(shells: List[Tuple[float, float, float]]
                         ) -> Callable[[float], float]:
    """Monitor that concentrates points near shell-burning locations.

    Parameters
    ----------
    shells : list of (centre, width, amplitude)
        Each shell contributes  amplitude * exp(-(m - centre)^2 / (2 width^2)).
    """
    def w(m: float) -> float:
        s = 1.0
        for centre, width, amplitude in shells:
            s += amplitude * math.exp(-((m - centre) ** 2)
                                      / (2.0 * max(width, 1.0e-30)**2))
        return s
    return w


# =====================================================================
# 1D CVT iteration
# =====================================================================

def cvt_1d(M_total: float, N: int,
           density: Callable[[float], float],
           n_samples: int = 50000,
           n_iter: int = 60,
           tol: float = 1.0e-8,
           seed: int = 12345) -> List[float]:
    """Compute a 1D Centroidal Voronoi Tessellation of [0, M_total]
    with non-uniform density `density`.

    Parameters
    ----------
    M_total : total mass.
    N : number of interior generators (total grid has N+1 points
        including 0 and M_total).
    density : positive density function on [0, M_total].
    n_samples : number of Monte Carlo samples per iteration.
    n_iter : maximum number of CVT iterations.
    tol : convergence tolerance on max generator displacement.
    seed : random seed for reproducible sampling.

    Returns
    -------
    generators : list of N+2 floats
        The sorted grid [0, g_1, g_2, ..., g_N, M_total].
    """
    if N < 1:
        raise ValueError("cvt_1d: N must be >= 1")
    rng = random.Random(seed)

    # Initial generators: uniform spacing
    g = [M_total * (i + 1) / (N + 1) for i in range(N)]

    for it in range(n_iter):
        # Sample from density via rejection sampling on [0, M_total]
        # Determine the maximum of density for rejection bound
        n_test = 200
        dmax = 1.0
        for k in range(n_test):
            mt = M_total * k / (n_test - 1)
            dmax = max(dmax, density(mt))
        samples = []
        while len(samples) < n_samples:
            s = rng.random() * M_total
            u = rng.random() * dmax
            if u <= density(s):
                samples.append(s)
        # Assign each sample to the nearest generator
        counts = [0] * N
        sums  = [0.0] * N
        for s in samples:
            # Find nearest generator by linear scan (fast enough for N~O(100))
            best = 0
            dbest = abs(s - g[0])
            for k in range(1, N):
                d = abs(s - g[k])
                if d < dbest:
                    dbest = d
                    best = k
            counts[best] += 1
            sums[best] += s
        # Update generators
        max_disp = 0.0
        g_new = g[:]
        for k in range(N):
            if counts[k] > 0:
                g_new[k] = sums[k] / counts[k]
            max_disp = max(max_disp, abs(g_new[k] - g[k]))
        g = g_new
        if max_disp < tol * M_total:
            break

    # Build final grid with boundary points
    grid = [0.0] + sorted(g) + [M_total]
    return grid


# =====================================================================
# Grid-quality diagnostics
# =====================================================================

def grid_ratio(grid: List[float]) -> float:
    """Return the ratio max(h) / min(h) of the grid spacings."""
    if len(grid) < 2:
        return 1.0
    hs = [grid[i+1] - grid[i] for i in range(len(grid)-1)]
    hs_pos = [h for h in hs if h > 0]
    if not hs_pos:
        return float("inf")
    return max(hs_pos) / min(hs_pos)


def grid_smoothness(grid: List[float]) -> float:
    """Return the smoothness measure  max |h_{i+1} - h_i| / h_avg."""
    if len(grid) < 3:
        return 0.0
    hs = [grid[i+1] - grid[i] for i in range(len(grid)-1)]
    havg = sum(hs) / len(hs)
    if havg <= 0.0:
        return float("inf")
    return max(abs(hs[i+1] - hs[i]) for i in range(len(hs)-1)) / havg


def interpolate_onto_grid(f_values: List[float], old_grid: List[float],
                           new_grid: List[float]) -> List[float]:
    """Linearly interpolate f from old_grid to new_grid."""
    N_old = len(old_grid)
    new_f = [0.0] * len(new_grid)
    j = 0
    for i, mn in enumerate(new_grid):
        while j < N_old - 2 and old_grid[j+1] < mn:
            j += 1
        if j >= N_old - 1:
            j = N_old - 2
        h = old_grid[j+1] - old_grid[j]
        if h <= 0.0:
            new_f[i] = f_values[j]
            continue
        t = (mn - old_grid[j]) / h
        t = max(0.0, min(1.0, t))
        new_f[i] = (1.0 - t) * f_values[j] + t * f_values[j+1]
    return new_f


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("mesh_adaptation self-test:")
    # Uniform density -> near-uniform grid
    N = 16
    M = 1.0
    g_uni = cvt_1d(M, N, monitor_equidistribute(N), n_iter=30, seed=1)
    print(f"  uniform CVT: ratio={grid_ratio(g_uni):.3f}  "
          f"smooth={grid_smoothness(g_uni):.3f}")
    # Shell-peaked density
    def w_shell(m):
        return 1.0 + 50.0 * math.exp(-((m - 0.3)**2) / 0.005)
    g_shell = cvt_1d(M, N, w_shell, n_iter=60, seed=1)
    print(f"  shell CVT:   ratio={grid_ratio(g_shell):.3f}  "
          f"smooth={grid_smoothness(g_shell):.3f}")
    print(f"  sample of shell grid (first 6): {[round(x,4) for x in g_shell[:6]]}")
    print("mesh_adaptation self-test OK")


if __name__ == "__main__":
    _self_test()
