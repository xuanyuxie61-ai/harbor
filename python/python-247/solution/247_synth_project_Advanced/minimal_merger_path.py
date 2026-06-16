"""
minimal_merger_path.py
======================
Optimal ordering of halo merger events as a Travelling Salesperson
Problem (TSP) in mass-redshift space.

Physical motivation
-------------------
When reconstructing a merger tree from a discrete set of progenitor
haloes detected at different redshifts, one wishes to find the
chronologically and physically most plausible chain of mergers.  We
pose this as a TSP: each progenitor is a "city" with coordinates

    x_i = ( log10(M_i / M_sun), z_i ),

and the distance is a physically-motivated metric

    d(i, j) = sqrt( (Delta log M / sigma_M)^2 + (Delta z / sigma_z)^2 )

where sigma_M, sigma_z are normalisation scales inspired by the
scatter of the halo mass function and the redshift spacing of the
snapshots.  A random-sampling heuristic (Burkardt's tsp_random) is
used to find a good tour for the small trees we consider (N <= 12).
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple

import numpy as np


# ---------- Distance matrix ---------------------------------------------------

SIGMA_M = 0.6     # dex  - typical log-mass scatter
SIGMA_Z = 0.3     #      - typical redshift spacing


def merger_distance(logM_i: float, z_i: float,
                    logM_j: float, z_j: float) -> float:
    """Physically normalised distance between two merger events."""
    dm = (logM_i - logM_j) / SIGMA_M
    dz = (z_i - z_j) / SIGMA_Z
    return math.sqrt(dm * dm + dz * dz)


def build_distance_matrix(progenitors: List[Tuple[float, float]]
                          ) -> np.ndarray:
    """Return the symmetric distance matrix for a list of
    (log10 M/M_sun, z) tuples."""
    n = len(progenitors)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = merger_distance(progenitors[i][0], progenitors[i][1],
                                progenitors[j][0], progenitors[j][1])
            D[i, j] = D[j, i] = d
    return D


# ---------- Tour length -------------------------------------------------------

def tour_length(tour: List[int], D: np.ndarray) -> float:
    """Total length of a closed tour."""
    L = 0.0
    n = len(tour)
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        L += D[i, j]
    return L


# ---------- Random-sampling TSP heuristic ------------------------------------

def tsp_random_sample(D: np.ndarray, n_samples: int = 2000,
                      rng: random.Random | None = None
                      ) -> Tuple[List[int], float]:
    """Sample n_samples random permutations and return the shortest
    closed tour found (Burkardt's tsp_random strategy)."""
    rng = rng or random.Random()
    n = D.shape[0]
    best_tour = list(range(n))
    best_len = tour_length(best_tour, D)
    for _ in range(n_samples):
        cand = list(range(n))
        rng.shuffle(cand)
        L = tour_length(cand, D)
        if L < best_len:
            best_tour, best_len = cand, L
    return best_tour, best_len


# ---------- 2-opt local improvement ------------------------------------------

def two_opt_improve(tour: List[int], D: np.ndarray,
                    max_iter: int = 200) -> Tuple[List[int], float]:
    """Classical 2-opt local search for TSP."""
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                before = D[a, b] + D[c, d]
                after = D[a, c] + D[b, d]
                if after + 1e-12 < before:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour, tour_length(tour, D)


# ---------- Public interface --------------------------------------------------

def optimal_merger_path(progenitors: List[Tuple[float, float]],
                        n_samples: int = 2000,
                        seed: int = 12345
                        ) -> dict:
    """Find a good ordering of merger events."""
    rng = random.Random(seed)
    D = build_distance_matrix(progenitors)
    tour_rand, L_rand = tsp_random_sample(D, n_samples=n_samples, rng=rng)
    tour_opt, L_opt = two_opt_improve(list(tour_rand), D)
    return dict(progenitors=progenitors,
                tour_random=tour_rand, length_random=L_rand,
                tour_opt=tour_opt, length_opt=L_opt,
                distance_matrix=D)


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    rng = random.Random(7)
    prog = [(10.0 + 0.3 * rng.gauss(0, 1), 0.2 * k + rng.gauss(0, 0.05))
            for k in range(6)]
    res = optimal_merger_path(prog, n_samples=500, seed=7)
    return dict(n=len(prog),
                L_rand=res["length_random"],
                L_opt=res["length_opt"])
