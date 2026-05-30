#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional


def knapsack_channel_selection(
    signal_yields: np.ndarray,
    background_yields: np.ndarray,
    luminosities: np.ndarray,
    max_lumi: float
) -> Tuple[float, np.ndarray]:
    n = signal_yields.size
    if n == 0:
        return 0.0, np.array([], dtype=bool)


    significances = np.zeros(n)
    for i in range(n):
        b = max(background_yields[i], 1.0)
        s = max(signal_yields[i], 0.0)
        significances[i] = s / np.sqrt(b)


    weights = np.maximum(np.round(luminosities).astype(int), 1)
    capacity = max(int(np.round(max_lumi)), 1)


    dp = np.zeros(capacity + 1)
    choice = np.full((n, capacity + 1), -1, dtype=int)

    for i in range(n):
        w = weights[i]
        v = significances[i]
        for j in range(capacity, w - 1, -1):
            if dp[j - w] + v > dp[j]:
                dp[j] = dp[j - w] + v
                choice[i, j] = j - w


    selected = np.zeros(n, dtype=bool)
    j = capacity
    for i in range(n - 1, -1, -1):
        if choice[i, j] >= 0:
            selected[i] = True
            j = choice[i, j]

    total_sig = dp[capacity]
    return total_sig, selected


def smolyak_parameter_scan(
    mass_range: Tuple[float, float],
    coupling_range: Tuple[float, float],
    width_range: Tuple[float, float],
    max_level: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    try:
        from interpolation_utils import order_from_level_135, cc_compute_points
    except ImportError:
        from .interpolation_utils import order_from_level_135, cc_compute_points

    levels = list(range(max_level + 1))
    all_mass = []
    all_coupling = []
    all_width = []

    for lm in levels:
        n_m = order_from_level_135(lm)
        pts_m = cc_compute_points(n_m)
        pts_m = 0.5 * ((1.0 - pts_m) * mass_range[0] + (1.0 + pts_m) * mass_range[1])

        for lc in levels:
            n_c = order_from_level_135(lc)
            pts_c = cc_compute_points(n_c)
            pts_c = 0.5 * ((1.0 - pts_c) * coupling_range[0] + (1.0 + pts_c) * coupling_range[1])

            for lw in levels:

                if lm + lc + lw > max_level + 2:
                    continue

                n_w = order_from_level_135(lw)
                pts_w = cc_compute_points(n_w)
                pts_w = 0.5 * ((1.0 - pts_w) * width_range[0] + (1.0 + pts_w) * width_range[1])

                for m in pts_m:
                    for c in pts_c:
                        for w in pts_w:
                            all_mass.append(m)
                            all_coupling.append(c)
                            all_width.append(w)

    return np.array(all_mass), np.array(all_coupling), np.array(all_width)


def expected_signal_yield(
    cross_section: float,
    luminosity: float,
    efficiency: float = 0.5,
    branching_ratio: float = 0.1
) -> float:
    if cross_section < 0.0 or luminosity < 0.0:
        return 0.0

    lumi_pb = luminosity * 1000.0
    eff = np.clip(efficiency, 0.0, 1.0)
    br = np.clip(branching_ratio, 0.0, 1.0)
    return cross_section * lumi_pb * eff * br


def exclusion_contour_2d(
    mass_grid: np.ndarray,
    coupling_grid: np.ndarray,
    significance_grid: np.ndarray,
    cl_threshold: float = 1.96
) -> List[Tuple[float, float]]:
    contour_points = []
    nm = len(mass_grid)
    nc = len(coupling_grid)

    for i in range(nm - 1):
        for j in range(nc - 1):

            vals = [
                significance_grid[i, j],
                significance_grid[i + 1, j],
                significance_grid[i, j + 1],
                significance_grid[i + 1, j + 1]
            ]
            above = [v >= cl_threshold for v in vals]
            n_above = sum(above)

            if n_above > 0 and n_above < 4:

                m_mid = (mass_grid[i] + mass_grid[i + 1]) / 2.0
                c_mid = (coupling_grid[j] + coupling_grid[j + 1]) / 2.0
                contour_points.append((m_mid, c_mid))

    return contour_points


def discovery_potential(
    signal_cross_sections: np.ndarray,
    background_cross_sections: np.ndarray,
    luminosities: np.ndarray,
    systematic_errors: np.ndarray
) -> np.ndarray:
    s = np.maximum(signal_cross_sections * luminosities * 1000.0, 0.0)
    b = np.maximum(background_cross_sections * luminosities * 1000.0, 1.0)
    epsilon = np.clip(systematic_errors, 0.0, 1.0)


    ratio = s / b
    ratio = np.clip(ratio, 1e-10, 1e6)

    z = np.sqrt(2.0 * ((s + b) * np.log(1.0 + ratio) - s))


    sigma_b = epsilon * b
    denom = np.sqrt(b + sigma_b ** 2)
    z_approx = s / denom


    z = np.minimum(z, z_approx)

    return z
