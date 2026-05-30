#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List, Optional






def barycentric_interpolate_triangle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    v1: float,
    v2: float,
    v3: float,
    p: np.ndarray
) -> float:

    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    if abs(denom) < 1e-14:
        return -1.0

    lam1 = ((p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])) / denom
    lam2 = ((p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])) / denom
    lam3 = 1.0 - lam1 - lam2


    if lam1 < -1e-8 or lam2 < -1e-8 or lam3 < -1e-8:
        return -1.0


    lam1 = max(0.0, min(1.0, lam1))
    lam2 = max(0.0, min(1.0, lam2))
    lam3 = max(0.0, min(1.0, lam3))
    total = lam1 + lam2 + lam3
    if total > 1e-15:
        lam1 /= total
        lam2 /= total
        lam3 /= total

    return lam1 * v1 + lam2 * v2 + lam3 * v3


def pwl_interp_2d_scattered(
    data_points: np.ndarray,
    data_values: np.ndarray,
    query_points: np.ndarray
) -> np.ndarray:
    n = data_points.shape[0]
    m = query_points.shape[0]
    result = np.zeros(m)

    for i in range(m):
        q = query_points[i]


        dists = np.linalg.norm(data_points - q, axis=1)
        idx = np.argsort(dists)[:3]

        if dists[idx[0]] < 1e-12:

            result[i] = data_values[idx[0]]
            continue

        p1, p2, p3 = data_points[idx[0]], data_points[idx[1]], data_points[idx[2]]
        v1, v2, v3 = data_values[idx[0]], data_values[idx[1]], data_values[idx[2]]

        val = barycentric_interpolate_triangle(p1, p2, p3, v1, v2, v3, q)
        if val < 0:

            w = 1.0 / (dists[idx] + 1e-10)
            val = np.sum(w * data_values[idx]) / np.sum(w)

        result[i] = val

    return result






def cc_compute_points(n: int) -> np.ndarray:
    if n < 1:
        return np.array([0.0])
    if n == 1:
        return np.array([0.0])
    j = np.arange(n)
    return np.cos(np.pi * j / (n - 1))


def order_from_level_135(level: int) -> int:
    if level <= 0:
        return 1
    elif level == 1:
        return 2
    else:
        return 2 ** (level - 1) + 1


def smolyak_coefficients(m: int, level_vec: np.ndarray) -> np.ndarray:
    l1_norm = np.sum(level_vec)
    k = l1_norm - m
    if k < 0:
        return np.array([0.0])


    from math import comb
    if k > m - 1:
        return np.array([0.0])

    coeff = ((-1) ** k) * comb(m - 1, k)
    return np.array([float(coeff)])


def lagrange_basis_1d(n: int, x_nodes: np.ndarray, x_query: float) -> np.ndarray:
    basis = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-14:
                    basis[j] = 0.0
                else:
                    basis[j] *= (x_query - x_nodes[k]) / denom
    return basis


def sparse_interp_nd_value(
    m: int,
    ind: np.ndarray,
    a_bounds: np.ndarray,
    b_bounds: np.ndarray,
    nd: int,
    zd: np.ndarray,
    xi: np.ndarray
) -> float:

    cc_nodes = []
    for i in range(m):
        n_1d = order_from_level_135(ind[i])
        x_1d = cc_compute_points(n_1d)

        x_1d = 0.5 * ((1.0 - x_1d) * a_bounds[i] + (1.0 + x_1d) * b_bounds[i])
        cc_nodes.append(x_1d)


    weights_per_dim = []
    for i in range(m):
        basis = lagrange_basis_1d(cc_nodes[i].size, cc_nodes[i], xi[i])
        weights_per_dim.append(basis)




    w = np.ones(nd)
    idx = 0


    strides = [1] * m
    for i in range(m - 2, -1, -1):
        strides[i] = strides[i + 1] * cc_nodes[i + 1].size

    result = 0.0
    for flat_idx in range(nd):
        w = 1.0
        temp_idx = flat_idx
        for i in range(m):
            node_idx = temp_idx // strides[i] if i < m - 1 else temp_idx
            temp_idx = temp_idx % strides[i] if i < m - 1 else 0
            if node_idx < len(weights_per_dim[i]):
                w *= weights_per_dim[i][node_idx]
            else:
                w = 0.0
                break
        if flat_idx < len(zd):
            result += w * zd[flat_idx]

    return result


def bsm_cross_section_interp_2d(
    mass_grid: np.ndarray,
    coupling_grid: np.ndarray,
    cross_section_table: np.ndarray,
    query_mass: float,
    query_coupling: float
) -> float:

    m_min, m_max = mass_grid[0], mass_grid[-1]
    c_min, c_max = coupling_grid[0], coupling_grid[-1]

    if query_mass < m_min or query_mass > m_max or query_coupling < c_min or query_coupling > c_max:

        query_mass = np.clip(query_mass, m_min, m_max)
        query_coupling = np.clip(query_coupling, c_min, c_max)


    i = np.searchsorted(mass_grid, query_mass, side='right') - 1
    j = np.searchsorted(coupling_grid, query_coupling, side='right') - 1

    i = max(0, min(i, len(mass_grid) - 2))
    j = max(0, min(j, len(coupling_grid) - 2))

    m1, m2 = mass_grid[i], mass_grid[i + 1]
    c1, c2 = coupling_grid[j], coupling_grid[j + 1]

    v11 = cross_section_table[i, j]
    v12 = cross_section_table[i, j + 1]
    v21 = cross_section_table[i + 1, j]
    v22 = cross_section_table[i + 1, j + 1]


    dm = m2 - m1
    dc = c2 - c1
    if abs(dm) < 1e-14 or abs(dc) < 1e-14:
        return v11

    t = (query_mass - m1) / dm
    s = (query_coupling - c1) / dc

    return (1.0 - t) * (1.0 - s) * v11 + (1.0 - t) * s * v12 \
         + t * (1.0 - s) * v21 + t * s * v22
