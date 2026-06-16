"""
Gauge fixing via codeword overlap reduction.

Adapted from:
  - 1081_FranciscoHS_toy-model-cis-code (binary codeword construction,
    overlap reduction via edge swaps)
"""

import numpy as np
from constants import LatticeParams
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField


def link_to_codeword(U, n_bits=4):
    if n_bits != 4:
        raise ValueError(f"Only n_bits=4 supported, got {n_bits}")
    a0 = 0.5 * (U[0, 0] + U[1, 1]).real
    a1 = 0.5 * (U[0, 1] + U[1, 0]).imag
    a2 = 0.5 * (-U[0, 1] + U[1, 0]).real
    a3 = 0.5 * (U[0, 0] - U[1, 1]).imag
    coeffs = np.array([a0, a1, a2, a3])
    return (coeffs >= 0).astype(int)


def codeword_overlap(c1, c2):
    return int(np.dot(c1, c2))


def build_codeword_matrix(gf):
    geo = gf.geo
    n_links = gf.p.V * 4
    M = np.zeros((n_links, 4), dtype=int)
    row = 0
    for s in geo.all_sites():
        for mu in range(4):
            U = gf.link(s, mu)
            M[row] = link_to_codeword(U)
            row += 1
    return M


def overlap_objective(M):
    n = M.shape[0]
    if n < 2:
        return 0
    Ov = M.astype(int) @ M.astype(int).T
    np.fill_diagonal(Ov, 0)
    return int((Ov ** 2).sum() // 2)


def reduce_overlap_gauge(M, n_iters=500, seed=42):
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    rng = np.random.default_rng(seed)
    M = M.copy()
    n_links, n_bits = M.shape
    edges = list(zip(*np.where(M)))
    if len(edges) < 2:
        return {
            'M_optimized': M, 'objective_history': [overlap_objective(M)],
            'initial_objective': overlap_objective(M), 'final_objective': overlap_objective(M),
        }
    obj = overlap_objective(M)
    history = [obj]
    for _ in range(n_iters):
        (i, a), (j, b) = (
            edges[rng.integers(len(edges))],
            edges[rng.integers(len(edges))],
        )
        if i == j or a == b or M[i, b] or M[j, a]:
            continue
        M[i, a] = M[j, b] = 0
        M[i, b] = M[j, a] = 1
        new_obj = overlap_objective(M)
        if new_obj <= obj:
            obj = new_obj
        else:
            M[i, a] = M[j, b] = 1
            M[i, b] = M[j, a] = 0
        history.append(obj)
    return {
        'M_optimized': M, 'objective_history': history,
        'initial_objective': history[0], 'final_objective': history[-1],
    }


def birregular_code(n_features, n_neurons, K, seed=42):
    F, N = n_features, n_neurons
    if F * K % N != 0:
        K = max(1, (F * K) // N)
        if F * K % N != 0:
            K = 1
    d = F * K // N
    if d < 1:
        d = 1
    rng = np.random.default_rng(seed)
    fstubs = np.repeat(np.arange(F), K)
    nstubs = rng.permutation(np.repeat(np.arange(N), d))
    M = np.zeros((F, N), int)
    for f, n in zip(fstubs, nstubs):
        M[f, n] += 1
    for _ in range(min(10000, F * N)):
        dup = np.argwhere(M > 1)
        if len(dup) == 0:
            break
        f, n = dup[0]
        cand = np.argwhere(M == 1)
        rng.shuffle(cand)
        swapped = False
        for f2, n2 in cand:
            if f2 != f and M[f, n2] == 0 and M[f2, n] == 0:
                M[f, n] -= 1
                M[f2, n2] -= 1
                M[f, n2] += 1
                M[f2, n] += 1
                swapped = True
                break
        if not swapped:
            break
    return M.astype(bool).astype(int)


def gauge_fix_via_codeword(gf, n_iters=200, seed=42):
    M = build_codeword_matrix(gf)
    initial_obj = overlap_objective(M)
    result = reduce_overlap_gauge(M, n_iters=n_iters, seed=seed)
    return {
        **result, 'initial_objective': initial_obj,
        'reduction_factor': initial_obj / (result['final_objective'] + 1e-30),
    }
