"""
Sparse Grid Quadrature and Interpolation Module (Smolyak Construction).

Integrates:
  - 1277_toms847: Smolyak sparse grid construction, hierarchical surpluses
  - 344_exactness: 1D quadrature rules (Clenshaw-Curtis, Gauss-Legendre)

Scientific formulas:
  Smolyak combination formula for level L in d dimensions:
    A_{L,d} = sum_{|l|_1 <= L+d-1} (Delta_{l_1} x ... x Delta_{l_d})

  where Delta_l = Q_l - Q_{l-1} is the difference between successive 1D rules.

  Hierarchical surplus on Clenshaw-Curtis grid:
    xi_j^{(l)} = cos( (j-1) * pi / (n_l - 1) ),  j = 1..n_l
    n_l = 2^{l-1} + 1  (number of points at level l)
"""

import numpy as np
import math


def clenshaw_curtis_nodes(n):
    """
    Generate n-point Clenshaw-Curtis nodes on [-1, 1]:
      x_j = cos( (j-1) * pi / (n-1) ),  j = 1..n
    For n=1: x_1 = 0.
    """
    if n == 1:
        return np.array([0.0])
    j = np.arange(n)
    return np.cos(j * math.pi / (n - 1))


def clenshaw_curtis_weights(n):
    """
    Generate n-point Clenshaw-Curtis quadrature weights.
    Using the DCT-based formulation.
    """
    if n == 1:
        return np.array([2.0])
    theta = np.arange(n) * math.pi / (n - 1)
    w = np.ones(n)
    v = np.ones(n - 2)
    for k in range(1, (n - 1) // 2 + 1):
        if 2 * k == n - 1:
            coeff = 1.0
        else:
            coeff = 2.0
        v -= coeff * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
    w[0] = 1.0 / (n - 1)
    w[1:-1] = 2.0 * v / (n - 1)
    w[-1] = 1.0 / (n - 1)
    return w


def level_to_n_cc(l):
    """Number of Clenshaw-Curtis points at level l (l >= 1)."""
    if l == 1:
        return 1
    return 2 ** (l - 1) + 1


def spgetseq(d, n):
    """
    Generate all d-dimensional index vectors with sum = n.
    From seed 1277_toms847 (spgetseq).

    Returns list of tuples, each of length d.
    """
    if d == 1:
        return [(n,)]
    result = []
    for i in range(n + 1):
        sub = spgetseq(d - 1, n - i)
        for s in sub:
            result.append((i,) + s)
    return result


def sparse_grid_points_weights(d, L):
    """
    Construct a sparse grid of dimension d and level L using
    the Smolyak combination with Clenshaw-Curtis 1D rules.

    Returns:
      points : (N, d) array of sparse grid points in [-1,1]^d
      weights: (N,) array of sparse grid weights
    """
    if d <= 0:
        return np.zeros((1, 0)), np.ones(1)
    if L < 1:
        L = 1

    # Build 1D rules for each level
    max_level = L + d - 1
    rules = {}
    for l in range(1, max_level + 1):
        n = level_to_n_cc(l)
        rules[l] = (clenshaw_curtis_nodes(n), clenshaw_curtis_weights(n))

    # Smolyak combination
    point_dict = {}
    for n in range(d, max_level + 1):
        seqs = spgetseq(d, n - d)
        for l_vec in seqs:
            # Compute coefficient: (-1)^{max_level - n} * C(d-1, max_level - n)
            coeff = (-1) ** (max_level - n) * math.comb(d - 1, max_level - n)
            # Get 1D nodes and weights for this level combination
            nodes_1d = [rules[l + 1][0] for l in l_vec]
            weights_1d = [rules[l + 1][1] for l in l_vec]
            # Tensor product
            grids = [g.ravel() for g in np.meshgrid(*nodes_1d, indexing='ij')]
            wgrids = [g.ravel() for g in np.meshgrid(*weights_1d, indexing='ij')]
            npts = len(grids[0])
            for k in range(npts):
                pt = tuple(round(grids[dim][k], 14) for dim in range(d))
                w = coeff * np.prod([wgrids[dim][k] for dim in range(d)])
                if pt in point_dict:
                    point_dict[pt] += w
                else:
                    point_dict[pt] = w

    points = np.array([pt for pt in point_dict.keys()])
    weights = np.array([point_dict[pt] for pt in point_dict.keys()])
    return points, weights


def sparse_grid_integrate(func, d, L):
    """
    Integrate func over [-1,1]^d using sparse grid quadrature.

    I = sum_i w_i * func(x_i)
    """
    pts, w = sparse_grid_points_weights(d, L)
    total = 0.0
    for i in range(len(w)):
        total += w[i] * func(pts[i, :])
    return total


def hierarchical_surplus_1d(values, levels):
    """
    Compute hierarchical surpluses for a 1D function on nested
    Clenshaw-Curtis grids.

    The surplus at a new point is the function value minus the
    interpolant from all coarser levels.
    """
    # For simplicity, construct piecewise linear interpolant on nested grids
    surpluses = []
    all_pts = []
    for l in range(1, max(levels) + 1):
        n = level_to_n_cc(l)
        pts = clenshaw_curtis_nodes(n)
        if l == 1:
            surpluses.append(values[0])
            all_pts.append(pts[0])
        else:
            # New points are the odd-index points
            new_pts = pts[1:-1:2]
            for j, xp in enumerate(new_pts):
                # Linear interpolation from previous level
                idx = np.searchsorted(all_pts, xp)
                if idx == 0:
                    interp = surpluses[0]
                elif idx >= len(all_pts):
                    interp = surpluses[-1]
                else:
                    x0, x1 = all_pts[idx - 1], all_pts[idx]
                    t = (xp - x0) / (x1 - x0) if abs(x1 - x0) > 1e-15 else 0.0
                    interp = surpluses[idx - 1] * (1 - t) + surpluses[idx] * t
                surpluses.append(values[len(all_pts)] - interp)
                all_pts.append(xp)
    return np.array(all_pts), np.array(surpluses)


def adaptive_sparse_grid_refine(func, d, L_max=5, abs_tol=1e-6, rel_tol=1e-4):
    """
    Adaptive sparse grid refinement.
    Iteratively refine until max hierarchical surplus <= tolerance.
    From seed 1277_toms847 adaptive strategy.
    """
    L = 1
    while L <= L_max:
        pts, w = sparse_grid_points_weights(d, L)
        n = len(w)
        vals = np.array([func(pts[i, :]) for i in range(n)])
        fmin, fmax = vals.min(), vals.max()
        # Estimate surplus as difference between L and L-1 integrals
        I_L = np.dot(w, vals)
        if L > 1:
            pts_old, w_old = sparse_grid_points_weights(d, L - 1)
            # Re-evaluate at old points (for consistent comparison)
            vals_old = np.array([func(pts_old[i, :]) for i in range(len(w_old))])
            I_old = np.dot(w_old, vals_old)
            surplus = abs(I_L - I_old)
            tol = max(rel_tol * (fmax - fmin), abs_tol)
            if surplus <= tol:
                return I_L, pts, w, vals, L
        L += 1
    return I_L, pts, w, vals, L
