
import numpy as np
import math


def clenshaw_curtis_nodes(n):
    if n == 1:
        return np.array([0.0])
    j = np.arange(n)
    return np.cos(j * math.pi / (n - 1))


def clenshaw_curtis_weights(n):
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
    if l == 1:
        return 1
    return 2 ** (l - 1) + 1


def spgetseq(d, n):
    if d == 1:
        return [(n,)]
    result = []
    for i in range(n + 1):
        sub = spgetseq(d - 1, n - i)
        for s in sub:
            result.append((i,) + s)
    return result


def sparse_grid_points_weights(d, L):
    if d <= 0:
        return np.zeros((1, 0)), np.ones(1)
    if L < 1:
        L = 1


    max_level = L + d - 1
    rules = {}
    for l in range(1, max_level + 1):
        n = level_to_n_cc(l)
        rules[l] = (clenshaw_curtis_nodes(n), clenshaw_curtis_weights(n))


    point_dict = {}
    for n in range(d, max_level + 1):
        seqs = spgetseq(d, n - d)
        for l_vec in seqs:

            coeff = (-1) ** (max_level - n) * math.comb(d - 1, max_level - n)

            nodes_1d = [rules[l + 1][0] for l in l_vec]
            weights_1d = [rules[l + 1][1] for l in l_vec]

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
    pts, w = sparse_grid_points_weights(d, L)
    total = 0.0
    for i in range(len(w)):
        total += w[i] * func(pts[i, :])
    return total


def hierarchical_surplus_1d(values, levels):

    surpluses = []
    all_pts = []
    for l in range(1, max(levels) + 1):
        n = level_to_n_cc(l)
        pts = clenshaw_curtis_nodes(n)
        if l == 1:
            surpluses.append(values[0])
            all_pts.append(pts[0])
        else:

            new_pts = pts[1:-1:2]
            for j, xp in enumerate(new_pts):

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
    L = 1
    while L <= L_max:
        pts, w = sparse_grid_points_weights(d, L)
        n = len(w)
        vals = np.array([func(pts[i, :]) for i in range(n)])
        fmin, fmax = vals.min(), vals.max()

        I_L = np.dot(w, vals)
        if L > 1:
            pts_old, w_old = sparse_grid_points_weights(d, L - 1)

            vals_old = np.array([func(pts_old[i, :]) for i in range(len(w_old))])
            I_old = np.dot(w_old, vals_old)
            surplus = abs(I_L - I_old)
            tol = max(rel_tol * (fmax - fmin), abs_tol)
            if surplus <= tol:
                return I_L, pts, w, vals, L
        L += 1
    return I_L, pts, w, vals, L
