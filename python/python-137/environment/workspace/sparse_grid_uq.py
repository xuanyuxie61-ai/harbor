# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations_with_replacement


def clenshaw_curtis_abscissa(order, i):
    if order == 1:
        return 0.0
    if i < 1 or i > order:
        raise ValueError("i out of range")
    return np.cos((order - i) * np.pi / (order - 1))


def clenshaw_curtis_weights(n):
    if n == 1:
        return np.array([2.0])

    w = np.zeros(n, dtype=float)
    theta = np.array([(n - i) * np.pi / (n - 1) for i in range(1, n + 1)], dtype=float)

    for i in range(n):
        ti = theta[i]
        wi = 0.0

        for j in range(1, (n - 1) // 2 + 1):
            b = 2.0 / (4.0 * j * j - 1.0)
            if 2 * j == n - 1:
                b /= 2.0
            wi -= b * np.cos(2.0 * j * ti)
        wi += 1.0
        if i == 0 or i == n - 1:
            wi /= 2.0
        wi *= 2.0 / (n - 1.0)
        w[i] = wi

    return w


def comp_next(n, k):
    if k == 1:
        return [(n,)]
    result = []
    def helper(remaining, parts, start):
        if len(parts) == k - 1:
            result.append(tuple(parts + [remaining]))
            return
        for i in range(remaining, -1, -1):
            helper(remaining - i, parts + [i], i)
    helper(n, [], n)
    return result


def sparse_grid_cc(dim_num, level_max):
    if dim_num <= 0 or level_max < 0:
        return np.zeros((0, max(1, dim_num))), np.zeros(0)



    def level_to_order(level):
        if level == 0:
            return 1
        return 2 ** level + 1


    L = level_max
    all_points = []
    all_weights = []

    from math import comb
    for total_level in range(0, L + 1):
        for comp in comp_next(total_level, dim_num):
            level_sum = sum(comp)


            coeff = (-1) ** (L - level_sum)
            coeff *= comb(dim_num - 1, L - level_sum)


            orders = [level_to_order(l) for l in comp]

            grids_1d = []
            weights_1d = []
            for order in orders:
                x = np.array([clenshaw_curtis_abscissa(order, i + 1) for i in range(order)])
                w = clenshaw_curtis_weights(order)
                grids_1d.append(x)
                weights_1d.append(w)


            from itertools import product
            for idx_tuple in product(*[range(len(g)) for g in grids_1d]):
                pt = np.array([grids_1d[d][idx_tuple[d]] for d in range(dim_num)])
                wt = coeff
                for d in range(dim_num):
                    wt *= weights_1d[d][idx_tuple[d]]
                all_points.append(pt)
                all_weights.append(wt)

    if not all_points:
        return np.zeros((0, dim_num)), np.zeros(0)

    points = np.array(all_points)
    weights = np.array(all_weights)



    tol = 1e-12
    unique_indices = []
    unique_points = []
    unique_weights = []

    for i in range(len(points)):
        found = False
        for j, up in enumerate(unique_points):
            if np.all(np.abs(points[i] - up) < tol):
                unique_weights[j] += weights[i]
                found = True
                break
        if not found:
            unique_points.append(points[i].copy())
            unique_weights.append(weights[i])

    points = np.array(unique_points)
    weights = np.array(unique_weights)

    return points, weights


def sparse_grid_integrate(func, dim_num, level_max):
    points, weights = sparse_grid_cc(dim_num, level_max)
    if points.size == 0:
        return 0.0
    f_vals = func(points)
    f_vals = np.asarray(f_vals, dtype=float)
    return float(np.dot(weights, f_vals))


def uncertainty_quantification_crystallization(model_func, param_distributions,
                                                level_max=3):
    dim_num = len(param_distributions)
    points, weights = sparse_grid_cc(dim_num, level_max)

    if points.size == 0:
        return 0.0, 0.0, 0.0


    means = np.array([d['mean'] for d in param_distributions])
    stds = np.array([d['std'] for d in param_distributions])


    lows = means - 3.0 * stds
    highs = means + 3.0 * stds

    actual_points = lows + 0.5 * (highs - lows) * (points + 1.0)

    outputs = np.array([model_func(p) for p in actual_points])
    mean_val = float(np.dot(weights, outputs))
    mean_sq = float(np.dot(weights, outputs ** 2))
    variance = max(mean_sq - mean_val ** 2, 0.0)
    std_val = np.sqrt(variance)

    return mean_val, variance, std_val
