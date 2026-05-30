import numpy as np
from scipy.special import comb


def laguerre_abscissa_and_weights(n):
    x, w = np.polynomial.laguerre.laggauss(n)
    return x, w


def level_to_order_open(level_1d):
    order = 2 ** (level_1d + 1) - 1
    return order


def _integer_compositions(n, k):
    if k == 1:
        yield np.array([n], dtype=int)
        return

    for first in range(n + 1):
        for rest in _integer_compositions(n - first, k - 1):
            yield np.concatenate(([first], rest))


def sparse_grid_laguerre(dim_num, level_max):
    level_min = max(0, level_max + 1 - dim_num)
    grid_points_list = []
    grid_weights_list = []

    for level in range(level_min, level_max + 1):
        coeff = ((-1) ** (level_max - level)) * int(comb(dim_num - 1, level_max - level))

        for level_1d in _integer_compositions(level, dim_num):
            order_1d = level_to_order_open(level_1d)
            order_nd = int(np.prod(order_1d))
            if order_nd == 0:
                continue


            indices = np.zeros((order_nd, dim_num), dtype=int)
            for d in range(dim_num):
                if d == 0:
                    repeats = 1
                    tiles = order_nd // order_1d[d]
                else:
                    repeats = repeats * order_1d[d - 1]
                    tiles = order_nd // (repeats * order_1d[d])
                indices[:, d] = np.tile(np.repeat(np.arange(order_1d[d]), repeats), tiles)

            for pt in range(order_nd):
                pt_coords = np.zeros(dim_num, dtype=float)
                w = float(coeff)
                for d in range(dim_num):
                    n = order_1d[d]
                    x, wg = laguerre_abscissa_and_weights(n)
                    pt_coords[d] = x[indices[pt, d]]
                    w *= wg[indices[pt, d]]
                grid_points_list.append(pt_coords)
                grid_weights_list.append(w)

    if len(grid_points_list) == 0:
        return np.zeros((dim_num, 0)), np.zeros(0)
    grid_points = np.column_stack(grid_points_list)
    grid_weights = np.array(grid_weights_list, dtype=float)
    return grid_points, grid_weights


def propagate_uncertainty(model_func, dim_num, level_max,
                          param_means, param_stds):
    points, weights = sparse_grid_laguerre(dim_num, level_max)
    n = points.shape[1]

    samples = np.zeros((dim_num, n), dtype=float)
    for d in range(dim_num):


        samples[d, :] = param_means[d] + param_stds[d] * (points[d, :] - 1.0)
        samples[d, :] = np.clip(samples[d, :], param_means[d] * 0.1, param_means[d] * 3.0)

    vals = model_func(samples.T)
    vals = np.asarray(vals, dtype=float)
    mean = np.sum(weights * vals)
    var = np.sum(weights * vals ** 2) - mean ** 2
    return mean, max(var, 0.0), np.sqrt(max(var, 0.0))
