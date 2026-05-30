
import numpy as np
from typing import List, Tuple, Callable, Optional
from math import comb as nchoosek


def hermite_abscissa(order: int) -> np.ndarray:
    from numpy.polynomial.hermite import hermgauss
    if order <= 0:
        return np.array([0.0])
    x, _ = hermgauss(order)
    return x


def hermite_weights(order: int) -> np.ndarray:
    from numpy.polynomial.hermite import hermgauss
    if order <= 0:
        return np.array([np.sqrt(np.pi)])
    _, w = hermgauss(order)
    return w


def comp_next(n: int, k: int, a: Optional[np.ndarray] = None,
              more: bool = False, h: int = 0, t: int = 0) -> Tuple[np.ndarray, bool, int, int]:
    if a is None or not more:
        a = np.zeros(k, dtype=int)
        a[0] = n
        h = 0
        t = n
        more = (a[-1] != n)
        return a, more, h, t

    if 1 < t:
        h = 0
    h += 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] += 1
    more = (a[-1] != n)
    return a, more, h, t


def level_to_order_open(dim_num: int, level_1d: np.ndarray) -> np.ndarray:
    level_1d = np.asarray(level_1d, dtype=int)
    order = 2 * level_1d + 1
    order = np.maximum(order, 1)
    return order


def product_weight_herm(dim_num: int, order_1d: np.ndarray, order_nd: int) -> np.ndarray:
    weights = np.ones(order_nd)




    grids = [hermite_weights(int(o)) for o in order_1d]

    if dim_num == 1:
        return grids[0]


    w_curr = grids[0]
    for d in range(1, dim_num):
        w_new = []
        for wi in w_curr:
            for wj in grids[d]:
                w_new.append(wi * wj)
        w_curr = np.array(w_new)

    return w_curr


def multigrid_index_z(dim_num: int, order_1d: np.ndarray, order_nd: int) -> np.ndarray:
    grids = []
    for d in range(dim_num):
        od = int(order_1d[d])
        base = (od - 1) // 2
        grids.append(np.arange(-base, base + 1))

    if dim_num == 1:
        return grids[0].reshape(1, -1)


    mesh = np.array(np.meshgrid(*grids, indexing='ij'))
    indices = mesh.reshape(dim_num, -1)
    return indices


def sparse_grid_herm_size(dim_num: int, level_max: int) -> int:
    point_num = 0
    level_min = max(0, level_max + 1 - dim_num)
    for level in range(level_min, level_max + 1):
        a = None
        more = False
        h = 0
        t = 0
        while True:
            a, more, h, t = comp_next(level, dim_num, a, more, h, t)
            order_1d = level_to_order_open(dim_num, a)
            point_num += int(np.prod(order_1d))
            if not more:
                break
    return point_num


def sparse_grid_hermite(dim_num: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    point_num = sparse_grid_herm_size(dim_num, level_max)
    grid_point = np.zeros((dim_num, point_num))
    grid_weight = np.zeros(point_num)
    point_num2 = 0

    level_min = max(0, level_max + 1 - dim_num)

    for level in range(level_min, level_max + 1):
        level_1d = None
        more = False
        h = 0
        t = 0

        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = level_to_order_open(dim_num, level_1d)
            order_nd = int(np.prod(order_1d))


            w2 = product_weight_herm(dim_num, order_1d, order_nd)


            coeff = ((-1) ** (level_max - level)
                     * nchoosek(dim_num - 1, level_max - level))


            idx = multigrid_index_z(dim_num, order_1d, order_nd)
            base2 = np.round((order_1d - 1) / 2).astype(int)

            for pt in range(order_nd):

                pt_coord = np.zeros(dim_num)
                for d in range(dim_num):
                    abs_idx = int(idx[d, pt])

                    herm_x = hermite_abscissa(int(order_1d[d]))

                    map_idx = abs_idx + base2[d]
                    map_idx = max(0, min(map_idx, int(order_1d[d]) - 1))
                    pt_coord[d] = herm_x[map_idx]


                found = False
                for pt2 in range(point_num2):
                    if np.allclose(grid_point[:, pt2], pt_coord, atol=1.0e-10):
                        grid_weight[pt2] += coeff * w2[pt]
                        found = True
                        break

                if not found:
                    grid_point[:, point_num2] = pt_coord
                    grid_weight[point_num2] = coeff * w2[pt]
                    point_num2 += 1

            if not more:
                break


    grid_point = grid_point[:, :point_num2]
    grid_weight = grid_weight[:point_num2]
    return grid_point, grid_weight


def propagate_uncertainty(model_func: Callable[[np.ndarray], float],
                          dim_num: int,
                          level_max: int = 3,
                          param_means: Optional[np.ndarray] = None,
                          param_stds: Optional[np.ndarray] = None) -> dict:
    if param_means is None:
        param_means = np.zeros(dim_num)
    if param_stds is None:
        param_stds = np.ones(dim_num)

    points, weights = sparse_grid_hermite(dim_num, level_max)
    n_points = points.shape[1]

    values = np.zeros(n_points)
    for i in range(n_points):
        xi = points[:, i]

        params = param_means + param_stds * xi
        try:
            values[i] = model_func(params)
        except Exception:
            values[i] = 0.0


    total_weight = np.sum(weights)
    if abs(total_weight) < 1.0e-15:
        total_weight = 1.0
    weights_norm = weights / total_weight

    mean_val = np.dot(weights_norm, values)
    var_val = np.dot(weights_norm, values ** 2) - mean_val ** 2
    var_val = max(var_val, 0.0)
    std_val = np.sqrt(var_val)


    if std_val > 1.0e-12:
        skew = np.dot(weights_norm, (values - mean_val) ** 3) / (std_val ** 3)
        kurt = np.dot(weights_norm, (values - mean_val) ** 4) / (std_val ** 4)
    else:
        skew = 0.0
        kurt = 3.0

    return {
        'mean': mean_val,
        'variance': var_val,
        'std': std_val,
        'skewness': skew,
        'kurtosis': kurt,
        'points': points,
        'weights': weights_norm,
        'values': values,
    }


def sensitivity_index_sobol(values: np.ndarray,
                            weights: np.ndarray,
                            points: np.ndarray,
                            dim_num: int) -> np.ndarray:
    total_var = np.dot(weights, (values - np.dot(weights, values)) ** 2)
    total_var = max(total_var, 1.0e-15)

    S1 = np.zeros(dim_num)
    for d in range(dim_num):

        unique_vals = np.unique(np.round(points[d, :], 6))
        cond_var = 0.0
        for uv in unique_vals:
            mask = np.isclose(points[d, :], uv, atol=1.0e-5)
            if np.sum(mask) == 0:
                continue
            w_sub = weights[mask]
            y_sub = values[mask]
            w_sum = np.sum(w_sub)
            if w_sum < 1.0e-15:
                continue
            cond_mean = np.dot(w_sub, y_sub) / w_sum
            cond_var += w_sum * cond_mean ** 2

        S1[d] = (cond_var - np.dot(weights, values) ** 2) / total_var
        S1[d] = max(0.0, min(1.0, S1[d]))

    return S1
