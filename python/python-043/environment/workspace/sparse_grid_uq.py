
import numpy as np
from typing import Callable, List, Tuple








def clenshaw_curtis_rule(level: int) -> Tuple[np.ndarray, np.ndarray]:
    if level < 1:
        raise ValueError("level must be >= 1")
    if level == 1:
        return np.array([0.0]), np.array([2.0])

    n = 2 ** (level - 1) + 1

    j = np.arange(n)
    x = np.cos(np.pi * j / (n - 1))

    w = np.zeros(n, dtype=float)
    for i in range(n):
        theta = np.pi * i / (n - 1)

        coeff = 1.0
        if i == 0 or i == n - 1:
            coeff = 0.5
        val = 0.0
        for k in range(1, (n - 1) // 2 + 1):
            if 2 * k == n - 1:
                b = 1.0
            else:
                b = 2.0
            val += b * np.cos(2.0 * k * theta) / (4.0 * k * k - 1.0)
        w[i] = coeff * (2.0 / (n - 1)) * (1.0 - val)
    return x, w








def sparse_grid_cc(dim: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    from itertools import product


    rules = {}
    max_1d_level = level_max
    for lv in range(1, max_1d_level + 1):
        rules[lv] = clenshaw_curtis_rule(lv)

    points_list = []
    weights_list = []


    for total in range(level_max + 1, level_max + dim + 1):

        def compositions(d_remain, sum_remain, current):
            if d_remain == 1:
                yield current + [sum_remain]
                return
            for val in range(1, sum_remain - d_remain + 2):
                yield from compositions(d_remain - 1, sum_remain - val, current + [val])

        for comp in compositions(dim, total, []):

            diff = level_max + dim - total
            sign = (-1) ** diff
            from math import comb
            coeff = sign * comb(dim - 1, diff)


            x_lists = [rules[lv][0] for lv in comp]
            w_lists = [rules[lv][1] for lv in comp]
            for idx_tuple in product(*[range(len(xl)) for xl in x_lists]):
                pt = np.array([x_lists[j][idx_tuple[j]] for j in range(dim)])
                w = coeff
                for j in range(dim):
                    w *= w_lists[j][idx_tuple[j]]
                points_list.append(pt)
                weights_list.append(w)

    if not points_list:
        return np.zeros((0, dim)), np.zeros(0)

    points = np.array(points_list, dtype=float)
    weights = np.array(weights_list, dtype=float)



    rounded = np.round(points, decimals=12)
    uniq, inverse = np.unique(rounded, axis=0, return_inverse=True)
    uniq_weights = np.zeros(uniq.shape[0], dtype=float)
    for i in range(len(weights)):
        uniq_weights[inverse[i]] += weights[i]

    return uniq, uniq_weights






def map_parameter_space(x_ref: np.ndarray, param_ranges: List[Tuple[float, float]]) -> np.ndarray:
    x_ref = np.asarray(x_ref, dtype=float)
    d = len(param_ranges)
    if x_ref.ndim == 1:
        x_ref = x_ref.reshape(1, -1)
    p = np.zeros_like(x_ref)
    for j in range(d):
        p_min, p_max = param_ranges[j]
        p[:, j] = p_min + 0.5 * (x_ref[:, j] + 1.0) * (p_max - p_min)
    return p





def uq_dynamo_reversal_rate(
    dynamo_runner: Callable,
    param_ranges: List[Tuple[float, float]],
    level_max: int = 3
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    dim = len(param_ranges)
    points_ref, weights = sparse_grid_cc(dim, level_max)
    n_points = points_ref.shape[0]

    points_phys = map_parameter_space(points_ref, param_ranges)
    rates = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        try:
            rates[i] = float(dynamo_runner(points_phys[i]))
        except Exception:
            rates[i] = 0.0


    mean = float(np.sum(weights * rates))

    mean_sq = float(np.sum(weights * rates * rates))
    var = max(0.0, mean_sq - mean ** 2)

    return mean, var, points_phys, rates





def _self_test():

    x, w = clenshaw_curtis_rule(3)
    assert len(x) == 5
    assert abs(np.sum(w) - 2.0) < 1e-12


    pts, ws = sparse_grid_cc(2, 3)
    total_weight = np.sum(ws)
    assert abs(total_weight - 4.0) < 1e-10, f"Total weight = {total_weight}"


    pts1, ws1 = sparse_grid_cc(1, 5)
    integral = np.sum(ws1 * (pts1[:, 0] ** 2))
    assert abs(integral - 2.0 / 3.0) < 1e-10


    p = map_parameter_space(np.array([[0.0, 0.0]]), [(1.0, 3.0), (0.0, 10.0)])
    assert abs(p[0, 0] - 2.0) < 1e-10
    assert abs(p[0, 1] - 5.0) < 1e-10

    print("sparse_grid_uq: self-test passed.")


if __name__ == "__main__":
    _self_test()
