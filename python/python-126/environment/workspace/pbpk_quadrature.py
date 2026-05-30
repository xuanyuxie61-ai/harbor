
import numpy as np
from typing import List, Tuple
from itertools import product
import math





def cc_abscissa(level: int) -> np.ndarray:
    if level < 0:
        raise ValueError("CC level must be non-negative")
    if level == 0:
        return np.array([0.0])
    order = 2 ** level + 1
    k = np.arange(order)
    x = np.cos(k * np.pi / (order - 1))

    x[0] = 1.0
    x[-1] = -1.0
    return x


def cc_weights(level: int) -> np.ndarray:
    if level < 0:
        raise ValueError("CC level must be non-negative")
    if level == 0:
        return np.array([2.0])
    n = 2 ** level
    j = np.arange(n + 1)

    V = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        V[k, :] = np.cos(k * j * np.pi / n)

    b = np.zeros(n + 1)
    b[0] = 2.0
    for k in range(2, n + 1, 2):
        b[k] = 2.0 / (1.0 - k * k)

    w = np.linalg.solve(V.T, b)
    return w






def comp_next(n: int, k: int, a: np.ndarray, more: bool, h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
    if not more:
        a[:] = 0
        a[0] = n
        h = 0
        t = n
        more = True if 1 < k else False
        return a, more, h, t
    if 1 < t:
        h = 0
    h = h + 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1
    if 1 < t:
        more = True
    else:
        more = False
    if not more and 1 < k:
        more = a[k - 1] != n
    return a, more, h, t


def sparse_grid_cc_index(dim: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    if dim < 1 or level_max < 0:
        raise ValueError("Invalid dimension or level")

    index_list = []
    base_list = []
    a = np.zeros(dim, dtype=int)
    more = False
    h = 0
    t = 0
    a, more, h, t = comp_next(level_max, dim, a, more, h, t)
    while True:
        level_sum = a.sum()
        if level_sum <= level_max:

            b = np.zeros(dim, dtype=int)
            b_more = False
            b_h = 0
            b_t = 0
            b, b_more, b_h, b_t = comp_next(level_sum, dim, b, b_more, b_h, b_t)
            while True:

                level_min = level_max - level_sum + dim - 1

                orders = []
                for i in range(dim):
                    if b[i] == 0:
                        orders.append([0])
                    else:
                        orders.append(list(range(2 ** b[i] + 1)))

                for idx in product(*orders):
                    index_list.append(list(b))
                    base_list.append(list(idx))
                if not b_more:
                    break
                b, b_more, b_h, b_t = comp_next(level_sum, dim, b, b_more, b_h, b_t)
        if not more:
            break
        a, more, h, t = comp_next(level_max, dim, a, more, h, t)



    return _generate_sparse_grid_direct(dim, level_max)


def _generate_sparse_grid_direct(dim: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    point_dict = {}
    for q in range(level_max + 1):

        def recurse(levels, idx, remaining):
            if idx == dim - 1:
                levels[idx] = remaining
                yield levels.copy()
                return
            for val in range(remaining + 1):
                levels[idx] = val
                yield from recurse(levels, idx + 1, remaining - val)
        for levels in recurse(np.zeros(dim, dtype=int), 0, q):

            coeff = ((-1) ** (level_max - q)) * math.comb(dim - 1, level_max - q)

            nodes_1d = [cc_abscissa(lv) for lv in levels]
            weights_1d = [cc_weights(lv) for lv in levels]
            for multi_idx in product(*[range(len(n)) for n in nodes_1d]):
                pt = tuple(nodes_1d[i][multi_idx[i]] for i in range(dim))
                w = coeff
                for i in range(dim):
                    w *= weights_1d[i][multi_idx[i]]
                if pt in point_dict:
                    point_dict[pt] += w
                else:
                    point_dict[pt] = w

    points = np.array(list(point_dict.keys()))
    weights = np.array(list(point_dict.values()))
    return points, weights






def line_unit_o01() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0])
    w = np.array([2.0])
    return x, w


def line_unit_o03() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([-np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0])
    w = np.array([1.0, 1.0])
    return x, w


def line_unit_o05() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([-np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
                  -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
                   0.0,
                   np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
                   np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0])
    w = np.array([(322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
                  (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
                  128.0 / 225.0,
                  (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
                  (322.0 - 13.0 * np.sqrt(70.0)) / 900.0])
    return x, w


def square_felippa_rule(a1: float, b1: float, a2: float, b2: float,
                        order: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order == 1:
        x1d, w1d = line_unit_o01()
    elif order == 3:
        x1d, w1d = line_unit_o03()
    elif order == 5:
        x1d, w1d = line_unit_o05()
    else:
        raise ValueError("Supported orders: 1, 3, 5")

    n1, n2 = len(x1d), len(x1d)

    x1_mapped = 0.5 * (b1 - a1) * x1d + 0.5 * (b1 + a1)
    x2_mapped = 0.5 * (b2 - a2) * x1d + 0.5 * (b2 + a2)
    jacobian = 0.25 * (b1 - a1) * (b2 - a2)

    x_nodes = []
    y_nodes = []
    w_nodes = []
    for i in range(n1):
        for j in range(n2):
            x_nodes.append(x1_mapped[i])
            y_nodes.append(x2_mapped[j])
            w_nodes.append(w1d[i] * w1d[j] * jacobian)
    return np.array(x_nodes), np.array(y_nodes), np.array(w_nodes)


def square_monomial_integral(a1: float, b1: float, a2: float, b2: float,
                              p: int, q: int) -> float:
    if p < 0 or q < 0:
        raise ValueError("Monomial exponents must be non-negative")
    val_x = (b1 ** (p + 1) - a1 ** (p + 1)) / (p + 1.0)
    val_y = (b2 ** (q + 1) - a2 ** (q + 1)) / (q + 1.0)
    return val_x * val_y






def sparse_grid_integrate(func, dim: int, level_max: int):
    points, weights = _generate_sparse_grid_direct(dim, level_max)
    total = 0.0
    for pt, w in zip(points, weights):
        total += w * func(pt)
    return total


def integrate_organ_slice(concentration_func, xlim: Tuple[float, float],
                           ylim: Tuple[float, float], order: int = 5) -> float:
    x, y, w = square_felippa_rule(xlim[0], xlim[1], ylim[0], ylim[1], order)
    vals = np.array([concentration_func(xi, yi) for xi, yi in zip(x, y)])
    return np.sum(w * vals)






if __name__ == "__main__":

    x = cc_abscissa(2)
    w = cc_weights(2)
    print(f"CC level 2: sum(w)={w.sum():.10f}, nodes={x}")

    dim, level = 2, 2
    pts, wts = _generate_sparse_grid_direct(dim, level)
    print(f"Sparse grid dim={dim}, level={level}: {len(pts)} points, sum(w)={wts.sum():.10f}")

    result = sparse_grid_integrate(lambda p: p[0]**2 * p[1]**2, 2, 3)
    print(f"Sparse grid integral of x^2 y^2: {result:.10f} (exact: {4.0/9.0:.10f})")

    xn, yn, wn = square_felippa_rule(0.0, 1.0, 0.0, 1.0, 5)
    integral = np.sum(wn)
    print(f"Felippa integral over unit square: {integral:.10f} (exact: 1.0)")
