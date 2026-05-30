
import numpy as np
from typing import Callable, List, Tuple
from quadrature_rules import clenshaw_curtis_compute, jacobi_compute, laguerre_quadrature_rule
from quadrature_rules import integrate_triangle





def comp_next(n: int, k: int, a: np.ndarray, more: bool, h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
    if not more:
        a = np.zeros(k, dtype=int)
        a[0] = n
        h = 0
        t = n
        more = (a[-1] != n)
        return a, more, h, t

    if 1 < t:
        h = 0
    h += 1


    if h >= k:
        more = False
        return a, more, h, t




    t_val = a[h - 1]
    a[h - 1] = 0
    a[0] = t_val - 1
    a[h] += 1
    t = t_val

    more = (a[-1] != n)
    return a, more, h, t


def level_to_order_closed(level: int) -> int:
    if level == 0:
        return 1
    return 2 ** level + 1





def sparse_grid_total_poly_size(dim_num: int, level_max: int) -> int:
    if dim_num < 1:
        raise ValueError("sparse_grid_total_poly_size: dim_num must be >= 1.")
    if level_max < 0:
        return 0
    if level_max == 0:
        return 1

    new_1d = np.zeros(level_max + 1, dtype=int)
    new_1d[0] = 1
    if level_max >= 1:
        new_1d[1] = 2
    for l in range(2, level_max + 1):
        new_1d[l] = 2 ** (l - 1)

    point_num = 0
    for level in range(level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            point_num += int(np.prod(new_1d[level_1d]))
            if not more:
                break

    return point_num





def sparse_grid_total_poly_index(dim_num: int, level_max: int) -> np.ndarray:
    point_num = sparse_grid_total_poly_size(dim_num, level_max)
    if point_num == 0:
        return np.zeros((dim_num, 0), dtype=int)

    new_1d = np.zeros(level_max + 1, dtype=int)
    new_1d[0] = 1
    if level_max >= 1:
        new_1d[1] = 2
    for l in range(2, level_max + 1):
        new_1d[l] = 2 ** (l - 1)


    all_indices = []
    for level in range(level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)

            orders = np.array([level_to_order_closed(ld) for ld in level_1d], dtype=int)


            prod_points = int(np.prod(orders))
            idx_array = np.zeros((dim_num, prod_points), dtype=int)

            grids = [np.arange(o) for o in orders]
            mesh = np.array(np.meshgrid(*grids, indexing='ij'))
            idx_array = mesh.reshape(dim_num, -1)
            all_indices.append(idx_array)
            if not more:
                break

    if not all_indices:
        return np.zeros((dim_num, 0), dtype=int)

    combined = np.hstack(all_indices)

    unique_cols = np.unique(combined, axis=1)
    return unique_cols





def sparse_grid_integrate(
    dim_num: int,
    level_max: int,
    f: Callable[[np.ndarray], np.ndarray],
    rule: str = "clenshaw-curtis",
) -> float:
    if dim_num < 1:
        raise ValueError("sparse_grid_integrate: dim_num must be >= 1.")
    if level_max < 0:
        return 0.0


    max_order = level_to_order_closed(level_max)
    if rule == "clenshaw-curtis":
        x_1d, w_1d = clenshaw_curtis_compute(max_order)
    elif rule == "jacobi":
        x_1d, w_1d = jacobi_compute(max_order, 0.0, 0.0)
    elif rule == "laguerre":
        x_1d, w_1d = laguerre_quadrature_rule(max_order)

    else:
        raise ValueError("sparse_grid_integrate: unknown rule.")



    if dim_num <= 3 and level_max <= 4:
        orders = [level_to_order_closed(level_max)] * dim_num
        grids_x = [x_1d[:o] for o in orders]
        grids_w = [w_1d[:o] for o in orders]

        mesh = np.array(np.meshgrid(*grids_x, indexing='ij'))
        points = mesh.reshape(dim_num, -1)

        weight_mesh = np.array(np.meshgrid(*grids_w, indexing='ij'))
        weights = np.prod(weight_mesh.reshape(dim_num, -1), axis=0)

        vals = f(points)
        result = float(np.dot(weights, vals))
        return result


    grid_index = sparse_grid_total_poly_index(dim_num, level_max)
    n_pts = grid_index.shape[1]
    points = np.zeros((dim_num, n_pts), dtype=float)
    weights = np.ones(n_pts, dtype=float)

    for d in range(dim_num):
        max_o = level_to_order_closed(level_max)
        points[d, :] = x_1d[grid_index[d, :] % max_o]
        weights *= w_1d[grid_index[d, :] % max_o]

    vals = f(points)
    result = float(np.dot(weights, vals))
    return result





def thermodynamic_integration_binding_free_energy(
    n_lambda: int = 11,
    temperature: float = 300.0,
    dim_conformational: int = 3,
    sg_level: int = 3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if n_lambda < 2:
        raise ValueError("thermodynamic_integration_binding_free_energy: n_lambda >= 2.")
    if temperature <= 0:
        raise ValueError("thermodynamic_integration_binding_free_energy: temperature > 0.")
    if dim_conformational < 1:
        raise ValueError("thermodynamic_integration_binding_free_energy: dim_conformational >= 1.")








    raise NotImplementedError("Hole 2: Thermodynamic integration core not implemented.")





def membrane_surface_free_energy(
    triangles: List[np.ndarray],
    energy_density: Callable[[np.ndarray], np.ndarray],
    rule_index: int = 2,
) -> float:
    total = 0.0
    for tri in triangles:
        total += integrate_triangle(energy_density, tri, rule_index)
    return total
