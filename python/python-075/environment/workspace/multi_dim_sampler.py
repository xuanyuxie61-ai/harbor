
import numpy as np


def r8vec_direct_product(factor_index, factor_order, factor_value,
                         factor_num, point_num, x):
    if factor_index == 0:
        x[:, :] = 0.0


    rep = point_num
    for j in range(factor_index + 1):
        if j < factor_index:
            pass
        rep //= factor_value.shape[0] if j == factor_index else 1


    if factor_index == 0:
        x[:, :] = 0.0
        start = 0
        skip = 1
        contig = 1
        rep = point_num
    else:

        prev_order = x.shape[1]
        for j in range(factor_index):

            pass


    if factor_index == 0:
        x[factor_index, :] = np.repeat(factor_value, point_num // factor_order)
    else:

        n_repeat = 1
        for j in range(factor_index):
            n_repeat *= factor_value.shape[0]

        n_repeat = point_num // (factor_order ** (factor_index + 1))
        if n_repeat == 0:
            n_repeat = 1
        pattern = np.tile(np.repeat(factor_value, max(1, point_num // (factor_order * (factor_index + 1)))),
                          max(1, factor_index))
        if len(pattern) < point_num:
            pattern = np.tile(pattern, int(np.ceil(point_num / len(pattern))))
        x[factor_index, :] = pattern[:point_num]

    return x


def hypercube_grid_nd(dim_num, ns, bounds, centering=None):
    if centering is None:
        centering = [1] * dim_num

    n_points = int(np.prod(ns))
    grid = np.zeros((dim_num, n_points))


    grids_1d = []
    for d in range(dim_num):
        a, b = bounds[d]
        n = ns[d]
        c = centering[d]
        if c == 1:
            if n == 1:
                g = np.array([0.5 * (a + b)])
            else:
                g = np.linspace(a, b, n)
        elif c == 2:
            g = np.linspace(a, b, n + 2)[1:-1]
        elif c == 5:
            g = ((2 * n - 2 * np.arange(1, n + 1) + 1) * a
                 + (2 * np.arange(1, n + 1) - 1) * b) / (2 * n)
        else:
            g = np.linspace(a, b, n)
        grids_1d.append(g)


    mesh = np.meshgrid(*grids_1d, indexing='ij')
    for d in range(dim_num):
        grid[d, :] = mesh[d].flatten()

    return grid


def sample_combustion_parameter_space(n_A=3, n_E=3, n_Re=3, n_phi=3):
    dim_num = 4
    ns = [n_A, n_E, n_Re, n_phi]
    bounds = [(0.5, 2.0), (0.8, 1.2), (100.0, 2000.0), (0.5, 2.0)]
    centering = [1, 1, 1, 1]
    return hypercube_grid_nd(dim_num, ns, bounds, centering)


def sensitivity_analysis_central_difference(func, base_params, delta_rel=0.01):
    base_val = func(base_params)
    if abs(base_val) < 1e-30:
        base_val = 1e-30

    sens = np.zeros_like(base_params)
    for i in range(len(base_params)):
        delta = delta_rel * abs(base_params[i])
        if delta < 1e-12:
            delta = 1e-6
        params_plus = base_params.copy()
        params_minus = base_params.copy()
        params_plus[i] += delta
        params_minus[i] -= delta
        f_plus = func(params_plus)
        f_minus = func(params_minus)
        df_dp = (f_plus - f_minus) / (2.0 * delta)
        sens[i] = df_dp * base_params[i] / base_val
    return sens
