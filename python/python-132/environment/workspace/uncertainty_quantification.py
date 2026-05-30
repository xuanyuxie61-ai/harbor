
import numpy as np
from utils import ensure_positive






def hexagon01_sample(n):
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    count = 0
    while count < n:

        xr = np.random.uniform(-1.0, 1.0, size=n)
        yr = np.random.uniform(-np.sqrt(3.0)/2.0, np.sqrt(3.0)/2.0, size=n)

        inside = (np.abs(xr) <= 1.0) & (np.abs(yr) <= np.sqrt(3.0)/2.0) & \
                 (np.abs(xr) + np.abs(yr) / np.sqrt(3.0) <= 1.0)
        valid = np.where(inside)[0]
        take = min(len(valid), n - count)
        x[count:count+take] = xr[valid[:take]]
        y[count:count+take] = yr[valid[:take]]
        count += take
    return x, y


def hexagon01_area():
    return 3.0 * np.sqrt(3.0) / 2.0


def hexagon_monte_carlo_integrate(f_func, n_samples=10000):
    area = hexagon01_area()
    x, y = hexagon01_sample(n_samples)
    vals = f_func(x, y)
    vals = np.asarray(vals, dtype=float)
    return float(area * np.mean(vals))






def rcont_random_table(nrow, ncol, nrowt, ncolt, seed=None):
    if seed is not None:
        np.random.seed(seed)

    nrowt = np.asarray(nrowt, dtype=int)
    ncolt = np.asarray(ncolt, dtype=int)

    if np.sum(nrowt) != np.sum(ncolt):

        diff = np.sum(nrowt) - np.sum(ncolt)
        if diff > 0:
            ncolt[-1] += diff
        else:
            nrowt[-1] -= diff

    ntotal = np.sum(nrowt)
    nvect = np.arange(1, ntotal + 1)


    perm = np.random.permutation(ntotal)
    nvect_perm = nvect[perm]


    nsubt = np.cumsum(ncolt)

    matrix = np.zeros((nrow, ncol), dtype=int)
    ii = 0
    for i in range(nrow):
        limit = nrowt[i]
        for k in range(limit):
            for j in range(ncol):
                if nvect_perm[ii] <= nsubt[j]:
                    ii += 1
                    matrix[i, j] += 1
                    break

    return matrix


def random_flow_distribution(n_trays, nc, total_flows, component_totals, n_samples=5, seed=42):
    total_flows = np.asarray(total_flows, dtype=int)
    component_totals = np.asarray(component_totals, dtype=int)


    scale = 1000
    total_flows_i = (total_flows * scale).astype(int)
    component_totals_i = (component_totals * scale).astype(int)

    samples = []
    for s in range(n_samples):
        mat = rcont_random_table(n_trays, nc, total_flows_i, component_totals_i, seed=seed + s)
        samples.append(mat.astype(float) / scale)

    return samples






def sobol_first_order_index_mc(model_func, param_names, param_ranges,
                                n_samples=2048):
    n_params = len(param_names)


    A = np.random.rand(n_samples, n_params)
    B = np.random.rand(n_samples, n_params)


    def map_params(X):
        params = {}
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params[name] = low + X[:, i] * (high - low)
        return params

    params_A = map_params(A)
    params_B = map_params(B)

    Y_A = np.array([model_func({name: params_A[name][j] for name in param_names})
                    for j in range(n_samples)], dtype=float)
    Y_B = np.array([model_func({name: params_B[name][j] for name in param_names})
                    for j in range(n_samples)], dtype=float)

    VY = np.var(np.concatenate([Y_A, Y_B]))
    if VY < 1e-15:
        VY = 1e-15

    S1 = {}
    for i, name in enumerate(param_names):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        params_ABi = map_params(A_Bi)
        Y_ABi = np.array([model_func({n: params_ABi[n][j] for n in param_names})
                          for j in range(n_samples)], dtype=float)


        mean_A = np.mean(Y_A)
        mean_B = np.mean(Y_B)
        V_i = np.mean(Y_B * (Y_ABi - Y_A))
        S1[name] = float(V_i / VY)
        S1[name] = float(np.clip(S1[name], -1.0, 1.0))

    return S1, VY


def uncertainty_propagation_mc(model_func, param_distributions, n_samples=5000):
    outputs = []
    for _ in range(n_samples):
        sample_params = {}
        for name, (dist_type, params) in param_distributions.items():
            if dist_type == 'uniform':
                sample_params[name] = np.random.uniform(params[0], params[1])
            elif dist_type == 'normal':
                sample_params[name] = np.random.normal(params[0], params[1])
            else:
                sample_params[name] = params[0]
        try:
            out = model_func(sample_params)
            outputs.append(float(out))
        except Exception:
            outputs.append(np.nan)

    outputs = np.array(outputs, dtype=float)
    outputs = outputs[~np.isnan(outputs)]

    if len(outputs) == 0:
        return 0.0, 0.0, (0.0, 0.0)

    mean = float(np.mean(outputs))
    std = float(np.std(outputs))
    ci_95 = (float(np.percentile(outputs, 2.5)), float(np.percentile(outputs, 97.5)))
    return mean, std, ci_95
