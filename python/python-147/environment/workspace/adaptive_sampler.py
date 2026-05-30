
import numpy as np


def compute_residual_magnitude(network, X, physics_loss_fn):
    residuals = physics_loss_fn(network, X)
    return np.abs(residuals)


def adaptive_refinement_sample(network, X_current, residual_vals,
                               n_add, threshold_percentile=80):
    if len(residual_vals) == 0:
        raise ValueError("residual_vals is empty")
    if n_add <= 0:
        return np.zeros((0, 2))

    threshold = np.percentile(residual_vals, threshold_percentile)
    high_res_mask = residual_vals >= threshold
    high_res_points = X_current[high_res_mask]

    if len(high_res_points) == 0:

        tmax = X_current[:, 0].max()
        xL = X_current[:, 1].max()
        rng = np.random.default_rng(42)
        X_new = np.column_stack([
            rng.uniform(0.0, tmax, size=n_add),
            rng.uniform(0.0, xL, size=n_add)
        ])
        return X_new


    rng = np.random.default_rng(42)
    indices = rng.choice(len(high_res_points), size=n_add, replace=True)
    centers = high_res_points[indices]


    dt_scale = (X_current[:, 0].max() - X_current[:, 0].min()) * 0.05
    dx_scale = (X_current[:, 1].max() - X_current[:, 1].min()) * 0.05

    noise = np.column_stack([
        rng.normal(0.0, dt_scale, size=n_add),
        rng.normal(0.0, dx_scale, size=n_add)
    ])
    X_new = centers + noise


    tmin, tmax = X_current[:, 0].min(), X_current[:, 0].max()
    xmin, xmax = X_current[:, 1].min(), X_current[:, 1].max()
    X_new[:, 0] = np.clip(X_new[:, 0], tmin, tmax)
    X_new[:, 1] = np.clip(X_new[:, 1], xmin, xmax)

    return X_new


def multi_level_grid_refinement(tmax, L_domain, base_nt, base_nx, levels):
    grids = []
    for lvl in range(levels):
        nt = base_nt * (2 ** lvl)
        nx = base_nx * (2 ** lvl)
        t = np.linspace(0.0, tmax, nt)
        x = np.linspace(0.0, L_domain, nx, endpoint=False)
        T, X = np.meshgrid(t, x, indexing='ij')
        grid = np.column_stack([T.ravel(), X.ravel()])
        grids.append(grid)
    return grids


def stochastic_rejection_sample(residual_fn, domain_bounds, n_samples,
                                max_trials=10000):
    rng = np.random.default_rng(42)
    (tmin, tmax), (xmin, xmax) = domain_bounds
    samples = []
    trials = 0


    probe_t = rng.uniform(tmin, tmax, size=500)
    probe_x = rng.uniform(xmin, xmax, size=500)
    probe_vals = residual_fn(probe_t, probe_x)
    r_max = np.max(np.abs(probe_vals)) * 1.5 + 1e-6

    while len(samples) < n_samples and trials < max_trials:
        t_cand = rng.uniform(tmin, tmax)
        x_cand = rng.uniform(xmin, xmax)
        r_cand = np.abs(residual_fn(np.array([t_cand]), np.array([x_cand])))
        accept_prob = r_cand / r_max
        if rng.random() < accept_prob:
            samples.append([t_cand, x_cand])
        trials += 1

    if len(samples) < n_samples:

        n_fill = n_samples - len(samples)
        fill = np.column_stack([
            rng.uniform(tmin, tmax, size=n_fill),
            rng.uniform(xmin, xmax, size=n_fill)
        ])
        if samples:
            samples = np.vstack([samples, fill])
        else:
            samples = fill
    else:
        samples = np.array(samples)

    return samples
