"""
adaptive_sampler.py
===================
Adaptive collocation point sampling for PINN training.

The key idea is that the PDE residual r(t,x) is not uniformly distributed
across the domain.  Regions with high residual require more collocation points.
We implement:

  1. Residual-based refinement: sample more points where |r(t,x)| > threshold
  2. Kd-tree-inspired spatial subdivision for multi-resolution coverage
  3. Stochastic rejection sampling proportional to residual magnitude

The adaptive sampler progressively enriches the training set based on the
current network's residual field, leading to faster convergence and better
accuracy in high-gradient regions (shocks, fronts, chaotic bursts).
"""

import numpy as np


def compute_residual_magnitude(network, X, physics_loss_fn):
    """
    Compute |r(t,x)| at collocation points X for adaptive sampling.

    Parameters
    ----------
    network : PINNNetwork
    X : ndarray, shape (n, 2)
        Collocation points.
    physics_loss_fn : callable
        Function that computes the PDE residual vector.

    Returns
    -------
    residuals : ndarray, shape (n,)
        Absolute residual at each point.
    """
    residuals = physics_loss_fn(network, X)
    return np.abs(residuals)


def adaptive_refinement_sample(network, X_current, residual_vals,
                               n_add, threshold_percentile=80):
    """
    Add new collocation points in high-residual regions.

    Strategy:
      - Identify points where residual > percentile threshold.
      - Cluster these points by spatial proximity.
      - Add random perturbations around cluster centers to create new points.

    Parameters
    ----------
    network : PINNNetwork
    X_current : ndarray, shape (n, 2)
        Current collocation points.
    residual_vals : ndarray, shape (n,)
        Residual magnitudes at current points.
    n_add : int
        Number of new points to add.
    threshold_percentile : float
        Percentile threshold for high-residual identification.

    Returns
    -------
    X_new : ndarray, shape (n_add, 2)
        New collocation points.
    """
    if len(residual_vals) == 0:
        raise ValueError("residual_vals is empty")
    if n_add <= 0:
        return np.zeros((0, 2))

    threshold = np.percentile(residual_vals, threshold_percentile)
    high_res_mask = residual_vals >= threshold
    high_res_points = X_current[high_res_mask]

    if len(high_res_points) == 0:
        # Fallback: uniform random sampling in domain
        tmax = X_current[:, 0].max()
        xL = X_current[:, 1].max()
        rng = np.random.default_rng(42)
        X_new = np.column_stack([
            rng.uniform(0.0, tmax, size=n_add),
            rng.uniform(0.0, xL, size=n_add)
        ])
        return X_new

    # Simple adaptive sampling: randomly sample near high-residual points
    rng = np.random.default_rng(42)
    indices = rng.choice(len(high_res_points), size=n_add, replace=True)
    centers = high_res_points[indices]

    # Perturbation scales based on domain size
    dt_scale = (X_current[:, 0].max() - X_current[:, 0].min()) * 0.05
    dx_scale = (X_current[:, 1].max() - X_current[:, 1].min()) * 0.05

    noise = np.column_stack([
        rng.normal(0.0, dt_scale, size=n_add),
        rng.normal(0.0, dx_scale, size=n_add)
    ])
    X_new = centers + noise

    # Clip to domain bounds
    tmin, tmax = X_current[:, 0].min(), X_current[:, 0].max()
    xmin, xmax = X_current[:, 1].min(), X_current[:, 1].max()
    X_new[:, 0] = np.clip(X_new[:, 0], tmin, tmax)
    X_new[:, 1] = np.clip(X_new[:, 1], xmin, xmax)

    return X_new


def multi_level_grid_refinement(tmax, L_domain, base_nt, base_nx, levels):
    """
    Generate a multi-level nested grid where finer grids are embedded in
    coarser ones.  This is analogous to multigrid or wavelet decomposition.

    Returns a list of grids from coarsest to finest.
    """
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
    """
    Sample points with probability proportional to residual magnitude
    using rejection sampling.

    Parameters
    ----------
    residual_fn : callable
        Function residual_fn(t, x) -> scalar or array.
    domain_bounds : tuple
        ((tmin, tmax), (xmin, xmax))
    n_samples : int
    max_trials : int

    Returns
    -------
    samples : ndarray, shape (n_samples, 2)
    """
    rng = np.random.default_rng(42)
    (tmin, tmax), (xmin, xmax) = domain_bounds
    samples = []
    trials = 0

    # Estimate maximum residual via random sampling
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
        # Fill remaining with uniform random
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
