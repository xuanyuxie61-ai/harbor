"""
parallel_sweep.py
-----------------
Parallel parameter-space sweeping for exceptional-point manifolds,
adapted from parallel for-loop patterns.

Adapted from seed project 514_hello_parfor.

Scientific Background
=====================
Exceptional points in non-Hermitian Hamiltonians often form
one-dimensional manifolds (lines) or higher-dimensional surfaces in
parameter space. To map these manifolds, one must sweep over a
multidimensional grid of control parameters (e.g., hopping amplitudes,
gain/loss rates, magnetic flux) and evaluate the discriminant
Δ(λ) = 0 condition at each point.

Because the evaluation at each grid point is independent, this workload
is embarrassingly parallel. We use Python's multiprocessing to distribute
the computation across CPU cores.

The computational complexity of a full d-dimensional sweep with N points
per dimension is O(N^d), making parallelism essential for d ≥ 2.
"""

import numpy as np
from multiprocessing import Pool, cpu_count


def _evaluate_discriminant_at_point(args):
    """
    Worker function: compute discriminant for a single parameter point.
    args = (param_dict, H_builder_func)
    """
    param_dict, H_builder_func = args
    try:
        H = H_builder_func(**param_dict)
        from hamiltonian_builder import discriminant_2x2
        delta = discriminant_2x2(H)
        return {
            'params': param_dict,
            'delta': delta,
            'abs_delta': abs(delta),
        }
    except Exception as e:
        return {
            'params': param_dict,
            'delta': np.nan,
            'abs_delta': np.nan,
            'error': str(e),
        }


def parallel_parameter_sweep(H_builder_func, param_grids, n_workers=None):
    """
    Sweep over a Cartesian product of parameter grids in parallel.

    Parameters
    ----------
    H_builder_func : callable
        Function that takes keyword arguments and returns a Hamiltonian matrix.
    param_grids : dict
        Dictionary mapping parameter names to 1D arrays of values.
        Example: {'t': np.linspace(0,2,21), 'gamma': np.linspace(0,1,11)}
    n_workers : int or None
        Number of parallel workers. If None, uses all CPUs.

    Returns
    -------
    results : list of dict
        Each dict contains the parameter point and the discriminant.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    names = list(param_grids.keys())
    grids = [param_grids[n] for n in names]

    # Build Cartesian product
    mesh = np.meshgrid(*grids, indexing='ij')
    flat_values = [m.flatten() for m in mesh]
    n_total = flat_values[0].size

    task_list = []
    for idx in range(n_total):
        pdict = {names[i]: flat_values[i][idx] for i in range(len(names))}
        task_list.append((pdict, H_builder_func))

    with Pool(processes=n_workers) as pool:
        results = pool.map(_evaluate_discriminant_at_point, task_list)

    return results


def find_ep_contours_from_sweep(results, threshold=1e-3):
    """
    From a parameter sweep result list, identify points where the
    discriminant magnitude is below a threshold (approximate EPs).

    Parameters
    ----------
    results : list of dict
        Output from parallel_parameter_sweep.
    threshold : float

    Returns
    -------
    ep_points : list of dict
        Points satisfying |Δ| < threshold.
    """
    ep_points = [r for r in results if r['abs_delta'] < threshold]
    return ep_points


def coarse_to_fine_ep_search(H_builder_func, param_ranges, levels=3, threshold=1e-3):
    """
    Hierarchical coarse-to-fine search for exceptional points.

    At each level, perform a coarse parallel sweep, identify regions
    near EPs, and zoom in for the next level.

    Parameters
    ----------
    H_builder_func : callable
    param_ranges : dict
        {name: (min, max)}
    levels : int
        Number of refinement levels.
    threshold : float

    Returns
    -------
    fine_ep_points : list of dict
    """
    current_ranges = {k: list(v) for k, v in param_ranges.items()}
    n_points = 16

    for level in range(levels):
        param_grids = {
            name: np.linspace(lo, hi, n_points)
            for name, (lo, hi) in current_ranges.items()
        }
        results = parallel_parameter_sweep(H_builder_func, param_grids)
        ep_points = find_ep_contours_from_sweep(results, threshold=threshold * (10 ** level))

        if not ep_points:
            break

        # Update ranges to bounding box of EP candidates
        for name in current_ranges:
            vals = [p['params'][name] for p in ep_points]
            margin = (max(vals) - min(vals)) * 0.2 + 1e-6
            current_ranges[name][0] = max(min(vals) - margin, param_ranges[name][0])
            current_ranges[name][1] = min(max(vals) + margin, param_ranges[name][1])

        n_points = min(n_points * 2, 64)

    return ep_points
