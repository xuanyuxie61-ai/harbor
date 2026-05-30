
import numpy as np
from multiprocessing import Pool, cpu_count


def _evaluate_discriminant_at_point(args):
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
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    names = list(param_grids.keys())
    grids = [param_grids[n] for n in names]


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
    ep_points = [r for r in results if r['abs_delta'] < threshold]
    return ep_points


def coarse_to_fine_ep_search(H_builder_func, param_ranges, levels=3, threshold=1e-3):
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


        for name in current_ranges:
            vals = [p['params'][name] for p in ep_points]
            margin = (max(vals) - min(vals)) * 0.2 + 1e-6
            current_ranges[name][0] = max(min(vals) - margin, param_ranges[name][0])
            current_ranges[name][1] = min(max(vals) + margin, param_ranges[name][1])

        n_points = min(n_points * 2, 64)

    return ep_points
