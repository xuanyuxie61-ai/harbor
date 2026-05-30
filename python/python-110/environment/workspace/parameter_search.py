
import numpy as np
from typing import Callable, Dict, List, Tuple
from utils import validate_array_1d


def int_to_binary_vector(i4: int, n_bits: int) -> np.ndarray:
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if i4 < 0:
        raise ValueError("i4 must be non-negative")
    bvec = np.zeros(n_bits, dtype=int)
    temp = i4
    for i in range(n_bits - 1, -1, -1):
        bvec[i] = temp % 2
        temp = temp // 2
    return bvec


def brute_force_optimize(
    parameter_ranges: Dict[str, Tuple[float, float, int]],
    objective: Callable[[Dict[str, float]], float],
    constraint: Callable[[Dict[str, float]], bool] = None,
) -> Dict[str, any]:
    param_names = list(parameter_ranges.keys())
    grids = []
    for name in param_names:
        pmin, pmax, n_pts = parameter_ranges[name]
        if n_pts < 2:
            raise ValueError(f"Parameter {name} needs at least 2 grid points")
        grid = np.linspace(pmin, pmax, n_pts)
        grids.append(grid)


    total_combinations = 1
    for g in grids:
        total_combinations *= g.size

    best_value = float('inf')
    best_params = None
    all_results = []


    def recursive_search(depth: int, current_params: Dict[str, float]):
        nonlocal best_value, best_params
        if depth == len(param_names):
            if constraint is not None and not constraint(current_params):
                return
            val = objective(current_params)
            all_results.append((current_params.copy(), val))
            if val < best_value:
                best_value = val
                best_params = current_params.copy()
            return
        name = param_names[depth]
        for val in grids[depth]:
            current_params[name] = float(val)
            recursive_search(depth + 1, current_params)

    recursive_search(0, {})
    return {
        "best_params": best_params,
        "best_value": best_value,
        "all_results": all_results,
        "total_evaluated": len(all_results),
    }


def single_photon_figure_of_merit(
    g2_0: float,
    purcell_factor: float,
    extraction_efficiency: float,
    dephasing_rate: float,
    target_dephasing: float = 1e9,
) -> float:
    eps = 1e-15
    w_p = 1.0
    w_eta = 2.0
    w_d = 1.0
    if g2_0 < 0 or purcell_factor < 0 or extraction_efficiency < 0 or dephasing_rate < 0:
        raise ValueError("All physical quantities must be non-negative")
    fom = (
        -np.log10(g2_0 + eps)
        + w_p * np.log10(purcell_factor + 1.0)
        + w_eta * extraction_efficiency
        - w_d * (dephasing_rate / target_dephasing) ** 2
    )
    return fom


def binary_encoded_parameter_search(
    n_bits_per_param: int,
    param_bounds: List[Tuple[float, float]],
    objective: Callable[[np.ndarray], float],
) -> Tuple[np.ndarray, float]:
    n_params = len(param_bounds)
    total_bits = n_params * n_bits_per_param
    if total_bits > 20:
        raise ValueError("Search space too large (> 2^20 combinations)")
    n_combinations = 2 ** total_bits
    best_value = float('inf')
    best_params = None
    for idx in range(n_combinations):
        bvec = int_to_binary_vector(idx, total_bits)
        params = np.zeros(n_params, dtype=float)
        for p in range(n_params):
            bits = bvec[p * n_bits_per_param:(p + 1) * n_bits_per_param]

            int_val = 0
            for b in bits:
                int_val = int_val * 2 + int(b)
            frac = int_val / (2 ** n_bits_per_param - 1) if (2 ** n_bits_per_param - 1) > 0 else 0.0
            pmin, pmax = param_bounds[p]
            params[p] = pmin + frac * (pmax - pmin)
        val = objective(params)
        if val < best_value:
            best_value = val
            best_params = params.copy()
    return best_params, best_value


def sensitivity_analysis(
    base_params: Dict[str, float],
    param_deltas: Dict[str, float],
    objective: Callable[[Dict[str, float]], float],
) -> Dict[str, float]:
    base_val = objective(base_params)
    sensitivities = {}
    for name, delta in param_deltas.items():
        if abs(delta) < 1e-15:
            sensitivities[name] = 0.0
            continue
        p_plus = base_params.copy()
        p_minus = base_params.copy()
        p_plus[name] += delta
        p_minus[name] -= delta
        val_plus = objective(p_plus)
        val_minus = objective(p_minus)
        sens = (val_plus - val_minus) / (2.0 * delta)
        sensitivities[name] = sens
    return sensitivities
