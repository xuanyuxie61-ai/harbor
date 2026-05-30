
import numpy as np
from typing import Callable, List, Tuple, Optional


class OrbitOptimizationError(Exception):
    pass


def box_behnken_design(dim_num: int, ranges: np.ndarray) -> np.ndarray:
    if ranges.shape != (dim_num, 2):
        raise OrbitOptimizationError("ranges 形状必须为 (dim_num, 2)")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise OrbitOptimizationError("范围下限必须小于上限")


    x_num = dim_num * (2 ** (dim_num - 1)) + 1
    design = np.zeros((x_num, dim_num))

    j = 0
    design[j, :] = (ranges[:, 0] + ranges[:, 1]) / 2.0

    for i in range(dim_num):

        j += 1
        design[j, :] = ranges[:, 0].copy()
        design[j, i] = (ranges[i, 0] + ranges[i, 1]) / 2.0


        while True:
            last_low = -1
            for i2 in range(dim_num):
                if design[j, i2] == ranges[i2, 0]:
                    last_low = i2
            if last_low == -1:
                break
            j += 1
            design[j, :] = design[j - 1, :].copy()
            design[j, last_low] = ranges[last_low, 1]
            for i2 in range(last_low + 1, dim_num):
                if design[j, i2] == ranges[i2, 1]:
                    design[j, i2] = ranges[i2, 0]



    actual = j + 1
    if actual < x_num:
        design = design[:actual, :]
    return design


def backtrack_binary_rc(
    n: int,
    reject: bool,
    n2: int,
    choice: np.ndarray
) -> Tuple[int, np.ndarray]:
    choice = choice.copy()
    if n2 == -1:
        choice[:] = -1
        n2 = 0
        choice[n2] = 1
    elif n2 == n - 1 or reject:
        while n2 > 0:
            if choice[n2] == 1:
                choice[n2] = 0
                break
            choice[n2] = -1
            n2 -= 1
        if n2 == 0:
            if choice[0] == 1:
                choice[0] = 0
            else:
                choice[0] = -1
                n2 = -1
    else:
        n2 += 1
        choice[n2] = 1
    return n2, choice


def optimize_orbit_binary_backtrack(
    n_params: int,
    evaluate: Callable[[np.ndarray], float],
    max_evals: int = 10000
) -> Tuple[np.ndarray, float]:
    choice = np.full(n_params, -1, dtype=int)
    n2 = -1
    best_score = -np.inf
    best_choice = np.zeros(n_params, dtype=int)
    eval_count = 0

    while eval_count < max_evals:
        n2, choice = backtrack_binary_rc(n_params, False, n2, choice)
        if n2 == -1:
            break

        if n2 == n_params - 1:

            score = evaluate(choice)
            eval_count += 1
            if score > best_score:
                best_score = score
                best_choice = choice.copy()
            n2, choice = backtrack_binary_rc(n_params, False, n2, choice)
            if n2 == -1:
                break

    return best_choice, best_score


class OrbitSensitivityAnalysis:

    def __init__(
        self,
        param_names: List[str],
        param_ranges: np.ndarray,
        objective_func: Callable[[np.ndarray], float]
    ):
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.objective = objective_func
        self.dim = len(param_names)

    def run_analysis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = box_behnken_design(self.dim, self.param_ranges)
        n = design.shape[0]
        responses = np.zeros(n)
        for i in range(n):
            responses[i] = self.objective(design[i])


        main_effects = np.zeros(self.dim)
        for d in range(self.dim):
            high_mask = np.abs(design[:, d] - self.param_ranges[d, 1]) < 1e-12
            low_mask = np.abs(design[:, d] - self.param_ranges[d, 0]) < 1e-12
            if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
                main_effects[d] = np.mean(responses[high_mask]) - np.mean(responses[low_mask])

        return design, responses, main_effects

    def find_optimal_from_design(self) -> Tuple[np.ndarray, float]:
        design, responses, _ = self.run_analysis()
        idx = np.argmax(responses)
        return design[idx], responses[idx]


def compute_orbit_quality_score(
    params: np.ndarray,
    simulate_lifetime_func: Callable[[np.ndarray], float],
    simulate_dv_func: Callable[[np.ndarray], float],
    simulate_collision_prob_func: Callable[[np.ndarray], float],
    weights: Optional[np.ndarray] = None
) -> float:
    if weights is None:
        weights = np.array([1.0, -0.5, -10.0])

    lifetime = simulate_lifetime_func(params)
    dv = simulate_dv_func(params)
    p_coll = simulate_collision_prob_func(params)


    lifetime = max(lifetime, 0.0)
    dv = max(dv, 0.0)
    p_coll = np.clip(p_coll, 0.0, 1.0)

    score = (
        weights[0] * np.log1p(lifetime / 86400.0) +
        weights[1] * dv * 1e3 +
        weights[2] * p_coll
    )
    return score
