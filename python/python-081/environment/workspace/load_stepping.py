
import numpy as np
from typing import List, Tuple, Optional


class AdaptiveLoadStepping:
    def __init__(self, lambda_max: float = 1.0,
                 initial_step: float = 0.1,
                 min_step: float = 0.001,
                 max_step: float = 0.5,
                 desired_iterations: int = 5,
                 max_total_steps: int = 200):
        self.lambda_max = lambda_max
        self.step = initial_step
        self.min_step = min_step
        self.max_step = max_step
        self.desired_iterations = desired_iterations
        self.max_total_steps = max_total_steps
        self.current_lambda = 0.0
        self.step_history = []
        self.iter_history = []
        self.n_backtracks = 0

    def adjust_step(self, n_iterations: int, converged: bool) -> float:
        if not converged:
            self.step = max(self.min_step, self.step / 2.0)
            self.n_backtracks += 1
        else:
            ratio = np.sqrt(self.desired_iterations / max(n_iterations, 1))
            self.step = min(self.max_step, max(self.min_step, ratio * self.step))
        self.step_history.append(self.step)
        self.iter_history.append(n_iterations)
        return self.step

    def next_lambda(self) -> Tuple[float, bool]:
        if self.current_lambda >= self.lambda_max - 1e-12:
            return self.current_lambda, True

        remaining = self.lambda_max - self.current_lambda
        actual_step = min(self.step, remaining)
        self.current_lambda += actual_step

        if self.current_lambda > self.lambda_max:
            self.current_lambda = self.lambda_max
        return self.current_lambda, False

    def reset_to_previous(self):
        if len(self.step_history) > 0:
            last_step = self.step_history.pop()
            self.iter_history.pop()
            self.current_lambda -= last_step
            if self.current_lambda < 0:
                self.current_lambda = 0.0


def integer_load_partition(total_load: int, step_sizes: List[int],
                            target_steps: int) -> Optional[List[int]]:
    step_sizes = sorted(step_sizes, reverse=True)
    partition = []
    remaining = total_load


    def backtrack(idx: int, rem: int, current: List[int]) -> Optional[List[int]]:
        if rem == 0:
            return current
        if idx >= len(step_sizes):
            return None
        s = step_sizes[idx]
        max_count = rem // s
        for count in range(max_count, -1, -1):
            new_rem = rem - count * s
            new_current = current + [s] * count
            result = backtrack(idx + 1, new_rem, new_current)
            if result is not None:
                return result
        return None

    result = backtrack(0, total_load, [])
    if result is None:
        return None

    while len(result) > target_steps and len(result) > 1:

        result = sorted(result)
        combined = result[0] + result[1]
        result = [combined] + result[2:]
    return result


def arc_length_control_step(u_n: np.ndarray, du_predictor: np.ndarray,
                             dlambda_predictor: float,
                             ds: float, psi: float = 0.0) -> Tuple[float, np.ndarray]:
    norm_du = np.linalg.norm(du_predictor)
    constraint = norm_du ** 2 + psi ** 2 * dlambda_predictor ** 2
    if constraint < 1e-14:
        return dlambda_predictor, du_predictor
    scale = ds / np.sqrt(constraint)
    dlambda = dlambda_predictor * scale
    du = du_predictor * scale
    return dlambda, du
