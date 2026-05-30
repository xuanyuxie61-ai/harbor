
import numpy as np


class LoadOptimizerError(Exception):
    pass


def subset_sum_backtrack(values, target):
    values = sorted([float(v) for v in values if v > 0])
    n = len(values)
    solutions = []

    def backtrack(start_idx, current_sum, current_indices):

        if abs(current_sum - target) < 1e-9:
            solutions.append(current_indices.copy())
            return

        if current_sum > target:
            return
        remaining_max = sum(values[start_idx:])
        if current_sum + remaining_max < target - 1e-9:
            return

        for i in range(start_idx, n):
            current_indices.append(i)
            backtrack(i + 1, current_sum + values[i], current_indices)
            current_indices.pop()

    backtrack(0, 0.0, [])
    return solutions


def diophantine_nonnegative_solutions(coeffs, target):
    coeffs = [int(c) for c in coeffs if c > 0]
    target = int(target)
    if target < 0:
        return []
    n = len(coeffs)
    solutions = []


    def search(idx, remaining, current):
        if idx == n - 1:
            if remaining % coeffs[idx] == 0:
                x = remaining // coeffs[idx]
                solutions.append(tuple(current + [x]))
            return
        max_x = remaining // coeffs[idx]
        for x in range(max_x + 1):
            current.append(x)
            search(idx + 1, remaining - x * coeffs[idx], current)
            current.pop()

    search(0, target, [])
    return solutions


def adaptive_load_stepping(initial_load, target_load, min_step, max_step,
                           convergence_history=None):
    remaining = target_load - initial_load
    if remaining <= 1e-12:
        return []

    step_sizes = []
    current_step = max_step
    current_load = initial_load

    iter_idx = 0
    while current_load < target_load - 1e-9:
        if convergence_history is not None and iter_idx < len(convergence_history):
            n_iter = convergence_history[iter_idx]
            if n_iter <= 3:
                current_step = min(current_step * 1.5, max_step)
            elif n_iter >= 8:
                current_step = max(current_step * 0.5, min_step)

        actual_step = min(current_step, target_load - current_load)
        if actual_step < min_step:
            actual_step = target_load - current_load

        step_sizes.append(float(actual_step))
        current_load += actual_step
        iter_idx += 1


        if len(step_sizes) > 10000:
            break

    return step_sizes


def optimize_load_increments(candidate_steps, target_total, strategy='subset_sum'):
    if strategy == 'subset_sum':
        sols = subset_sum_backtrack(candidate_steps, target_total)
        if not sols:

            seq = []
            remaining = target_total
            for v in sorted(candidate_steps, reverse=True):
                while remaining >= v - 1e-9:
                    seq.append(v)
                    remaining -= v
            if remaining > 1e-9:
                seq.append(remaining)
            return sorted(seq), 0


        best = min(sols, key=len)
        best_sequence = [candidate_steps[i] for i in best]
        return best_sequence, len(sols)

    elif strategy == 'diophantine':

        scale = 1000
        int_coeffs = [int(round(c * scale)) for c in candidate_steps]
        int_target = int(round(target_total * scale))
        sols = diophantine_nonnegative_solutions(int_coeffs, int_target)
        if not sols:
            return optimize_load_increments(candidate_steps, target_total, 'subset_sum')

        best = min(sols, key=sum)
        best_sequence = []
        for i, count in enumerate(best):
            best_sequence.extend([candidate_steps[i]] * count)
        return best_sequence, len(sols)

    else:
        raise LoadOptimizerError(f"Unknown strategy: {strategy}")
