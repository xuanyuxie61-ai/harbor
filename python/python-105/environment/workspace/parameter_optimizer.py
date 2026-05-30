
import numpy as np
from typing import List, Tuple, Optional





def gray_code_subsets(n: int) -> Tuple[List[np.ndarray], List[int]]:
    if n < 0:
        raise ValueError("n 必须非负。")
    subsets = []
    iadds = []
    a = np.zeros(n, dtype=int)
    ncard = 0
    iadd = -1
    subsets.append(a.copy())
    iadds.append(iadd)
    more = True

    while more:
        iadd = 1
        if ncard % 2 != 0:
            while iadd <= n and a[iadd - 1] == 0:
                iadd += 1
            iadd += 1
        if iadd <= n:
            a[iadd - 1] = 1 - a[iadd - 1]
            ncard += 2 * a[iadd - 1] - 1
            subsets.append(a.copy())
            iadds.append(iadd - 1)
            if ncard == a[n - 1]:
                more = False
        else:
            more = False

    return subsets, iadds





def diophantine_nd_nonnegative(a: np.ndarray, b: int) -> np.ndarray:
    a = np.asarray(a, dtype=int)
    if np.any(a <= 0):
        raise ValueError("系数 a 必须全为正整数。")
    if b < 0:
        return np.empty((0, len(a)), dtype=int)
    d = len(a)


    def _recurse(idx: int, remaining: int) -> List[Tuple[int, ...]]:
        if idx == d - 1:
            if remaining % a[idx] == 0:
                return [(remaining // a[idx],)]
            else:
                return []
        sols = []
        max_x = remaining // a[idx]
        for x in range(max_x + 1):
            sub = _recurse(idx + 1, remaining - x * a[idx])
            for s in sub:
                sols.append((x,) + s)
        return sols

    raw = _recurse(0, b)
    if not raw:
        return np.empty((0, d), dtype=int)
    return np.array(raw, dtype=int)


def count_parameter_combinations(package_sizes: np.ndarray,
                                  target_value: int) -> int:
    sols = diophantine_nd_nonnegative(package_sizes, target_value)
    return sols.shape[0]





def subset_sum_backtrack_all(s: int, v: np.ndarray) -> List[np.ndarray]:
    v = np.asarray(v, dtype=int)
    n = len(v)
    solutions = []
    u = np.zeros(n, dtype=int)
    t = 0
    more = False


    while True:
        if not more:
            t = 0
            u[:] = 0
        else:
            more = False
            u[t] = 0
            told = t
            t = -1
            for i in range(told - 1, -1, -1):
                if u[i] == 1:
                    t = i
                    break
            if t < 0:
                break
            u[t] = 0
            t += 1
            u[t] = 1

        while True:
            su = np.dot(u, v)
            if su < s and t < n - 1:
                t += 1
                u[t] = 1
            elif su == s:
                solutions.append(u.copy())
                more = True
                break
            else:
                u[t] = 0
                told = t
                t = -1
                for i in range(told - 1, -1, -1):
                    if u[i] == 1:
                        t = i
                        break
                if t < 0:
                    break
                u[t] = 0
                t += 1
                u[t] = 1

        if not more:
            break

    return solutions





def optimize_polling_period_and_length(
    allowed_periods_nm: np.ndarray,
    allowed_lengths_mm: np.ndarray,
    allowed_temperatures_c: np.ndarray,
    objective_func: callable,
    max_evals: Optional[int] = None
) -> Tuple[float, float, float, float]:
    n_p = len(allowed_periods_nm)
    n_l = len(allowed_lengths_mm)
    n_t = len(allowed_temperatures_c)


    n_total = n_p * n_l * n_t
    if max_evals is None or max_evals > n_total:
        max_evals = n_total

    best_obj = -np.inf
    best_p = allowed_periods_nm[0]
    best_l = allowed_lengths_mm[0]
    best_t = allowed_temperatures_c[0]



    eval_count = 0
    for p_idx in range(n_p):
        for l_idx in range(n_l):
            for t_idx in range(n_t):
                if eval_count >= max_evals:
                    return best_p, best_l, best_t, best_obj
                period = allowed_periods_nm[p_idx]
                length = allowed_lengths_mm[l_idx]
                temp = allowed_temperatures_c[t_idx]
                obj = objective_func(period, length, temp)
                eval_count += 1
                if obj > best_obj:
                    best_obj = obj
                    best_p = period
                    best_l = length
                    best_t = temp

    return best_p, best_l, best_t, best_obj
