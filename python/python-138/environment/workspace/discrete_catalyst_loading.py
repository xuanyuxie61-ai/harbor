
import numpy as np
from typing import Tuple, List, Optional


class DiscreteCatalystLoadingOptimizer:

    def __init__(self, a_coeffs: np.ndarray, budget: float):
        self.a = np.asarray(a_coeffs, dtype=float).flatten()
        self.n = len(self.a)
        if np.any(self.a <= 0.0):
            raise ValueError("所有系数必须为正")
        if budget < 0.0:
            raise ValueError("预算必须非负")
        self.B = budget

    def solve_exact_nonnegative(
        self, max_solutions: int = 1000
    ) -> Tuple[np.ndarray, int]:

        scale = 1.0
        for i in range(self.n):

            pass

        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(self.B * 1000.0))

        g = int(np.gcd.reduce(a_int))
        if g > 1:
            a_int = a_int // g
            B_int = B_int // g

        solutions = []
        y = np.zeros(self.n, dtype=int)
        j = 0
        r = B_int

        while True:

            r = B_int
            for i in range(j):
                r -= a_int[i] * y[i]

            if j < self.n:
                j += 1
                y[j - 1] = r // a_int[j - 1]
            else:
                if r == 0:
                    solutions.append(y.copy())
                    if len(solutions) >= max_solutions:
                        break

                while j > 0:
                    if y[j - 1] > 0:
                        y[j - 1] -= 1
                        break
                    j -= 1
                if j == 0:
                    break

        if len(solutions) == 0:
            return np.empty((0, self.n)), 0
        sol_array = np.array(solutions)
        return sol_array, len(sol_array)

    def solve_bounded(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        max_solutions: int = 500,
    ) -> Tuple[np.ndarray, int]:
        lower = np.asarray(lower_bounds, dtype=int).flatten()
        upper = np.asarray(upper_bounds, dtype=int).flatten()
        if len(lower) != self.n or len(upper) != self.n:
            raise ValueError("边界维度与系数维度不匹配")


        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(self.B * 1000.0))
        B_prime = B_int - np.dot(a_int, lower)

        if B_prime < 0:
            return np.empty((0, self.n)), 0


        u_prime = upper - lower
        solutions = []
        y = np.zeros(self.n, dtype=int)
        j = 0

        while True:
            r = B_prime
            for i in range(j):
                r -= a_int[i] * y[i]

            if j < self.n:
                j += 1
                max_val = min(r // a_int[j - 1], u_prime[j - 1])
                y[j - 1] = max_val
            else:
                if r == 0:
                    sol = y + lower
                    solutions.append(sol.copy())
                    if len(solutions) >= max_solutions:
                        break
                while j > 0:
                    if y[j - 1] > 0:
                        y[j - 1] -= 1
                        if y[j - 1] > u_prime[j - 1]:
                            y[j - 1] = u_prime[j - 1]
                        break
                    j -= 1
                if j == 0:
                    break

        if len(solutions) == 0:
            return np.empty((0, self.n)), 0
        return np.array(solutions), len(solutions)

    def greedy_heuristic_solution(
        self, objective_weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        weights = np.asarray(objective_weights, dtype=float)
        if len(weights) != self.n:
            raise ValueError("权重维度不匹配")

        ratios = weights / self.a
        order = np.argsort(-ratios)
        x_greedy = np.zeros(self.n, dtype=int)
        remaining = self.B
        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(remaining * 1000.0))
        w_int = np.round(weights * 1000.0).astype(int)

        for idx in order:
            if a_int[idx] <= 0:
                continue
            max_units = B_int // a_int[idx]
            x_greedy[idx] = max_units
            B_int -= max_units * a_int[idx]

        objective = float(np.dot(weights, x_greedy))
        return x_greedy, objective

    def select_optimal_loading(
        self,
        solutions: np.ndarray,
        objective_func: Optional[callable] = None,
    ) -> Tuple[np.ndarray, float]:
        if solutions.shape[0] == 0:
            return np.zeros(self.n), float("inf")

        if objective_func is None:

            def objective_func(x):
                loads = self.a * x
                mu = np.mean(loads)
                if mu < 1.0e-12:
                    return 0.0
                return np.std(loads) / mu

        best_obj = float("inf")
        best_sol = solutions[0]
        for sol in solutions:
            obj = objective_func(sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = sol
        return best_sol, best_obj

    def compute_loading_efficiency(
        self, x: np.ndarray
    ) -> Tuple[float, float]:
        loads = self.a * x
        total = np.sum(loads)
        utilization = total / max(self.B, 1.0e-12)
        mu = np.mean(loads)
        if mu < 1.0e-12:
            uniformity = 0.0
        else:
            uniformity = max(0.0, 1.0 - np.std(loads) / mu)
        return utilization, uniformity
