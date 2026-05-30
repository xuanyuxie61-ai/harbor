# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, List, Tuple, Optional


class GlobalParameterCalibration:

    def __init__(self, f: Callable[[float], float],
                 a: float, b: float,
                 M_bound: Optional[float] = None,
                 tol: float = 1e-6, max_iter: int = 100):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        if M_bound is None:

            self.M = self._estimate_second_derivative_bound()
        else:
            self.M = M_bound

    def _estimate_second_derivative_bound(self, num_samples: int = 20) -> float:
        x_samples = np.linspace(self.a, self.b, num_samples)
        h = (self.b - self.a) / (num_samples * 10.0)
        max_d2 = 0.0
        for x in x_samples[1:-1]:
            if x - h < self.a or x + h > self.b:
                continue
            f_pp = (self.f(x + h) - 2.0 * self.f(x) + self.f(x - h)) / (h ** 2)
            max_d2 = max(max_d2, abs(f_pp))
        return max(max_d2, 1.0)

    def _lower_bound(self, c: float, fc: float, dfdc: float,
                     x: float) -> float:
        dx = x - c
        return fc + dfdc * dx - 0.5 * self.M * dx ** 2

    def minimize(self) -> Tuple[float, float]:

        intervals = [(self.a, self.b)]
        best_x = (self.a + self.b) / 2.0
        best_f = self.f(best_x)

        for _ in range(self.max_iter):
            new_intervals = []
            for a_int, b_int in intervals:
                if b_int - a_int < self.tol:
                    new_intervals.append((a_int, b_int))
                    continue

                c = (a_int + b_int) / 2.0
                fc = self.f(c)
                h = min(self.tol * 10.0, (b_int - a_int) * 0.01)
                h = max(h, 1e-12)
                dfdc = (self.f(c + h) - self.f(c - h)) / (2.0 * h)

                if fc < best_f:
                    best_f = fc
                    best_x = c


                lb_a = self._lower_bound(c, fc, dfdc, a_int)
                lb_b = self._lower_bound(c, fc, dfdc, b_int)
                lb_min = min(lb_a, lb_b)


                if lb_min > best_f:
                    continue


                mid = (a_int + b_int) / 2.0
                new_intervals.append((a_int, mid))
                new_intervals.append((mid, b_int))

            intervals = new_intervals
            if not intervals:
                break

            intervals = [(a_i, b_i) for a_i, b_i in intervals
                         if b_i - a_i > self.tol]

        return best_x, best_f

    @staticmethod
    def multivariable_search(f: Callable, bounds: List[Tuple[float, float]],
                             num_grid: int = 10) -> Tuple[np.ndarray, float]:
        ndim = len(bounds)

        best_x = None
        best_f = np.inf
        samples = np.random.rand(num_grid * ndim, ndim)
        for i in range(ndim):
            samples[:, i] = bounds[i][0] + samples[:, i] * (bounds[i][1] - bounds[i][0])

        for s in samples:
            try:
                val = f(s)
            except Exception:
                val = np.inf
            if val < best_f:
                best_f = val
                best_x = s.copy()


        for _ in range(3):
            for i in range(ndim):
                def f_1d(xi):
                    x_temp = best_x.copy()
                    x_temp[i] = xi
                    return f(x_temp)
                calib = GlobalParameterCalibration(f_1d, bounds[i][0], bounds[i][1],
                                                    tol=1e-4, max_iter=50)
                xi_opt, _ = calib.minimize()
                best_x[i] = xi_opt
                best_f = f(best_x)

        return best_x, best_f


class FailureSequenceOptimizer:

    def __init__(self, num_plies: int,
                 ply_strengths: np.ndarray,
                 ply_thicknesses: np.ndarray,
                 E_ply: np.ndarray,
                 area: float = 1.0):
        self.n = num_plies
        self.sigma_ult = np.asarray(ply_strengths)
        self.h = np.asarray(ply_thicknesses)
        self.E = np.asarray(E_ply)
        self.A = area

        if len(self.sigma_ult) != self.n:
            raise ValueError("Length of ply_strengths must match num_plies.")

    def _failure_work(self, ply_index: int, damaged_set: int) -> float:

        remaining = [j for j in range(self.n) if not (damaged_set & (1 << j))]
        if ply_index not in remaining:
            return np.inf

        h_total = sum(self.h[j] for j in remaining)
        if h_total < 1e-30:
            return 0.0


        E_eff = sum(self.E[j] * self.h[j] for j in remaining) / h_total
        eps_ult = self.sigma_ult[ply_index] / (E_eff + 1e-30)
        W = 0.5 * self.A * self.h[ply_index] * E_eff * (eps_ult ** 2)
        return W

    def optimize_min_work(self) -> Tuple[float, List[int]]:
        total_states = 1 << self.n
        INF = 1e30
        dp = np.full(total_states, INF)
        parent = np.full(total_states, -1, dtype=int)
        dp[0] = 0.0

        for mask in range(total_states):
            if dp[mask] >= INF:
                continue
            for k in range(self.n):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    work = self._failure_work(k, mask)
                    if dp[mask] + work < dp[new_mask]:
                        dp[new_mask] = dp[mask] + work
                        parent[new_mask] = k


        sequence = []
        mask = total_states - 1
        while mask > 0:
            k = parent[mask]
            if k < 0:
                break
            sequence.append(k)
            mask ^= (1 << k)
        sequence.reverse()

        return dp[total_states - 1], sequence

    def optimize_max_ductility(self) -> Tuple[float, List[int]]:
        total_states = 1 << self.n
        INF = 1e30

        dp_var = np.full(total_states, INF)
        dp_work = np.zeros(total_states)
        parent = np.full(total_states, -1, dtype=int)
        dp_var[0] = 0.0

        for mask in range(total_states):
            if dp_var[mask] >= INF:
                continue
            for k in range(self.n):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    work = self._failure_work(k, mask)
                    new_total = dp_work[mask] + work

                    min_increment = work
                    if dp_work[mask] > 0:
                        num_steps = bin(mask).count('1') + 1
                        avg = new_total / num_steps

                        penalty = abs(work - avg)
                    else:
                        penalty = 0.0

                    score = dp_var[mask] + penalty
                    if score < dp_var[new_mask]:
                        dp_var[new_mask] = score
                        dp_work[new_mask] = new_total
                        parent[new_mask] = k

        sequence = []
        mask = total_states - 1
        while mask > 0:
            k = parent[mask]
            if k < 0:
                break
            sequence.append(k)
            mask ^= (1 << k)
        sequence.reverse()

        return dp_work[total_states - 1], sequence


if __name__ == "__main__":

    def test_func(x):
        return (x - 0.3) ** 2 + 0.1 * np.sin(20 * np.pi * x)

    calib = GlobalParameterCalibration(test_func, 0.0, 1.0, tol=1e-5)
    x_opt, f_opt = calib.minimize()
    print("Global optimization result:", x_opt, f_opt)


    def f2d(x):
        return (x[0] - 0.5) ** 2 + (x[1] + 0.2) ** 2 + 0.05 * np.sin(10 * x[0]) * np.cos(10 * x[1])

    x_opt2, f_opt2 = GlobalParameterCalibration.multivariable_search(
        f2d, [(-1.0, 1.0), (-1.0, 1.0)], num_grid=20)
    print("Multivariable optimization result:", x_opt2, f_opt2)


    n_plies = 4
    strengths = np.array([1500e6, 1200e6, 1200e6, 1500e6])
    thicknesses = np.array([0.125e-3, 0.125e-3, 0.125e-3, 0.125e-3])
    E_plies = np.array([181e9, 10.3e9, 10.3e9, 181e9])
    optimizer = FailureSequenceOptimizer(n_plies, strengths, thicknesses, E_plies, area=1e-4)
    min_work, seq = optimizer.optimize_min_work()
    print("Minimum work failure sequence:", seq, "work=", min_work)
