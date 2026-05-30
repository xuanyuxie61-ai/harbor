
import numpy as np
from typing import List, Tuple
from numerical_solver import VandermondeSolver


class TSPBruteForce:

    def __init__(self):
        pass

    def path_cost(self, distance: np.ndarray, perm: Tuple[int, ...]) -> float:
        n = len(perm)
        cost = 0.0
        for i in range(n):
            j = (i + 1) % n
            cost += distance[perm[i], perm[j]]
        return cost

    def _next_permutation(self, perm: List[int]) -> bool:
        n = len(perm)
        i = n - 2
        while i >= 0 and perm[i] >= perm[i + 1]:
            i -= 1
        if i < 0:
            return False
        j = n - 1
        while perm[j] <= perm[i]:
            j -= 1
        perm[i], perm[j] = perm[j], perm[i]
        perm[i + 1:] = reversed(perm[i + 1:])
        return True

    def solve(self, points: np.ndarray) -> Tuple[Tuple[int, ...], float, float, float]:
        n = points.shape[0]
        if n < 2:
            return tuple(range(n)), 0.0, 0.0, 0.0

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distance = np.linalg.norm(diff, axis=2)

        perm = list(range(n))
        min_cost = float('inf')
        max_cost = -float('inf')
        sum_cost = 0.0
        count = 0
        best_perm = tuple(perm)

        while True:
            cost = self.path_cost(distance, tuple(perm))
            sum_cost += cost
            count += 1
            if cost < min_cost:
                min_cost = cost
                best_perm = tuple(perm)
            if cost > max_cost:
                max_cost = cost
            if not self._next_permutation(perm):
                break

        avg_cost = sum_cost / count if count > 0 else 0.0
        return best_perm, min_cost, avg_cost, max_cost


class PolynomialSwingTrajectory:

    def __init__(self):
        self.vandermonde = VandermondeSolver()

    def fit_quintic(self, T: float,
                    p_start: float, p_end: float,
                    v_start: float, v_end: float,
                    a_start: float, a_end: float) -> np.ndarray:
        if T <= 1e-9:
            T = 1e-6

        A = np.array([
            [1.0, 0.0, 0.0,       0.0,         0.0,         0.0],
            [1.0, T,   T**2,     T**3,        T**4,        T**5],
            [0.0, 1.0, 0.0,       0.0,         0.0,         0.0],
            [0.0, 1.0, 2.0*T,   3.0*T**2,    4.0*T**3,    5.0*T**4],
            [0.0, 0.0, 2.0,       0.0,         0.0,         0.0],
            [0.0, 0.0, 2.0,     6.0*T,      12.0*T**2,   20.0*T**3],
        ], dtype=float)
        b = np.array([p_start, p_end, v_start, v_end, a_start, a_end], dtype=float)

        c = np.linalg.solve(A, b)
        return c

    def evaluate(self, coeffs: np.ndarray, t: float) -> Tuple[float, float, float]:
        c = coeffs

        p = c[5]
        for i in range(4, -1, -1):
            p = p * t + c[i]

        dc = np.array([c[1], 2.0*c[2], 3.0*c[3], 4.0*c[4], 5.0*c[5]], dtype=float)
        v = dc[4]
        for i in range(3, -1, -1):
            v = v * t + dc[i]

        d2c = np.array([2.0*c[2], 6.0*c[3], 12.0*c[4], 20.0*c[5]], dtype=float)
        a = d2c[3]
        for i in range(2, -1, -1):
            a = a * t + d2c[i]
        return p, v, a


class FootfallPlanner:

    def __init__(self, swing_height: float = 0.05, swing_period: float = 0.3):
        self.swing_h = swing_height
        self.swing_T = swing_period
        self.tsp = TSPBruteForce()
        self.traj = PolynomialSwingTrajectory()

    def plan_footholds(self, candidate_points: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
        best_perm, min_cost, avg_cost, max_cost = self.tsp.solve(candidate_points)
        sorted_points = candidate_points[list(best_perm), :]
        return best_perm, sorted_points

    def generate_swing_trajectory(self, p_start: np.ndarray, p_end: np.ndarray,
                                  n_samples: int = 30) -> np.ndarray:
        traj_points = np.zeros((n_samples, 3))
        for dim in range(3):
            if dim == 2:

                c = self.traj.fit_quintic(
                    self.swing_T,
                    p_start[2], p_end[2],
                    0.0, 0.0,
                    0.0, 0.0
                )




                t_vals = np.linspace(0, self.swing_T, n_samples)
                for i, t in enumerate(t_vals):
                    z, _, _ = self.traj.evaluate(c, t)

                    arch = 4.0 * self.swing_h * (t / self.swing_T) * (1.0 - t / self.swing_T)
                    traj_points[i, dim] = z + arch
            else:

                t_vals = np.linspace(0, 1, n_samples)
                traj_points[:, dim] = p_start[dim] + t_vals * (p_end[dim] - p_start[dim])
        return traj_points
