
import numpy as np
from typing import Callable, Tuple, List, Optional
from utils import clip_to_bounds


class LogisticMap:

    def __init__(self, r: float = 4.0, x0: float = 0.3):
        if not (3.57 <= r <= 4.0):
            r = clip_to_bounds(np.array([r]), np.array([3.57]), np.array([4.0]))[0]
        self.r = r
        self.x = x0

    def next(self) -> float:
        self.x = self.r * self.x * (1.0 - self.x)
        return self.x

    def generate(self, n: int) -> np.ndarray:
        seq = np.zeros(n)
        for i in range(n):
            seq[i] = self.next()
        return seq


class BarnsleyFernIFS:

    def __init__(self):

        self.A = [
            np.array([[0.0, 0.0], [0.0, 0.16]]),
            np.array([[0.85, 0.04], [-0.04, 0.85]]),
            np.array([[0.2, -0.26], [0.23, 0.22]]),
            np.array([[-0.15, 0.28], [0.26, 0.24]]),
        ]
        self.b = [
            np.array([0.0, 0.0]),
            np.array([0.0, 1.6]),
            np.array([0.0, 1.6]),
            np.array([0.0, 0.44]),
        ]
        self.probs = np.array([0.01, 0.85, 0.07, 0.07])
        self.state = np.array([0.0, 0.0])

    def step(self) -> np.ndarray:
        k = np.random.choice(len(self.A), p=self.probs)
        self.state = self.A[k] @ self.state + self.b[k]
        return self.state.copy()

    def sample(self, n: int, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        lower, upper = bounds
        lower2 = np.asarray(lower).flatten()[:2]
        upper2 = np.asarray(upper).flatten()[:2]
        samples = np.zeros((n, 2))
        for i in range(n):
            s = self.step()

            s_mapped = lower2 + (s - np.array([-2.5, 0.0])) / np.array([5.0, 10.0]) * (upper2 - lower2)
            samples[i] = clip_to_bounds(s_mapped, lower2, upper2)
        return samples


class ChaoticSimulatedAnnealing:

    def __init__(self, dim: int, bounds: np.ndarray,
                 T0: float = 10.0, T_min: float = 1e-4,
                 cooling_rate: float = 0.95, max_iter: int = 500,
                 logistic_r: float = 4.0):
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=float)
        self.T0 = T0
        self.T_min = T_min
        self.alpha = cooling_rate
        self.max_iter = max_iter
        self.chaos_maps = [LogisticMap(r=logistic_r, x0=0.1 + 0.1 * i) for i in range(dim)]

    def optimize(self, objective: Callable[[np.ndarray], float],
                 x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        if x0 is None:
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        x_current = clip_to_bounds(x0, self.bounds[:, 0], self.bounds[:, 1])
        f_current = objective(x_current)
        x_best = x_current.copy()
        f_best = f_current
        T = self.T0
        beta = 0.5

        for iteration in range(self.max_iter):
            if T < self.T_min:
                break

            chaos_vals = np.array([m.next() for m in self.chaos_maps])
            delta = beta * T * (2.0 * chaos_vals - 1.0)
            x_new = x_current + delta
            x_new = clip_to_bounds(x_new, self.bounds[:, 0], self.bounds[:, 1])
            f_new = objective(x_new)
            delta_E = f_new - f_current

            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (T + 1e-12)):
                x_current = x_new.copy()
                f_current = f_new
                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new

            T *= self.alpha

        return x_best, f_best


class GaitParameterOptimizer:

    def __init__(self):
        self.bounds = np.array([
            [0.4, 1.5],
            [0.05, 0.30],
            [0.02, 0.10],
            [1.0, 15.0],
            [0.5, 5.0],
        ])
        self.csa = ChaoticSimulatedAnnealing(
            dim=5, bounds=self.bounds, T0=5.0, max_iter=300
        )
        self.ifs = BarnsleyFernIFS()

    def evaluate_gait_fitness(self, params: np.ndarray,
                              stability_margin_func: Callable[[np.ndarray], float],
                              energy_cost_func: Callable[[np.ndarray], float]) -> float:
        T_gait, stride, swing_h, coupling, damping = params

        margin = stability_margin_func(params)
        energy = energy_cost_func(params)

        penalty = 0.0
        if swing_h > 0.5 * stride:
            penalty += 100.0 * (swing_h - 0.5 * stride)
        if T_gait < 0.3:
            penalty += 50.0 * (0.3 - T_gait)
        fitness = -5.0 * margin + 2.0 * energy + penalty
        return fitness

    def optimize(self, stability_margin_func: Callable[[np.ndarray], float],
                 energy_cost_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        def obj(x):
            return self.evaluate_gait_fitness(x, stability_margin_func, energy_cost_func)


        best_f = float('inf')
        best_x = None
        ifs_samples = self.ifs.sample(20, (self.bounds[:, 0], self.bounds[:, 1]))
        for s in ifs_samples:

            s5 = np.concatenate([s[:2], np.random.uniform(self.bounds[2:, 0], self.bounds[2:, 1])])
            f = obj(s5)
            if f < best_f:
                best_f = f
                best_x = s5.copy()


        x_opt, f_opt = self.csa.optimize(obj, x0=best_x)
        return x_opt, f_opt
