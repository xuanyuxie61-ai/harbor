
import numpy as np
from typing import List, Tuple, Optional, Callable


class LayoutOptimizer:

    def __init__(self, n_turbines: int = 10,
                 domain: Tuple[float, float, float, float] = (0.0, 5000.0, 0.0, 5000.0),
                 min_spacing: float = 500.0,
                 rated_power: float = 5.0,
                 max_grid_capacity: float = 100.0):
        if n_turbines <= 0:
            raise ValueError("风机数量必须为正")
        if min_spacing <= 0:
            raise ValueError("最小间距必须为正")
        self.n_turbines = n_turbines
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.d_min = min_spacing
        self.rated_power = rated_power
        self.max_grid_capacity = max_grid_capacity
        self.positions = np.zeros((n_turbines, 2))

    def initialize_random(self, seed: Optional[int] = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.positions[:, 0] = rng.uniform(self.xmin, self.xmax, self.n_turbines)
        self.positions[:, 1] = rng.uniform(self.ymin, self.ymax, self.n_turbines)

    def initialize_grid(self):
        nx = int(np.ceil(np.sqrt(self.n_turbines * (self.xmax - self.xmin) / (self.ymax - self.ymin))))
        ny = int(np.ceil(self.n_turbines / nx))
        x = np.linspace(self.xmin + 100, self.xmax - 100, nx)
        y = np.linspace(self.ymin + 100, self.ymax - 100, ny)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        self.positions = pts[:self.n_turbines]

    def _nearest_neighbor_indices(self) -> List[Tuple[int, int]]:
        n = self.n_turbines
        nn_pairs = []
        for i in range(n):
            min_dist = float('inf')
            min_j = -1
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < min_dist:
                    min_dist = d
                    min_j = j
            nn_pairs.append((i, min_j))
        return nn_pairs

    def check_spacing_constraints(self) -> Tuple[bool, List[Tuple[int, int, float]]]:
        violations = []
        n = self.n_turbines
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < self.d_min:
                    violations.append((i, j, d))
        return len(violations) == 0, violations

    def repair_spacing(self, max_iterations: int = 1000) -> bool:
        alpha = 0.3
        for _ in range(max_iterations):
            ok, violations = self.check_spacing_constraints()
            if ok:
                return True
            for i, j, d in violations:
                if d < 1e-6:

                    theta = np.random.uniform(0, 2 * np.pi)
                    dx = self.min_spacing * 0.5 * np.cos(theta)
                    dy = self.min_spacing * 0.5 * np.sin(theta)
                    self.positions[i] += np.array([dx, dy])
                    self.positions[j] -= np.array([dx, dy])
                else:
                    direction = (self.positions[i] - self.positions[j]) / d
                    delta = alpha * (self.d_min - d) * direction
                    self.positions[i] += delta
                    self.positions[j] -= delta


            self.positions[:, 0] = np.clip(self.positions[:, 0], self.xmin, self.xmax)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.ymin, self.ymax)

        return False

    def cvt_optimize(self, objective_func: Callable[[np.ndarray], float],
                     n_samples: int = 5000,
                     n_iterations: int = 50,
                     step_size: float = 0.3) -> np.ndarray:
        rng = np.random.default_rng(42)
        n = self.n_turbines

        for it in range(n_iterations):

            samples = np.column_stack([
                rng.uniform(self.xmin, self.xmax, n_samples),
                rng.uniform(self.ymin, self.ymax, n_samples)
            ])


            assignments = np.zeros(n_samples, dtype=int)
            for s_idx, s in enumerate(samples):
                dists = np.linalg.norm(self.positions - s, axis=1)
                assignments[s_idx] = int(np.argmin(dists))


            new_positions = np.zeros_like(self.positions)
            counts = np.zeros(n)
            for s_idx, assign in enumerate(assignments):
                new_positions[assign] += samples[s_idx]
                counts[assign] += 1

            for i in range(n):
                if counts[i] > 0:
                    new_positions[i] /= counts[i]
                else:

                    new_positions[i] = rng.uniform(
                        [self.xmin, self.ymin], [self.xmax, self.ymax]
                    )


            beta = 0.7

            current_obj = objective_func(self.positions)
            grad = np.zeros_like(self.positions)
            eps = 10.0
            for i in range(n):
                for dim in range(2):
                    pos_plus = self.positions.copy()
                    pos_plus[i, dim] += eps

                    pos_plus[i, dim] = np.clip(pos_plus[i, dim],
                                               self.xmin if dim == 0 else self.ymin,
                                               self.xmax if dim == 0 else self.ymax)
                    obj_plus = objective_func(pos_plus)
                    grad[i, dim] = (obj_plus - current_obj) / eps


            self.positions = (1 - beta) * new_positions + \
                             beta * (self.positions + step_size * grad)


            self.positions[:, 0] = np.clip(self.positions[:, 0], self.xmin, self.xmax)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.ymin, self.ymax)


            self.repair_spacing(max_iterations=50)

        return self.positions

    def min_spacing(self) -> float:
        dist = self.pairwise_distances()
        n = self.n_turbines
        if n <= 1:
            return float('inf')
        np.fill_diagonal(dist, float('inf'))
        return float(np.min(dist))

    def pairwise_distances(self) -> np.ndarray:
        n = self.n_turbines
        if n == 0:
            return np.zeros((0, 0))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def capacity_subset_optimization(self, available_capacities: List[float],
                                      target_capacity: float) -> Tuple[bool, List[int]]:
        n = len(available_capacities)
        best_diff = float('inf')
        best_subset = []


        max_search = min(n, 15)
        for mask in range(1 << max_search):
            subset = []
            total = 0.0
            for i in range(max_search):
                if mask & (1 << i):
                    subset.append(i)
                    total += available_capacities[i]

            if total > self.max_grid_capacity:
                continue

            diff = abs(total - target_capacity)
            if diff < best_diff:
                best_diff = diff
                best_subset = subset

        if best_subset:
            return True, best_subset
        return False, []

    def compute_aep(self, wind_speeds: np.ndarray, wind_directions: np.ndarray,
                    power_func: Callable[[float], float],
                    wake_func: Callable[[int, float, float], float]) -> float:
        n_cases = len(wind_speeds)
        if n_cases == 0:
            return 0.0
        hours_per_year = 8760.0
        hours_per_case = hours_per_year / n_cases

        aep = 0.0
        for u0, theta in zip(wind_speeds, wind_directions):
            case_power = 0.0
            for i in range(self.n_turbines):
                u_eff = wake_func(i, u0, theta)
                case_power += power_func(u_eff)
            aep += case_power * hours_per_case

        return aep
