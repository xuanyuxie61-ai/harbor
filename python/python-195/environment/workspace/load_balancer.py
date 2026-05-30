
import numpy as np
from typing import Tuple, List, Optional
from utils import check_bounds


class LoadBalancer:
    def __init__(self, n_procs: int, domain: Tuple[float, float, float, float],
                 imbalance_threshold: float = 1.3,
                 migration_cost_factor: float = 1.0e-4):
        self.n_procs = n_procs
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.imbalance_threshold = imbalance_threshold
        self.migration_cost_factor = migration_cost_factor


        self._init_decomposition()

    def _init_decomposition(self):

        px = int(np.sqrt(self.n_procs))
        while self.n_procs % px != 0 and px > 1:
            px -= 1
        py = self.n_procs // px
        self.px = px
        self.py = py

        self.proc_bounds = []
        dx = (self.xmax - self.xmin) / px
        dy = (self.ymax - self.ymin) / py
        for j in range(py):
            for i in range(px):
                bounds = (
                    self.xmin + i * dx,
                    self.xmin + (i + 1) * dx,
                    self.ymin + j * dy,
                    self.ymin + (j + 1) * dy
                )
                self.proc_bounds.append(bounds)

    def compute_loads(self, particles: np.ndarray,
                      field_cost: Optional[np.ndarray] = None) -> np.ndarray:
        particles = np.asarray(particles, dtype=float)
        loads = np.zeros(self.n_procs, dtype=float)

        for p in range(self.n_procs):
            bxmin, bxmax, bymin, bymax = self.proc_bounds[p]
            mask = (
                (particles[:, 0] >= bxmin) & (particles[:, 0] < bxmax) &
                (particles[:, 1] >= bymin) & (particles[:, 1] < bymax)
            )

            if p % self.px == self.px - 1:
                mask = mask | ((particles[:, 0] >= bxmax - 1e-12) & (particles[:, 0] <= self.xmax + 1e-12))
            if p // self.px == self.py - 1:
                mask = mask | ((particles[:, 1] >= bymax - 1e-12) & (particles[:, 1] <= self.ymax + 1e-12))

            n_particles = np.sum(mask)

            loads[p] = float(n_particles)
            if field_cost is not None and p < len(field_cost):
                loads[p] += field_cost[p]

        return loads

    def imbalance_factor(self, loads: np.ndarray) -> float:
        avg = np.mean(loads)
        if avg < 1e-14:
            return 1.0
        return np.max(loads) / avg

    def find_optimal_split(self, particles_in_domain: np.ndarray,
                           axis: int = 0) -> Tuple[float, float, float]:
        particles_in_domain = np.asarray(particles_in_domain, dtype=float)
        if particles_in_domain.shape[0] == 0:
            mid = (self.domain[axis * 2 + 1] + self.domain[axis * 2]) / 2.0
            return mid, 0.0, 0.0

        coords = particles_in_domain[:, axis]
        coords_sorted = np.sort(coords)
        n = len(coords_sorted)


        best_diff = float('inf')
        best_split = coords_sorted[n // 2]


        for idx in range(1, n):
            split = 0.5 * (coords_sorted[idx - 1] + coords_sorted[idx])
            left_load = float(idx)
            right_load = float(n - idx)
            diff = abs(left_load - right_load)
            if diff < best_diff:
                best_diff = diff
                best_split = split

        left_mask = coords < best_split
        load_left = float(np.sum(left_mask))
        load_right = float(n) - load_left

        return best_split, load_left, load_right

    def recursive_bisection(self, particles: np.ndarray,
                            bounds: Tuple[float, float, float, float],
                            proc_id: int, n_subprocs: int,
                            result: dict, depth: int = 0):
        xmin, xmax, ymin, ymax = bounds
        if n_subprocs == 1:
            result[proc_id] = {
                'bounds': bounds,
                'n_particles': particles.shape[0],
                'depth': depth
            }
            return


        if particles.shape[0] > 0:
            std_x = np.std(particles[:, 0]) if particles.shape[0] > 1 else 0.0
            std_y = np.std(particles[:, 1]) if particles.shape[0] > 1 else 0.0
            axis = 0 if std_x >= std_y else 1
        else:
            axis = 0 if (xmax - xmin) >= (ymax - ymin) else 1

        split, load_left, load_right = self.find_optimal_split(particles, axis)


        total_load = load_left + load_right
        if total_load < 1e-14:
            n_left = n_subprocs // 2
        else:
            n_left = max(1, min(n_subprocs - 1,
                                 int(round(n_subprocs * load_left / total_load))))
        n_right = n_subprocs - n_left

        if axis == 0:
            bounds_left = (xmin, split, ymin, ymax)
            bounds_right = (split, xmax, ymin, ymax)
        else:
            bounds_left = (xmin, xmax, ymin, split)
            bounds_right = (xmin, xmax, split, ymax)


        if axis == 0:
            mask_left = particles[:, 0] < split
        else:
            mask_left = particles[:, 1] < split

        particles_left = particles[mask_left]
        particles_right = particles[~mask_left]

        self.recursive_bisection(particles_left, bounds_left, proc_id, n_left, result, depth + 1)
        self.recursive_bisection(particles_right, bounds_right, proc_id + n_left, n_right, result, depth + 1)

    def rebalance(self, particles: np.ndarray,
                  field_cost: Optional[np.ndarray] = None) -> dict:
        current_loads = self.compute_loads(particles, field_cost)
        current_imbalance = self.imbalance_factor(current_loads)

        result = {
            'old_imbalance': current_imbalance,
            'old_loads': current_loads,
            'rebalanced': False,
            'new_decomposition': None,
            'migration_count': 0
        }

        if current_imbalance <= self.imbalance_threshold:
            result['new_imbalance'] = current_imbalance
            return result


        new_decomp = {}
        self.recursive_bisection(
            particles, self.domain, 0, self.n_procs, new_decomp
        )


        new_bounds = []
        for p in range(self.n_procs):
            new_bounds.append(new_decomp[p]['bounds'])
        self.proc_bounds = new_bounds


        new_loads = self.compute_loads(particles, field_cost)
        new_imbalance = self.imbalance_factor(new_loads)


        migration_count = 0

        if current_imbalance > 1.0:
            migration_count = int(particles.shape[0] * (current_imbalance - 1.0) / current_imbalance)

        result.update({
            'rebalanced': True,
            'new_decomposition': new_decomp,
            'new_imbalance': new_imbalance,
            'new_loads': new_loads,
            'migration_count': migration_count
        })

        return result

    def evaluate_efficiency(self, loads: np.ndarray) -> dict:
        mu = np.mean(loads)
        sigma = np.std(loads)
        I = self.imbalance_factor(loads)
        cv = sigma / mu if mu > 1e-14 else 0.0
        eta = 1.0 / I if I > 1e-14 else 0.0

        return {
            'mean_load': mu,
            'std_load': sigma,
            'imbalance_factor': I,
            'coefficient_variation': cv,
            'parallel_efficiency': eta
        }


def diffusion_based_load_balance(loads: np.ndarray,
                                  connectivity: np.ndarray,
                                  n_iterations: int = 100,
                                  tolerance: float = 1e-3) -> np.ndarray:
    loads = np.asarray(loads, dtype=float).copy()
    n = len(loads)
    degrees = np.sum(connectivity, axis=1)
    degrees = np.maximum(degrees, 1.0)

    alpha = 0.5 / np.max(degrees)

    for it in range(n_iterations):
        loads_old = loads.copy()
        for i in range(n):
            for j in range(n):
                if connectivity[i, j] > 0:
                    delta = alpha * (loads_old[i] - loads_old[j])
                    loads[i] -= delta
                    loads[j] += delta

        max_diff = np.max(np.abs(loads - loads_old))
        if max_diff < tolerance:
            break

    return loads
