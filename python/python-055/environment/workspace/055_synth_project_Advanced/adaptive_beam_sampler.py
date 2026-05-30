
import numpy as np


class DiskGridGenerator:

    @staticmethod
    def disk_grid_count(n: int) -> int:
        if n < 0:
            return 0
        ng = 1
        for j in range(1, n + 1):
            i = 0
            while True:
                i += 1
                x = 2.0 * i / (2.0 * n + 1.0)
                y = 2.0 * j / (2.0 * n + 1.0)
                if x ** 2 + y ** 2 > 1.0:
                    break
                if j == 0:
                    ng += 2
                else:
                    ng += 4
        return ng

    @staticmethod
    def disk_grid(n: int, r: float = 1.0, c: np.ndarray = None) -> np.ndarray:
        if c is None:
            c = np.zeros(2)
        c = np.asarray(c, dtype=np.float64)

        points = []

        points.append([c[0], c[1]])

        for j in range(1, n + 1):
            y = c[1] + r * 2.0 * j / (2.0 * n + 1.0)

            y_vals = [y]
            if j > 0:
                y_vals.append(2.0 * c[1] - y)

            for y_val in y_vals:

                points.append([c[0], y_val])
                i = 0
                while True:
                    i += 1
                    x = c[0] + r * 2.0 * i / (2.0 * n + 1.0)
                    if (x - c[0]) ** 2 + (y_val - c[1]) ** 2 > r ** 2 + 1e-12:
                        break

                    points.append([x, y_val])
                    points.append([2.0 * c[0] - x, y_val])

        return np.array(points, dtype=np.float64)


class CVTBeamOptimizer:

    def __init__(self, bounds: tuple, density_func=None):
        self.bounds = bounds
        self.density_func = density_func

    def _region_sampler(self, n_samples: int) -> np.ndarray:
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        pts = np.random.rand(n_samples, 2)
        pts[:, 0] = x_min + (x_max - x_min) * pts[:, 0]
        pts[:, 1] = y_min + (y_max - y_min) * pts[:, 1]
        return pts

    def _find_closest(self, generators: np.ndarray, point: np.ndarray) -> int:
        diff = generators - point
        dists = np.sum(diff ** 2, axis=1)
        return int(np.argmin(dists))

    def optimize(
        self,
        n_generators: int,
        n_samples: int = 5000,
        n_iterations: int = 20,
        init_points: np.ndarray = None
    ) -> np.ndarray:
        if init_points is not None:
            generators = np.asarray(init_points, dtype=np.float64).copy()
        else:
            generators = self._region_sampler(n_generators)

        for it in range(n_iterations):
            samples = self._region_sampler(n_samples)
            new_generators = np.zeros_like(generators)
            counts = np.zeros(n_generators)

            for s in samples:
                idx = self._find_closest(generators, s)
                new_generators[idx] += s
                counts[idx] += 1.0

            for j in range(n_generators):
                if counts[j] > 0:
                    new_generators[j] /= counts[j]
                else:

                    new_generators[j] = self._region_sampler(1)[0]

            change = np.linalg.norm(new_generators - generators, 'fro')
            generators = new_generators
            if change < 1e-6:
                break


        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        generators[:, 0] = np.clip(generators[:, 0], x_min, x_max)
        generators[:, 1] = np.clip(generators[:, 1], y_min, y_max)

        return generators


class KnapsackBeamSelector:

    @staticmethod
    def random_subset(n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=n)

    @staticmethod
    def greedy_select(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
        n = len(values)
        density = values / (weights + 1e-15)
        order = np.argsort(-density)

        selected = np.zeros(n, dtype=int)
        total_weight = 0.0
        for idx in order:
            if total_weight + weights[idx] <= capacity:
                selected[idx] = 1
                total_weight += weights[idx]
        return selected

    @staticmethod
    def random_search_select(
        values: np.ndarray,
        weights: np.ndarray,
        capacity: float,
        n_trials: int = 200,
        seed: int = 55
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = len(values)
        best_value = -1.0
        best_subset = np.zeros(n, dtype=int)

        for _ in range(n_trials):
            subset = rng.integers(0, 2, size=n)
            total_weight = np.dot(subset, weights)
            if total_weight <= capacity:
                total_value = np.dot(subset, values)
                if total_value > best_value:
                    best_value = total_value
                    best_subset = subset.copy()

        return best_subset

    @staticmethod
    def optimize_beam_subset(
        beam_info_gains: np.ndarray,
        beam_costs: np.ndarray,
        time_budget: float,
        method: str = "greedy"
    ) -> dict:
        if method == "greedy":
            selected = KnapsackBeamSelector.greedy_select(
                beam_info_gains, beam_costs, time_budget
            )
        elif method == "random_search":
            selected = KnapsackBeamSelector.random_search_select(
                beam_info_gains, beam_costs, time_budget
            )
        else:
            raise ValueError("method 必须是 'greedy' 或 'random_search'")

        total_gain = float(np.dot(selected, beam_info_gains))
        total_cost = float(np.dot(selected, beam_costs))

        return {
            'selected': selected,
            'total_gain': total_gain,
            'total_cost': total_cost,
            'n_selected': int(np.sum(selected)),
        }


class AdaptiveSamplingPlanner:

    def __init__(self, survey_area: tuple):
        self.survey_area = survey_area
        self.cvt_optimizer = CVTBeamOptimizer(survey_area)

    def plan_survey(
        self,
        n_beams: int = 64,
        time_budget: float = 3600.0,
        n_cvt_iter: int = 15
    ) -> dict:

        beam_positions = self.cvt_optimizer.optimize(
            n_generators=n_beams,
            n_iterations=n_cvt_iter
        )


        info_gains = np.zeros(n_beams)
        for i in range(n_beams):

            dists = np.linalg.norm(beam_positions - beam_positions[i], axis=1)
            dists = dists[dists > 1e-12]
            if len(dists) > 0:
                info_gains[i] = np.min(dists) ** 2
            else:
                info_gains[i] = 1.0


        center = np.array([
            (self.survey_area[0][0] + self.survey_area[0][1]) / 2.0,
            (self.survey_area[1][0] + self.survey_area[1][1]) / 2.0,
        ])
        dists_to_center = np.linalg.norm(beam_positions - center, axis=1)
        beam_costs = 10.0 + 0.5 * dists_to_center


        result = KnapsackBeamSelector.optimize_beam_subset(
            info_gains, beam_costs, time_budget, method="greedy"
        )

        selected_positions = beam_positions[result['selected'] == 1]

        return {
            'all_positions': beam_positions,
            'selected_positions': selected_positions,
            'selected_mask': result['selected'],
            'total_gain': result['total_gain'],
            'total_cost': result['total_cost'],
            'n_selected': result['n_selected'],
            'coverage_efficiency': result['total_gain'] / (result['total_cost'] + 1e-15),
        }
