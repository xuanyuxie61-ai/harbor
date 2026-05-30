
import numpy as np
from typing import List, Callable, Tuple, Optional


class MonteCarloEngine:

    def __init__(self,
                 n_samples: int = 200,
                 n_strata: int = 4,
                 use_antithetic: bool = True,
                 use_control_variate: bool = False,
                 random_seed_base: int = 42):
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        self.n_samples = n_samples
        self.n_strata = n_strata
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.random_seed_base = random_seed_base

    def run_ensemble(self,
                     sampler: Callable[[int], np.ndarray],
                     observable: Callable[[np.ndarray], float]) -> dict:
        values = []
        strata_values = [[] for _ in range(self.n_strata)]

        for i in range(self.n_samples):
            seed = self.random_seed_base + i
            U = sampler(seed)
            val = observable(U)
            values.append(val)


            stratum_idx = i % self.n_strata
            strata_values[stratum_idx].append(val)

            if self.use_antithetic:

                anti_seed = self.random_seed_base + (self.n_samples - 1 - i)
                U_anti = sampler(anti_seed)
                val_anti = observable(U_anti)
                values.append(0.5 * (val + val_anti))

        arr = np.array(values, dtype=np.float64)
        mean_est = float(np.mean(arr))
        var_est = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        rmse_est = float(np.sqrt(var_est / len(arr)))


        strat_means = []
        strat_vars = []
        for layer in strata_values:
            if len(layer) > 1:
                strat_means.append(np.mean(layer))
                strat_vars.append(np.var(layer, ddof=1) / len(layer))
        strat_var_est = float(np.sum(strat_vars)) if strat_vars else var_est


        mid = len(arr) // 2
        if mid > 1:
            chain1 = arr[:mid]
            chain2 = arr[mid:2 * mid]
            W = 0.5 * (np.var(chain1, ddof=1) + np.var(chain2, ddof=1))
            B = mid * np.var([np.mean(chain1), np.mean(chain2)], ddof=1)
            if W > 1e-16:
                r_hat = np.sqrt((W + B / mid) / W)
            else:
                r_hat = 1.0
        else:
            r_hat = 1.0

        return {
            "mean": mean_est,
            "variance": var_est,
            "rmse": rmse_est,
            "stratified_variance": strat_var_est,
            "r_hat": float(r_hat),
            "n_effective": len(arr),
        }

    def confidence_interval(self,
                            values: np.ndarray,
                            level: float = 0.95) -> Tuple[float, float]:
        from math import erf, sqrt
        if len(values) < 2:
            return (float(values[0]), float(values[0]))
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        z = sqrt(2.0) * erf(level)
        margin = z * std / sqrt(len(values))
        return (mean - margin, mean + margin)


def hierarchical_distance_matrix(n: int) -> np.ndarray:

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            d = abs(i - j) + phi * np.sin(i * j)
            dist[i, j] = d
            dist[j, i] = d

    max_d = dist.max()
    if max_d > 0:
        dist /= max_d
    return dist


def stratified_sampler(seeds: List[int],
                       sampler: Callable[[int], np.ndarray]) -> List[np.ndarray]:
    samples = []
    for s in seeds:
        samples.append(sampler(s))
    return samples
