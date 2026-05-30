
import numpy as np
from typing import Callable, Tuple, List, Optional






def sample_hypercube_uniform(d: int, n_samples: int,
                              bounds: Optional[np.ndarray] = None,
                              seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.rand(n_samples, d)
    if bounds is not None:
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    return samples


def hypercube_distance_statistics(samples: np.ndarray) -> dict:
    n = samples.shape[0]
    if n < 2:
        return {"mean": 0.0, "var": 0.0, "min": 0.0, "max": 0.0}

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(samples[i] - samples[j])
            dists.append(d)
    dists = np.array(dists)
    return {
        "mean": float(np.mean(dists)),
        "var": float(np.var(dists)),
        "std": float(np.std(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "median": float(np.median(dists)),
    }






def inverse_transform_sampling(cdf_func: Callable[[float], float],
                                n_samples: int,
                                xmin: float = -3.0,
                                xmax: float = 3.0,
                                n_grid: int = 1000,
                                seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    x_grid = np.linspace(xmin, xmax, n_grid)
    cdf_grid = np.array([cdf_func(x) for x in x_grid])

    cdf_grid = np.maximum.accumulate(cdf_grid)

    cdf_min, cdf_max = cdf_grid[0], cdf_grid[-1]
    if cdf_max > cdf_min:
        cdf_grid = (cdf_grid - cdf_min) / (cdf_max - cdf_min)
    else:
        cdf_grid = np.linspace(0.0, 1.0, n_grid)

    U = np.random.rand(n_samples)

    samples = np.interp(U, cdf_grid, x_grid)
    return samples


def gaussian_cdf_approx(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    from math import erf
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))


def sample_gaussian(mu: float, sigma: float, n_samples: int,
                     seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(mu, sigma, n_samples)


def sample_discrete_distribution(pmf: np.ndarray, n_samples: int,
                                  seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    pmf = np.array(pmf, dtype=np.float64)
    pmf = pmf / np.sum(pmf)
    cdf = np.cumsum(pmf)
    U = np.random.rand(n_samples)
    samples = np.searchsorted(cdf, U)
    return samples






def monte_carlo_reliability(performance_func: Callable[[np.ndarray], float],
                            input_sampler: Callable[[], np.ndarray],
                            n_samples: int,
                            threshold: float = 0.0) -> dict:
    failures = 0
    g_vals = []
    for _ in range(n_samples):
        X = input_sampler()
        g = performance_func(X)
        g_vals.append(g)
        if g <= threshold:
            failures += 1

    pf = failures / n_samples
    std_err = np.sqrt(pf * (1.0 - pf) / n_samples)
    cov = std_err / (pf + 1e-14)

    g_vals = np.array(g_vals)
    return {
        "pf_estimate": pf,
        "std_error": std_err,
        "cov": cov,
        "mean_g": float(np.mean(g_vals)),
        "std_g": float(np.std(g_vals)),
        "min_g": float(np.min(g_vals)),
        "max_g": float(np.max(g_vals)),
        "n_samples": n_samples,
    }


def latin_hypercube_sampling(d: int, n_samples: int,
                              bounds: Optional[np.ndarray] = None,
                              seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    samples = np.zeros((n_samples, d), dtype=np.float64)
    for dim in range(d):

        perm = np.random.permutation(n_samples)

        u = (perm + np.random.rand(n_samples)) / n_samples
        samples[:, dim] = u
    if bounds is not None:
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    return samples






def sobol_first_order_indices(func: Callable[[np.ndarray], float],
                               d: int, n_samples: int = 1024,
                               bounds: Optional[np.ndarray] = None) -> np.ndarray:
    A = latin_hypercube_sampling(d, n_samples, bounds, seed=42)
    B = latin_hypercube_sampling(d, n_samples, bounds, seed=43)

    f_A = np.array([func(a) for a in A])
    f_B = np.array([func(b) for b in B])
    var_y = np.var(f_A)
    if var_y < 1e-14:
        return np.zeros(d)

    S1 = np.zeros(d)
    for i in range(d):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        f_ABi = np.array([func(x) for x in A_Bi])

        S1[i] = np.mean(f_B * (f_ABi - f_A)) / var_y

        S1[i] = max(0.0, min(1.0, S1[i]))
    return S1






def generate_am_process_parameters(n_samples: int,
                                    seed: Optional[int] = None) -> np.ndarray:
    d = 5
    samples = latin_hypercube_sampling(d, n_samples, seed=seed)
    return samples


def parameter_to_physical(sample: np.ndarray) -> dict:
    return {
        "laser_power_var": 0.9 + 0.2 * sample[0],
        "scan_speed_var": 0.9 + 0.2 * sample[1],
        "layer_thickness_var": 0.85 + 0.3 * sample[2],
        "preheat_temp_var": 0.95 + 0.1 * sample[3],
        "powder_size_var": 0.8 + 0.4 * sample[4],
    }
