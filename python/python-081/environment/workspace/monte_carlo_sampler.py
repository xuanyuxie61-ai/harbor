
import numpy as np
from typing import Callable, Tuple, Optional
from uncertainty_quantification import truncated_normal_sample


def monte_carlo_simulation(model_func: Callable[[np.ndarray], float],
                            mu_params: np.ndarray,
                            sigma_params: np.ndarray,
                            bounds: np.ndarray,
                            n_samples: int = 1000,
                            seed: int = 42) -> dict:
    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    responses = np.zeros(n_samples, dtype=np.float64)
    param_samples = np.zeros((n_samples, n_param), dtype=np.float64)

    for i in range(n_samples):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        param_samples[i] = sample
        try:
            responses[i] = model_func(sample)
        except Exception:
            responses[i] = np.nan


    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask]
    n_valid = len(valid_responses)

    if n_valid == 0:
        raise ValueError("所有蒙特卡洛样本均产生NaN，模型不稳定")

    mean_val = float(np.mean(valid_responses))
    std_val = float(np.std(valid_responses, ddof=1))
    se = std_val / np.sqrt(n_valid)
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se


    q25 = float(np.percentile(valid_responses, 25))
    q50 = float(np.percentile(valid_responses, 50))
    q75 = float(np.percentile(valid_responses, 75))

    return {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "mean": mean_val,
        "std": std_val,
        "se": se,
        "ci_95": (ci_lower, ci_upper),
        "q25": q25,
        "median": q50,
        "q75": q75,
        "min": float(np.min(valid_responses)),
        "max": float(np.max(valid_responses)),
        "responses": valid_responses,
        "param_samples": param_samples[valid_mask],
    }


def estimate_failure_probability(model_func: Callable[[np.ndarray], float],
                                  threshold: float,
                                  mu_params: np.ndarray,
                                  sigma_params: np.ndarray,
                                  bounds: np.ndarray,
                                  n_samples: int = 5000,
                                  seed: int = 42) -> dict:
    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    failures = 0
    responses = []

    for _ in range(n_samples):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        try:
            y = model_func(sample)
            responses.append(y)
            if y < threshold:
                failures += 1
        except Exception:
            pass

    pf = failures / n_samples if n_samples > 0 else 0.0

    z = 1.96
    n = n_samples
    denom = 1 + z**2 / n
    centre = (pf + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((pf * (1 - pf) + z**2 / (4 * n)) / n) / denom

    return {
        "failure_probability": pf,
        "pf_ci_lower": max(0.0, centre - margin),
        "pf_ci_upper": min(1.0, centre + margin),
        "n_failures": failures,
        "n_samples": n_samples,
    }


def convergence_analysis(model_func: Callable[[np.ndarray], float],
                          mu_params: np.ndarray,
                          sigma_params: np.ndarray,
                          bounds: np.ndarray,
                          sample_sizes: Optional[np.ndarray] = None,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_sizes is None:
        sample_sizes = np.array([50, 100, 200, 500, 1000, 2000], dtype=np.int32)

    rng = np.random.default_rng(seed=seed)
    n_param = len(mu_params)
    means = []
    stds = []

    max_n = int(np.max(sample_sizes))
    all_samples = np.zeros((max_n, n_param), dtype=np.float64)
    all_responses = np.zeros(max_n, dtype=np.float64)

    for i in range(max_n):
        sample = np.zeros(n_param, dtype=np.float64)
        for p in range(n_param):
            s = truncated_normal_sample(
                mu_params[p], sigma_params[p],
                bounds[p, 0], bounds[p, 1],
                n_samples=1, rng=rng
            )
            sample[p] = s[0]
        all_samples[i] = sample
        try:
            all_responses[i] = model_func(sample)
        except Exception:
            all_responses[i] = np.nan

    for N in sample_sizes:
        vals = all_responses[:N]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            means.append(float(np.mean(valid)))
            stds.append(float(np.std(valid, ddof=1)))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return sample_sizes, np.array(means), np.array(stds)
