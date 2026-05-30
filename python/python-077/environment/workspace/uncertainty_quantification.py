
import numpy as np
from typing import Tuple, List, Callable, Optional
from numerical_utils import alnorm, alnorm_array, weibull_pdf


class UncertaintyQuantification:

    def __init__(self, n_mc_samples: int = 1000, seed: Optional[int] = 42):
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(seed)

    def propagate_speed_uncertainty(self, u_nominal: float,
                                     sigma_u: float,
                                     power_curve: Callable[[float], float]) -> Tuple[float, float]:
        samples = self.rng.normal(u_nominal, sigma_u, self.n_mc_samples)
        samples = np.maximum(samples, 0.0)
        powers = np.array([power_curve(float(u)) for u in samples])
        return float(np.mean(powers)), float(np.std(powers))

    def aep_monte_carlo(self, wind_speed_distribution: Callable[[int], np.ndarray],
                        power_curve: Callable[[float], float],
                        wake_model_func: Callable[[np.ndarray], np.ndarray],
                        sigma_params: dict) -> Tuple[float, float, np.ndarray]:
        aep_samples = np.zeros(self.n_mc_samples)

        for i in range(self.n_mc_samples):

            u_samples = wind_speed_distribution(100)

            u_eff = wake_model_func(u_samples)

            powers = np.array([power_curve(float(u)) for u in u_eff])

            hours = 8760.0
            aep_samples[i] = np.mean(powers) * hours

        return float(np.mean(aep_samples)), float(np.std(aep_samples)), aep_samples

    def confidence_interval(self, samples: np.ndarray,
                            confidence: float = 0.95) -> Tuple[float, float, float]:
        alpha = 1.0 - confidence
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        n = len(samples)



        z = self._inverse_normal_cdf(1.0 - alpha / 2.0)
        margin = z * std / np.sqrt(n)
        return mean, mean - margin, mean + margin

    def _inverse_normal_cdf(self, p: float, tol: float = 1e-8) -> float:
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0

        lo, hi = -10.0, 10.0
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            cdf_mid = 1.0 - alnorm(mid, upper=True)
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def ks_test_normality(self, samples: np.ndarray,
                          n_bins: int = 20) -> Tuple[float, bool]:
        samples = np.asarray(samples, dtype=float)
        n = len(samples)
        if n < 3:
            return 0.0, False

        mu = np.mean(samples)
        sigma = np.std(samples, ddof=1)
        if sigma < 1e-14:
            return 0.0, False

        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, n + 1) / n
        z_scores = (sorted_samples - mu) / sigma
        theoretical_cdf = 1.0 - alnorm_array(z_scores, upper=True)

        d_statistic = float(np.max(np.abs(empirical_cdf - theoretical_cdf.ravel())))


        critical_value = 1.36 / np.sqrt(n)
        reject = d_statistic > critical_value
        return d_statistic, reject

    def sensitivity_analysis(self, base_aep: float,
                              param_names: List[str],
                              param_base: List[float],
                              param_sigma: List[float],
                              aep_func: Callable[[List[float]], float],
                              delta: float = 0.01) -> dict:
        results = {}
        n_params = len(param_names)

        for k in range(n_params):
            name = param_names[k]
            p_base = param_base[k]
            p_sigma = param_sigma[k]


            p_plus = param_base.copy()
            p_minus = param_base.copy()
            step = max(abs(p_base) * delta, 1e-6)
            p_plus[k] += step
            p_minus[k] -= step

            aep_plus = aep_func(p_plus)
            aep_minus = aep_func(p_minus)

            derivative = (aep_plus - aep_minus) / (2.0 * step)
            sensitivity = derivative * p_sigma / base_aep if base_aep != 0 else 0.0

            results[name] = {
                'derivative': derivative,
                'sensitivity_index': sensitivity,
                'uncertainty_contribution': abs(derivative * p_sigma)
            }

        return results

    def integral_deficit_uncertainty(self, deficit_func: Callable[[float], float],
                                      x_range: Tuple[float, float],
                                      sigma_x: float,
                                      n_points: int = 100) -> Tuple[float, float]:
        x1, x2 = x_range
        integral_samples = np.zeros(self.n_mc_samples)

        for i in range(self.n_mc_samples):

            dx = self.rng.normal(0.0, sigma_x)
            x1_p = max(0.0, x1 + dx)
            x2_p = max(x1_p + 1.0, x2 + dx)

            x = np.linspace(x1_p, x2_p, n_points)
            dx = x[1] - x[0]
            deficits = np.array([deficit_func(float(xi)) for xi in x])

            integral_samples[i] = dx / 3.0 * (
                deficits[0] + deficits[-1] +
                4.0 * np.sum(deficits[1:-1:2]) +
                2.0 * np.sum(deficits[2:-1:2])
            )

        return float(np.mean(integral_samples)), float(np.std(integral_samples))
