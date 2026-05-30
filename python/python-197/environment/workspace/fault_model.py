
import math
import numpy as np
from special_functions import gammad, ppchi2, ppnd, alnorm, digamma


class GammaFaultModel:

    def __init__(self, alpha: float = 2.0, beta: float = 0.001):
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return (self.beta ** self.alpha / math.gamma(self.alpha)
                * t ** (self.alpha - 1.0) * math.exp(-self.beta * t))

    def cdf(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        x = self.beta * t
        val, _ = gammad(x, self.alpha)
        return val

    def survival(self, t: float) -> float:
        return 1.0 - self.cdf(t)

    def hazard(self, t: float) -> float:
        s = self.survival(t)
        if s < 1.0e-14:
            return 0.0
        return self.pdf(t) / s

    def mean(self) -> float:
        return self.alpha / self.beta

    def variance(self) -> float:
        return self.alpha / (self.beta ** 2)

    def entropy(self) -> float:
        psi_val, _ = digamma(self.alpha)
        return (self.alpha - math.log(self.beta) + math.lgamma(self.alpha)
                + (1.0 - self.alpha) * psi_val)

    def quantile(self, p: float) -> float:
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0e10
        g = math.lgamma(self.alpha)
        chi2_val, _ = ppchi2(p, 2.0 * self.alpha, g)
        return chi2_val / (2.0 * self.beta)

    def sample(self, size: int = 1, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.gamma(shape=self.alpha, scale=1.0 / self.beta, size=size)


class FaultPredictor:

    def __init__(self, significance: float = 0.05):
        self.significance = significance
        self.history = []

    def observe(self, inter_arrival: float):
        if inter_arrival > 0.0:
            self.history.append(inter_arrival)

    def test_increase(self) -> tuple:
        if len(self.history) < 3:
            return False, 1.0
        arr = np.array(self.history[-20:])
        n = len(arr)
        mean_est = np.mean(arr)
        var_est = np.var(arr, ddof=1)
        if var_est <= 0.0:
            return False, 1.0


        sigma0_sq = mean_est ** 2
        if sigma0_sq <= 0.0:
            return False, 1.0
        chi2_stat = (n - 1) * var_est / sigma0_sq

        from special_functions import gammad
        cdf, _ = gammad(chi2_stat * 0.5, (n - 1) * 0.5)
        p_value = 1.0 - cdf
        is_increased = p_value < self.significance
        return is_increased, p_value

    def recommended_checkpoint_interval(self, safety_factor: float = 0.8) -> float:
        if len(self.history) < 2:
            return 100.0
        arr = np.array(self.history)
        mean_est = np.mean(arr)
        var_est = np.var(arr, ddof=1)
        if var_est <= 0.0:
            return safety_factor * mean_est

        alpha_est = mean_est ** 2 / var_est
        beta_est = mean_est / var_est
        model = GammaFaultModel(alpha=max(alpha_est, 0.1), beta=max(beta_est, 1.0e-6))
        q = model.quantile(0.5)
        return safety_factor * q
