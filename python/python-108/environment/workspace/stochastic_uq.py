# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional, Callable


class HypersphereSampler:

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim < 2:
            raise ValueError("维度必须 ≥ 2")
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        g = self.rng.standard_normal(size=(n, self.dim))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return g / norms

    def angle_statistics(self, n_pairs: int) -> dict:
        u = self.sample(n_pairs)
        v = self.sample(n_pairs)
        cos_theta = np.sum(u * v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        abs_cos = np.abs(cos_theta)
        theta = np.arccos(abs_cos)
        return {
            "dim": self.dim,
            "n_pairs": n_pairs,
            "mean_abs_cos": float(np.mean(abs_cos)),
            "std_abs_cos": float(np.std(abs_cos, ddof=1)),
            "mean_angle_rad": float(np.mean(theta)),
            "std_angle_rad": float(np.std(theta, ddof=1)),
            "theoretical_mean_abs_cos": self._theoretical_mean_abs_cos(),
        }

    def _theoretical_mean_abs_cos(self) -> float:
        from scipy.special import gamma as Gamma
        m = self.dim
        return float(Gamma(m / 2.0) / (np.sqrt(np.pi) * Gamma((m + 1) / 2.0)))


class RandomVariateGenerator:

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)


    def uniform(self, a: float = 0.0, b: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        return self.rng.uniform(a, b, size=size)

    def normal(self, mu: float = 0.0, sigma: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        return self.rng.normal(mu, sigma, size=size)

    def exponential(self, lam: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        if lam <= 0:
            raise ValueError("λ 必须 > 0")
        return self.rng.exponential(1.0 / lam, size=size)

    def gamma(self, shape: float, scale: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        if shape <= 0 or scale <= 0:
            raise ValueError("shape 和 scale 必须 > 0")
        return self.rng.gamma(shape, scale, size=size)

    def beta(self, a: float, b: float, size: Optional[Tuple] = None) -> np.ndarray:
        if a <= 0 or b <= 0:
            raise ValueError("a 和 b 必须 > 0")
        return self.rng.beta(a, b, size=size)

    def chi_square(self, df: int, size: Optional[Tuple] = None) -> np.ndarray:
        if df <= 0:
            raise ValueError("自由度必须 > 0")
        return self.rng.chisquare(df, size=size)


    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        return self.rng.multivariate_normal(mean, cov, size=size)


class MonteCarloUQ:

    def __init__(self, n_samples: int = 2000, seed: Optional[int] = 42):
        self.n_samples = n_samples
        self.rng = RandomVariateGenerator(seed)
        self.hyper = HypersphereSampler(dim=5, seed=seed)

    def parameter_perturbation(self,
                                base_params: dict,
                                std_params: dict) -> dict:
        perturbed = {}
        for key, base in base_params.items():
            std = std_params.get(key, 0.0)
            if std > 0:
                perturbed[key] = base + self.rng.normal(0.0, std)
            else:
                perturbed[key] = base

        for key in ["R_major", "r_minor", "n_ring", "kappa", "alpha_abs"]:
            if key in perturbed and perturbed[key] <= 0:
                perturbed[key] = base_params[key] * 0.5
        return perturbed

    def run_mc_propagation(self,
                           base_params: dict,
                           std_params: dict,
                           forward_model: Callable[[dict], dict]) -> dict:
        outputs = []
        for _ in range(self.n_samples):
            p = self.parameter_perturbation(base_params, std_params)
            try:
                out = forward_model(p)
                outputs.append(out)
            except Exception:

                continue

        if not outputs:
            raise RuntimeError("所有蒙特卡洛样本均失败")


        scalar_keys = set()
        for out in outputs:
            for k, v in out.items():
                if np.isscalar(v):
                    scalar_keys.add(k)

        summary = {"n_success": len(outputs), "n_requested": self.n_samples}
        for k in scalar_keys:
            vals = np.array([o[k] for o in outputs if k in o])
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }
        return summary

    def sample_on_hypersphere(self, radius: float, n: int) -> np.ndarray:
        pts = self.hyper.sample(n)

        return radius * pts
